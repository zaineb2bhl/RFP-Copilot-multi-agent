"""Tests for the MCP server and the optional Langfuse tracing."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp.exceptions import ToolError

from rfp_copilot.config import Settings, runs_db_url
from rfp_copilot.db.repository import Database
from rfp_copilot.llm import MockLLM, build_llm
from rfp_copilot.mcp_server.server import create_server
from rfp_copilot.models import Requirement
from rfp_copilot.retrieval.retriever import HybridRetriever
from rfp_copilot.tracing import LangfuseTracer, Tracer, get_tracer
from tests.conftest import KB_DIR


def _call(server: Any, tool: str, args: dict[str, Any]) -> Any:
    _content, structured = asyncio.run(server.call_tool(tool, args))
    # FastMCP wraps non-object results (lists) as {"result": ...}; dicts come back as-is.
    return structured["result"] if set(structured) == {"result"} else structured


# -- MCP: in-process -------------------------------------------------------------------


def test_mcp_lists_the_three_tools(settings: Settings, retriever: HybridRetriever) -> None:
    server = create_server(settings, retriever=retriever)
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert set(tools) == {"search_past_projects", "get_requirement_status", "list_runs"}
    assert "query" in tools["search_past_projects"].inputSchema["properties"]


def test_mcp_search_uses_pipeline_retrieval(settings: Settings, retriever: HybridRetriever) -> None:
    server = create_server(settings, retriever=retriever)
    results = _call(server, "search_past_projects", {"query": "ISO 27001 certification", "k": 3})
    assert len(results) == 3
    assert {"chunk_id", "source", "title", "text", "score"} <= set(results[0])
    expected = [p.chunk_id for p in retriever.search("ISO 27001 certification", k=3)]
    assert [r["chunk_id"] for r in results] == expected
    with pytest.raises(ToolError):
        _call(server, "search_past_projects", {"query": "  "})


def test_mcp_runs_and_requirement_status(settings: Settings, retriever: HybridRetriever) -> None:
    db = Database(runs_db_url(settings))
    db.create_run("run1", "rfp.pdf")
    req = Requirement(id="REQ-001", text="Must X.", type="mandatory", source_page=1)
    db.replace_requirements("run1", [req])
    server = create_server(settings, retriever=retriever, db=db)
    assert [r["id"] for r in _call(server, "list_runs", {})] == ["run1"]
    status = _call(server, "get_requirement_status", {"run_id": "run1"})
    assert status["summary"] == {"pending": 1}
    assert status["requirements"][0]["id"] == "REQ-001"
    with pytest.raises(ToolError, match="not found"):
        _call(server, "get_requirement_status", {"run_id": "nope"})


# -- MCP: real stdio subprocess ----------------------------------------------------------


def test_mcp_server_over_stdio(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "LLM_PROVIDER": "mock",
        "EMBEDDING_PROVIDER": "mock",
        "KNOWLEDGE_BASE_DIR": str(KB_DIR),
        "CHROMA_DIR": str(tmp_path / "chroma"),
        "RUNS_DIR": str(tmp_path / "runs"),
        "LLM_CACHE_ENABLED": "false",
        "MIN_RELEVANCE": "0",  # hash vectors: the nomic-calibrated floor does not apply
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rfp_copilot.mcp_server", "--project-dir", str(tmp_path)],
        env=env,
    )

    async def scenario() -> tuple[list[str], list[dict[str, Any]], list[Any]]:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            search = await session.call_tool("search_past_projects", {"query": "SLA", "k": 2})
            runs = await session.call_tool("list_runs", {})
            return tools, json.loads(search.content[0].text), runs.structuredContent["result"]

    tools, first_hit, runs = asyncio.run(asyncio.wait_for(scenario(), timeout=120))
    assert sorted(tools) == ["get_requirement_status", "list_runs", "search_past_projects"]
    assert "chunk_id" in first_hit
    assert runs == []


# -- tracing ---------------------------------------------------------------------------


class FakeObservation:
    def __init__(self, log: list, kind: str, kwargs: dict) -> None:
        self.log, self.record = log, {"kind": kind, **kwargs}
        log.append(self.record)

    def update(self, **kwargs: Any) -> None:
        self.record.setdefault("updates", []).append(kwargs)

    def update_trace(self, **kwargs: Any) -> None:
        self.record["trace"] = kwargs


class FakeLangfuse:
    """Records what the tracer sends, shaped like the Langfuse v3 client."""

    def __init__(self) -> None:
        self.log: list[dict[str, Any]] = []
        self.flushed = 0

    @contextmanager
    def start_as_current_span(self, **kwargs: Any):
        yield FakeObservation(self.log, "span", kwargs)

    @contextmanager
    def start_as_current_generation(self, **kwargs: Any):
        yield FakeObservation(self.log, "generation", kwargs)

    def flush(self) -> None:
        self.flushed += 1


class BrokenLangfuse(FakeLangfuse):
    def start_as_current_generation(self, **kwargs: Any):
        raise RuntimeError("langfuse is down")


def test_tracing_disabled_without_keys() -> None:
    assert type(get_tracer(Settings())) is Tracer
    assert type(get_tracer(Settings(langfuse_public_key="pk"))) is Tracer  # secret missing


def test_llm_calls_are_traced_with_usage() -> None:
    fake = FakeLangfuse()
    llm = MockLLM(["x" * 40], tracer=LangfuseTracer(fake))
    with llm.tracer.span("run", session_id="r1"):
        llm.complete([{"role": "user", "content": "hi"}], tag="write")
    span, generation = fake.log
    assert span["name"] == "run" and span["trace"]["session_id"] == "r1"
    assert generation["name"] == "write" and generation["model"] == "mock-model"
    update = generation["updates"][0]
    assert update["output"] == "x" * 40
    assert update["usage_details"] == {"input": 0, "output": 10}
    assert update["metadata"]["cached"] is False


def test_tracing_failures_never_break_calls() -> None:
    llm = MockLLM(["still works"], tracer=LangfuseTracer(BrokenLangfuse()))
    assert llm.complete([{"role": "user", "content": "hi"}]).content == "still works"


def test_build_llm_uses_noop_tracer_by_default() -> None:
    llm = build_llm(Settings(llm_provider="mock", llm_cache_enabled=False))
    assert type(llm.tracer) is Tracer
