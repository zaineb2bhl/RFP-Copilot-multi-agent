"""MCP server (official Python SDK, stdio transport).

Tools:
- search_past_projects(query, k): hybrid search over the knowledge base (same code as the pipeline)
- get_requirement_status(run_id): compliance status of every requirement of a run
- list_runs(limit): recent pipeline runs

Run: `python -m rfp_copilot.mcp_server --project-dir <repo>`. stdout carries the MCP protocol,
so all logging goes to stderr.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from collections import Counter
from typing import Any

from mcp.server.fastmcp import FastMCP

from rfp_copilot.config import Settings, get_settings, runs_db_url
from rfp_copilot.db.repository import Database
from rfp_copilot.retrieval.retriever import HybridRetriever, build_retriever

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "RFP Copilot tools. Use search_past_projects to find evidence in the company knowledge base "
    "(past projects, CVs, methodologies); cite results by their chunk_id. Use list_runs and "
    "get_requirement_status to inspect RFP analyses produced by the pipeline."
)
MAX_K = 20


def create_server(
    settings: Settings | None = None,
    *,
    retriever: HybridRetriever | None = None,
    db: Database | None = None,
) -> FastMCP:
    """Build the MCP server. The retriever is built lazily on the first search."""
    settings = settings or get_settings()
    database = db or Database(runs_db_url(settings))
    lock = threading.Lock()
    cache: dict[str, HybridRetriever] = {"retriever": retriever} if retriever else {}

    def get_retriever() -> HybridRetriever:
        with lock:
            if "retriever" not in cache:
                cache["retriever"] = build_retriever(settings)
            return cache["retriever"]

    mcp = FastMCP("rfp-copilot", instructions=INSTRUCTIONS)

    @mcp.tool()
    def search_past_projects(query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search the knowledge base (past projects, CVs, methodologies).

        Uses the pipeline's hybrid retrieval (BM25 + embeddings fused with RRF). Each result has
        a `chunk_id` usable as a citation like [chunk_id], its source file, title, text and score.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        results = get_retriever().search(query, k=max(1, min(k, MAX_K)))
        return [p.model_dump() for p in results]

    @mcp.tool()
    def get_requirement_status(run_id: str) -> dict[str, Any]:
        """Status of every requirement extracted for a run.

        Status is `pending` until the run is approved and exported, then `addressed`,
        `partially_addressed` or `not_addressed`, with where it is addressed and its sources.
        """
        run = database.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found; call list_runs to see available runs")
        requirements = database.get_requirements(run_id)
        return {
            "run": run.model_dump(mode="json"),
            "summary": dict(Counter(r.status for r in requirements)),
            "requirements": [r.model_dump() for r in requirements],
        }

    @mcp.tool()
    def list_runs(limit: int = 20) -> list[dict[str, Any]]:
        """Most recent pipeline runs (id, RFP file name, status, current stage, timestamps)."""
        runs = database.list_runs(max(1, min(limit, 200)))
        keys = ("id", "rfp_filename", "status", "stage", "created_at", "updated_at", "error")
        return [{k: v for k, v in r.model_dump(mode="json").items() if k in keys} for r in runs]

    return mcp


def main(argv: list[str] | None = None) -> None:
    """Entry point: `python -m rfp_copilot.mcp_server [--project-dir DIR]`."""
    parser = argparse.ArgumentParser(prog="rfp-copilot-mcp", description="RFP Copilot MCP server")
    parser.add_argument(
        "--project-dir",
        help="repo root holding .env, data/, runs/ and chroma_db/ (default: current directory)",
    )
    args = parser.parse_args(argv)
    if args.project_dir:
        os.chdir(args.project_dir)
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "chromadb", "openai", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    create_server().run()  # stdio transport
