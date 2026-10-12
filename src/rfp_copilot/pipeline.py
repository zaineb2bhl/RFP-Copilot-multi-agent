"""Run orchestration shared by the API, the CLI and (later) the MCP server.

`PipelineService` owns the SQLite checkpointer, the application database and a lazily built
retriever. `execute` and `resume` are blocking and meant to run in a background task.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from rfp_copilot.config import LLMRole, Settings, runs_db_url
from rfp_copilot.db.repository import Database, RunRecord
from rfp_copilot.export.docx_exporter import build_compliance_rows
from rfp_copilot.graph.graph import build_graph
from rfp_copilot.graph.nodes import PipelineDeps
from rfp_copilot.llm import BaseLLMClient, UsageTracker, build_llm
from rfp_copilot.models import Requirement, ResponseGroup, ReviewResult
from rfp_copilot.retrieval.retriever import HybridRetriever, build_retriever
from rfp_copilot.tracing import Tracer, get_tracer

logger = logging.getLogger(__name__)

LLMFactory = Callable[[LLMRole, UsageTracker], BaseLLMClient]


class ReviewDecision(BaseModel):
    """A human decision at the review step."""

    action: Literal["approve", "reject"]
    edits: dict[str, str] = Field(default_factory=dict, description="group id -> edited text")


def _merge_usage(previous: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    keys = set(previous) | set(new)
    return {k: round(previous.get(k, 0) + new.get(k, 0), 6) for k in sorted(keys)}


class PipelineService:
    """Create, execute, inspect and resume pipeline runs."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm_factory: LLMFactory | None = None,
        retriever: HybridRetriever | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.settings = settings
        self.tracer = tracer or get_tracer(settings)
        settings.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(runs_db_url(settings))
        connection = sqlite3.connect(
            settings.runs_dir / "checkpoints.sqlite", check_same_thread=False
        )
        self.checkpointer = SqliteSaver(connection)
        self._llm_factory = llm_factory or (
            lambda role, usage: build_llm(settings, role, usage=usage)
        )
        self._retriever = retriever
        self._retriever_lock = threading.Lock()

    # -- dependencies -----------------------------------------------------------------------

    @property
    def retriever(self) -> HybridRetriever:
        """The hybrid retriever, built (and the index synced) on first use."""
        with self._retriever_lock:
            if self._retriever is None:
                self._retriever = build_retriever(self.settings)
            return self._retriever

    def _graph(self, usage: UsageTracker | None):
        """Compile the graph; with `usage=None` it can only inspect state (no LLM, no index)."""
        deps = PipelineDeps(
            settings=self.settings,
            writer=self._llm_factory("writer", usage) if usage else None,
            reviewer=self._llm_factory("reviewer", usage) if usage else None,
            retriever=self.retriever if usage else None,
            tracer=self.tracer,
        )
        return build_graph(deps, self.checkpointer)

    @staticmethod
    def _config(run_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": run_id}}

    # -- run lifecycle ----------------------------------------------------------------------

    def create_run(self, filename: str, pdf_bytes: bytes) -> RunRecord:
        """Store the uploaded PDF and register a queued run."""
        run_id = uuid.uuid4().hex[:12]
        upload = self.upload_path(run_id)
        upload.parent.mkdir(parents=True, exist_ok=True)
        upload.write_bytes(pdf_bytes)
        return self.db.create_run(run_id, Path(filename).name)

    def upload_path(self, run_id: str) -> Path:
        """Where the run's RFP PDF is stored."""
        return self.settings.runs_dir / "uploads" / f"{run_id}.pdf"

    def execute(self, run_id: str) -> RunRecord:
        """Run the graph from the start until the human-review interrupt (or failure)."""
        initial = {"run_id": run_id, "rfp_path": str(self.upload_path(run_id))}
        return self._drive(run_id, initial)

    def resume(self, run_id: str, decision: ReviewDecision) -> RunRecord:
        """Resume a run waiting for human review with the given decision."""
        if "human_review" not in self.get_snapshot(run_id).next:
            raise ValueError(f"Run {run_id} is not awaiting review")
        return self._drive(run_id, Command(resume=decision.model_dump()))

    def _drive(self, run_id: str, graph_input: Any) -> RunRecord:
        usage = UsageTracker()
        self.db.update_run(run_id, status="running", error=None)
        try:
            graph = self._graph(usage)
            phase = "resume" if isinstance(graph_input, Command) else "start"
            with self.tracer.span(
                f"rfp-run-{phase}", metadata={"run_id": run_id}, session_id=run_id
            ):
                config = self._config(run_id)
                for update in graph.stream(graph_input, config, stream_mode="updates"):
                    for node in update:
                        if not node.startswith("__"):
                            self.db.update_run(run_id, stage=node)
                            logger.info("Run %s: finished %s", run_id, node)
        except Exception as exc:  # report any failure on the run instead of losing it
            logger.exception("Run %s failed", run_id)
            self._save_usage(run_id, usage)
            return self.db.update_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            self.tracer.flush()
        self._save_usage(run_id, usage)
        return self._sync_status(run_id)

    def _save_usage(self, run_id: str, usage: UsageTracker) -> None:
        run = self.db.get_run(run_id)
        previous = run.usage if run else {}
        self.db.update_run(run_id, usage=_merge_usage(previous, usage.summary()))

    def _sync_status(self, run_id: str) -> RunRecord:
        """Derive the run status from the checkpointed graph state, and store requirements."""
        snapshot = self.get_snapshot(run_id)
        values = snapshot.values
        requirements = [Requirement.model_validate(r) for r in values.get("requirements", [])]
        compliance = None
        if values.get("output_path"):
            compliance = build_compliance_rows(
                requirements,
                [ResponseGroup.model_validate(g) for g in values.get("groups", [])],
                values.get("drafts", {}),
                {k: ReviewResult.model_validate(v) for k, v in values.get("reviews", {}).items()},
            )
        self.db.replace_requirements(run_id, requirements, compliance)
        if "human_review" in snapshot.next:
            return self.db.update_run(run_id, status="awaiting_review", stage="human_review")
        if values.get("outcome") == "rejected":
            return self.db.update_run(run_id, status="rejected")
        return self.db.update_run(run_id, status="completed", output_path=values.get("output_path"))

    # -- inspection -------------------------------------------------------------------------

    def get_snapshot(self, run_id: str):
        """The latest checkpointed graph state (LangGraph StateSnapshot)."""
        graph = self._graph(None)
        return graph.get_state(self._config(run_id))

    def review_payload(self, run_id: str) -> dict[str, Any]:
        """Everything a reviewer needs: groups, drafts, reviews, requirements, passages."""
        values = self.get_snapshot(run_id).values
        return {
            "iteration": values.get("iteration", 0),
            "review_history": values.get("review_history", []),
            "requirements": values.get("requirements", []),
            "groups": values.get("groups", []),
            "drafts": values.get("drafts", {}),
            "reviews": values.get("reviews", {}),
            "passages": {
                g: [{k: p[k] for k in ("chunk_id", "source", "title", "score")} for p in plist]
                for g, plist in values.get("passages", {}).items()
            },
        }
