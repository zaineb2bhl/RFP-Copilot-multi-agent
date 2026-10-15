"""Evaluation runners. Every item is stored once computed (see `ResultStore`), so runs resume.

All LLM calls go through the project wrapper, so the RPM throttle and the response cache
protect free-tier quotas. Token counts, cost and latency are recorded per item.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rfp_copilot.config import Settings
from rfp_copilot.evaluation.gold import GoldFile, RagQuery, gold_name
from rfp_copilot.evaluation.judges import judge_context_precision, judge_faithfulness
from rfp_copilot.evaluation.metrics import (
    DEFAULT_MATCH_THRESHOLD,
    ExtractionScores,
    mean,
    micro_average,
    score_extraction,
    score_retrieval,
)
from rfp_copilot.evaluation.store import ResultStore
from rfp_copilot.graph.nodes import write_section
from rfp_copilot.ingestion.pdf_parser import parse_pdf
from rfp_copilot.ingestion.requirements import extract_requirements
from rfp_copilot.llm import BaseLLMClient, UsageTracker, build_llm
from rfp_copilot.models import Requirement, RetrievedPassage, extract_citations
from rfp_copilot.pipeline import PipelineService, ReviewDecision
from rfp_copilot.retrieval.retriever import HybridRetriever, build_retriever

logger = logging.getLogger(__name__)

USAGE_KEYS = (
    "calls",
    "cached_calls",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "latency_s",
)


@dataclass
class Judge:
    """A fixed LLM judge (same model across configs) with its own usage tracker."""

    llm: BaseLLMClient
    usage: UsageTracker


def make_judge(settings: Settings) -> Judge:
    """Judge from the base settings (EVAL_JUDGE_*, else the reviewer), fixed across configs."""
    usage = UsageTracker()
    return Judge(build_llm(settings, "judge", usage=usage), usage)


def _usage(summary: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {f"{prefix}{k}": summary.get(k, 0) for k in USAGE_KEYS}
    out[f"{prefix}tokens"] = summary.get("prompt_tokens", 0) + summary.get("completion_tokens", 0)
    return out


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {k: round(after.get(k, 0) - before.get(k, 0), 6) for k in USAGE_KEYS}


def cached_share(rows: list[dict[str, Any]], prefix: str = "") -> float | None:
    """Share of LLM calls served from the cache (their latency is ~0, so say so in reports)."""
    calls = sum(r.get(f"{prefix}calls", 0) or 0 for r in rows)
    cached = sum(r.get(f"{prefix}cached_calls", 0) or 0 for r in rows)
    return cached / calls if calls else None


def model_label(settings: Settings, role: str = "writer") -> str:
    """`provider:model` of a role, without building a client."""
    from rfp_copilot.config import resolve_llm_profile

    try:
        profile = resolve_llm_profile(settings, role)  # type: ignore[arg-type]
    except Exception:  # missing key etc.: still label the row
        return f"{settings.llm_provider}:{settings.llm_model or 'default'}"
    return f"{profile.provider}:{profile.model}"


# ---------------------------------------------------------------------------
# (a) requirement extraction
# ---------------------------------------------------------------------------


def eval_extraction_doc(
    settings: Settings,
    gold_path: Path,
    gold: GoldFile,
    store: ResultStore,
    config: str,
    *,
    root: Path,
    force: bool = False,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """Extract requirements from one annotated RFP and score them (resumable)."""
    doc_name = gold_name(gold_path)
    if not force and (cached := store.get("extraction", config, doc_name)):
        return cached["row"]
    usage = UsageTracker()
    llm = build_llm(settings, "writer", usage=usage)
    start = time.perf_counter()
    document = parse_pdf(gold.rfp_path(root))
    predicted = extract_requirements(document, llm, batch_chars=settings.extraction_batch_chars)
    wall = time.perf_counter() - start
    scores = score_extraction(predicted, gold.requirements, threshold)
    row = {
        "document": doc_name,
        "config": config,
        "model": model_label(settings),
        "pages": document.page_count,
        "gold": len(gold.requirements),
        "predicted": len(predicted),
        **scores.model_dump(),
        **_usage(usage.summary()),
        "wall_s": round(wall, 2),
    }
    store.put(
        "extraction",
        config,
        doc_name,
        data={"row": row, "predictions": [p.model_dump() for p in predicted]},
    )
    return row


def summarize_extraction(rows: list[dict[str, Any]], config: str) -> dict[str, Any]:
    """Micro-averaged scores and mean per-document usage over a config's rows."""
    keys = set(ExtractionScores.model_fields)
    pooled = micro_average(
        ExtractionScores(**{k: v for k, v in r.items() if k in keys}) for r in rows
    )
    return {
        "config": config,
        "model": rows[0]["model"] if rows else "-",
        "documents": len(rows),
        **pooled.model_dump(),
        "tokens_per_doc": mean(r["tokens"] for r in rows),
        "cached_share": cached_share(rows),
        "cost_per_doc": mean(r["cost_usd"] for r in rows),
        "wall_s_per_doc": mean(r["wall_s"] for r in rows),
    }


# ---------------------------------------------------------------------------
# (b) RAG: retrieval metrics (gold sources) + LLM-judged context precision and faithfulness
# ---------------------------------------------------------------------------


def generate_answer(writer: BaseLLMClient, query: str, passages: list[RetrievedPassage]) -> str:
    """Answer one requirement exactly as the pipeline's writer would (same prompts)."""
    requirement = Requirement(id="REQ-001", text=query, type="mandatory", source_page=1)
    return write_section(writer, "Evaluation", [requirement], passages)


def eval_rag_query(
    settings: Settings,
    retriever: HybridRetriever,
    rerank_usage: UsageTracker,
    query: RagQuery,
    store: ResultStore,
    config: str,
    *,
    judge: Judge | None,
    generate: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Retrieve, optionally answer, and score one labelled query (resumable)."""
    if not force and (cached := store.get("rag", config, query.id)):
        return cached["row"]
    k = settings.retrieval_k
    before = rerank_usage.summary()
    start = time.perf_counter()
    passages = retriever.search(query.query, k=k)
    retrieval_s = time.perf_counter() - start
    rerank = _delta(before, rerank_usage.summary())
    row: dict[str, Any] = {
        "query_id": query.id,
        "config": config,
        "k": k,
        **score_retrieval([p.source for p in passages], set(query.relevant_sources)).model_dump(),
        "retrieval_s": round(retrieval_s, 3),
        "rerank_tokens": rerank["prompt_tokens"] + rerank["completion_tokens"],
    }
    judge_before = judge.usage.summary() if judge else {}
    if judge:
        context = judge_context_precision(judge.llm, query.query, passages)
        row["context_precision"] = context["context_precision"]
        row["context_avg_precision"] = context["context_avg_precision"]
    answer = None
    if generate:
        usage = UsageTracker()
        writer = build_llm(settings, "writer", usage=usage)
        start = time.perf_counter()
        answer = generate_answer(writer, query.query, passages)
        row["generation_s"] = round(time.perf_counter() - start, 3)
        row.update(_usage(usage.summary(), prefix="gen_"))
        cited = extract_citations(answer)
        known = {p.chunk_id for p in passages}
        row["citations"] = len(cited)
        row["citation_validity"] = sum(c in known for c in cited) / len(cited) if cited else None
        if judge:
            row.update(judge_faithfulness(judge.llm, answer, passages))
    if judge:
        spent = _delta(judge_before, judge.usage.summary())
        row["judge_tokens"] = spent["prompt_tokens"] + spent["completion_tokens"]
    store.put(
        "rag",
        config,
        query.id,
        data={
            "row": row,
            "query": query.model_dump(),
            "passages": [p.chunk_id for p in passages],
            "answer": answer,
        },
    )
    return row


def summarize_rag(rows: list[dict[str, Any]], config: str) -> dict[str, Any]:
    """Mean of every metric over a config's query rows."""
    metrics = (
        "hit",
        "precision",
        "recall",
        "reciprocal_rank",
        "context_precision",
        "context_avg_precision",
        "faithfulness",
        "citation_validity",
        "retrieval_s",
        "generation_s",
        "gen_tokens",
        "rerank_tokens",
        "judge_tokens",
    )
    return {
        "config": config,
        "queries": len(rows),
        **{m: mean(r.get(m) for r in rows) for m in metrics},
        "gen_cached_share": cached_share(rows, "gen_"),
    }


def build_eval_retriever(settings: Settings) -> tuple[HybridRetriever, UsageTracker]:
    """Retriever for a config, with a tracker for reranker calls."""
    usage = UsageTracker()
    return build_retriever(settings, usage=usage), usage


# ---------------------------------------------------------------------------
# full pipeline (reviewer loop on/off)
# ---------------------------------------------------------------------------


def eval_pipeline_doc(
    settings: Settings,
    gold_path: Path,
    gold: GoldFile,
    store: ResultStore,
    config: str,
    *,
    root: Path,
    judge: Judge | None,
    force: bool = False,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """Run the whole graph (auto-approved) on one RFP and score the final drafts (resumable).

    The pipeline's own reviewer scores are reported, but the comparison metric is the
    independent judge's faithfulness, since a reviewer grading its own loop is biased.
    """
    doc_name = gold_name(gold_path)
    if not force and (cached := store.get("pipeline", config, doc_name)):
        return cached["row"]
    run_settings = settings.model_copy(update={"runs_dir": store.root / "pipeline_runs" / config})
    service = PipelineService(run_settings)
    pdf = gold.rfp_path(root)
    start = time.perf_counter()
    run = service.create_run(pdf.name, pdf.read_bytes())
    run = service.execute(run.id)
    if run.status == "awaiting_review":
        run = service.resume(run.id, ReviewDecision(action="approve"))
    wall = time.perf_counter() - start
    if run.status != "completed":
        raise RuntimeError(f"Pipeline run {run.id} ended as {run.status}: {run.error}")

    values = service.get_snapshot(run.id).values
    requirements = [Requirement.model_validate(r) for r in values.get("requirements", [])]
    reviews = values.get("reviews", {})
    statuses = [r.status for r in service.db.get_requirements(run.id)]
    row: dict[str, Any] = {
        "document": doc_name,
        "config": config,
        "model": model_label(run_settings),
        "requirements": len(requirements),
        "sections": len(values.get("groups", [])),
        "write_passes": values.get("iteration", 0),
        "reviewer_score": mean((r["coverage"] + r["groundedness"]) / 2 for r in reviews.values()),
        "addressed_share": statuses.count("addressed") / len(statuses) if statuses else None,
        "extraction_f1": score_extraction(requirements, gold.requirements, threshold).f1,
        **_usage(run.usage),
        "wall_s": round(wall, 2),
    }
    if judge:
        judge_before = judge.usage.summary()
        claims = supported = 0
        for group_id, draft in values.get("drafts", {}).items():
            passages = [RetrievedPassage.model_validate(p) for p in values["passages"][group_id]]
            verdict = judge_faithfulness(judge.llm, draft, passages)
            claims += verdict["claims"] or 0
            supported += verdict["supported_claims"] or 0
        row["faithfulness"] = supported / claims if claims else None
        spent = _delta(judge_before, judge.usage.summary())
        row["judge_tokens"] = spent["prompt_tokens"] + spent["completion_tokens"]
    store.put("pipeline", config, doc_name, data={"row": row, "run_id": run.id})
    return row


def summarize_pipeline(rows: list[dict[str, Any]], config: str) -> dict[str, Any]:
    """Mean of the pipeline metrics over a config's documents."""
    metrics = (
        "write_passes",
        "reviewer_score",
        "faithfulness",
        "addressed_share",
        "extraction_f1",
        "tokens",
        "cost_usd",
        "wall_s",
    )
    return {
        "config": config,
        "documents": len(rows),
        **{m: mean(r.get(m) for r in rows) for m in metrics},
        "cached_share": cached_share(rows),
    }
