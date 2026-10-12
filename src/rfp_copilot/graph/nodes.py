"""Graph nodes. Each node takes the state and returns a partial update.

LLM-dependent nodes receive their clients through `PipelineDeps`, so tests can inject
`MockLLM`. Routing functions are plain deterministic code.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langgraph.types import interrupt

from rfp_copilot.config import Settings
from rfp_copilot.export.docx_exporter import build_compliance_rows, export_docx
from rfp_copilot.graph.state import RFPState
from rfp_copilot.ingestion.pdf_parser import parse_pdf
from rfp_copilot.ingestion.requirements import extract_requirements, group_requirements
from rfp_copilot.llm import BaseLLMClient, StructuredOutputError
from rfp_copilot.models import (
    ParsedDocument,
    Requirement,
    ResponseGroup,
    RetrievedPassage,
    ReviewResult,
    extract_citations,
)
from rfp_copilot.retrieval.retriever import HybridRetriever
from rfp_copilot.tracing import Tracer

logger = logging.getLogger(__name__)

_LEADING_HEADING_RE = re.compile(r"\A\s*(#{1,6}[^\n]*\n+)+")

WRITER_SYSTEM_PROMPT = """You write one section of a technical proposal answering an RFP.

Rules:
- Use ONLY facts found in the provided knowledge-base passages. Never invent clients,
  figures, certifications, people or dates.
- Cite every factual statement with the id of the supporting passage in square brackets,
  exactly as given, e.g. [proj-city-portal#2]. Several citations are allowed.
- Address EVERY listed requirement explicitly and mention its id in bold, e.g. **REQ-004**.
- Requirements of type "administrative" (submission format, deadlines, forms) only need a
  one-line statement of compliance each, without citation. Do not mention administrative
  requirements when none are listed.
- If the passages do not support an answer for a requirement, say so plainly
  ("to be confirmed by the bid team") instead of guessing.
- Write in a professional, concise tone, in markdown (paragraphs and bullet lists).
  Do not add any title or heading for the section. Output only the section text."""

REVIEWER_SYSTEM_PROMPT = """You are a strict reviewer of proposal drafts (LLM-as-judge).
Given the RFP requirements, the knowledge-base passages and a draft section, score:
- coverage: fraction (0 to 1) of the requirements the draft fully and explicitly answers;
- groundedness: fraction (0 to 1) of the draft's factual claims supported by the cited passages.
Administrative requirements (submission format, deadlines, forms) are answered by a clear
statement of compliance and need no citation; a compliance statement is not an unsupported
claim. List the ids of requirements not fully answered, quote unsupported claims, and give
short, concrete feedback the writer can act on."""


@dataclass
class PipelineDeps:
    """Everything the nodes need besides the state.

    Clients may be None for a graph compiled only to inspect checkpointed state.
    """

    settings: Settings
    writer: BaseLLMClient | None
    reviewer: BaseLLMClient | None
    retriever: HybridRetriever | None = None
    tracer: Tracer = field(default_factory=Tracer)

    def require_writer(self) -> BaseLLMClient:
        """The writer client (raises if this graph was built for inspection only)."""
        if self.writer is None:
            raise RuntimeError("No writer LLM configured")
        return self.writer

    def require_reviewer(self) -> BaseLLMClient:
        """The reviewer client (raises if this graph was built for inspection only)."""
        if self.reviewer is None:
            raise RuntimeError("No reviewer LLM configured")
        return self.reviewer


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def citations(text: str) -> list[str]:
    """Citation ids found in `text`, in order of first appearance."""
    return extract_citations(text)


def clean_draft(text: str) -> str:
    """Drop markdown headings the model put before the body (the DOCX adds its own title)."""
    return _LEADING_HEADING_RE.sub("", text.strip()).strip()


def groups_needing_revision(state: RFPState, threshold: float) -> list[str]:
    """Ids of groups whose latest review scores below `threshold`."""
    return [
        group_id
        for group_id, review in state.get("reviews", {}).items()
        if ReviewResult.model_validate(review).score < threshold
    ]


def route_after_review(
    state: RFPState, *, threshold: float, max_iterations: int
) -> Literal["write", "human_review"]:
    """Back to `write` if any section is below threshold and passes remain, else human review."""
    failing = groups_needing_revision(state, threshold)
    if failing and state.get("iteration", 0) < max_iterations:
        return "write"
    return "human_review"


def route_after_human(state: RFPState) -> Literal["export_docx", "__end__"]:
    """Export only approved runs."""
    return "export_docx" if state.get("outcome") == "approved" else "__end__"


def apply_deterministic_checks(
    review: ReviewResult, draft: str, requirements: list[Requirement], passage_ids: set[str]
) -> ReviewResult:
    """Tighten an LLM review with checks that need no judgement.

    * requirement ids the draft never mentions are uncovered;
    * citations that match no retrieved passage are unsupported claims, and cap groundedness
      at 0.5; a draft with no citation at all gets groundedness 0, unless every requirement
      is administrative (format, deadline...), which a compliance statement answers;
    * uncovered ids outside this section are dropped; coverage is capped by the share of
      requirements actually mentioned.
    """
    group_ids = [r.id for r in requirements]
    uncovered = [i for i in review.uncovered_requirements if i in group_ids]
    uncovered += [i for i in group_ids if i not in draft and i not in uncovered]
    cited = citations(draft)
    invalid = [c for c in cited if c not in passage_ids]
    unsupported = list(review.unsupported_claims) + [
        f"Citation [{c}] does not match any retrieved passage" for c in invalid
    ]
    groundedness = review.groundedness
    needs_sources = any(r.type != "administrative" for r in requirements)
    if not cited and needs_sources:
        groundedness = 0.0
    elif invalid:
        groundedness = min(groundedness, 0.5)
    mentioned_share = (
        1 - (sum(i not in draft for i in group_ids) / len(group_ids)) if group_ids else 1
    )
    return review.model_copy(
        update={
            "coverage": min(review.coverage, mentioned_share),
            "groundedness": groundedness,
            "uncovered_requirements": uncovered,
            "unsupported_claims": unsupported,
        }
    )


def _requirements_block(requirements: list[Requirement]) -> str:
    return "\n".join(f"- {r.id} ({r.type}, RFP p.{r.source_page}): {r.text}" for r in requirements)


def _passages_block(passages: list[RetrievedPassage]) -> str:
    if not passages:
        return "(no passages retrieved)"
    return "\n\n".join(f"[{p.chunk_id}] ({p.source} - {p.title})\n{p.text}" for p in passages)


def _group_context(
    state: RFPState, group_id: str
) -> tuple[ResponseGroup, list[Requirement], list[RetrievedPassage]]:
    group = next(ResponseGroup.model_validate(g) for g in state["groups"] if g["id"] == group_id)
    by_id = {r["id"]: Requirement.model_validate(r) for r in state["requirements"]}
    requirements = [by_id[i] for i in group.requirement_ids]
    passages = [
        RetrievedPassage.model_validate(p) for p in state.get("passages", {}).get(group_id, [])
    ]
    return group, requirements, passages


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def ingest(state: RFPState) -> dict[str, Any]:
    """Parse the RFP PDF into sections with page numbers."""
    document = parse_pdf(state["rfp_path"])
    logger.info(
        "Parsed %s: %d pages, %d sections",
        document.source,
        document.page_count,
        len(document.sections),
    )
    return {"document": document.model_dump(), "iteration": 0, "review_history": []}


def make_extract_node(deps: PipelineDeps):
    """Node: extract requirements and group them into response sections."""

    def extract_requirements_node(state: RFPState) -> dict[str, Any]:
        document = ParsedDocument.model_validate(state["document"])
        requirements = extract_requirements(
            document, deps.require_writer(), batch_chars=deps.settings.extraction_batch_chars
        )
        groups = group_requirements(
            requirements, document, max_per_group=deps.settings.max_requirements_per_section
        )
        logger.info("Extracted %d requirements in %d groups", len(requirements), len(groups))
        return {
            "requirements": [r.model_dump() for r in requirements],
            "groups": [g.model_dump() for g in groups],
        }

    return extract_requirements_node


def make_retrieve_node(deps: PipelineDeps):
    """Node: hybrid retrieval, one fused search per group (one query per requirement)."""

    def retrieve(state: RFPState) -> dict[str, Any]:
        if deps.retriever is None:
            raise RuntimeError("No retriever configured")
        by_id = {r["id"]: r for r in state["requirements"]}
        passages: dict[str, list[dict[str, Any]]] = {}
        for group in state["groups"]:
            queries = [by_id[i]["text"] for i in group["requirement_ids"]]
            results = deps.retriever.search_many(queries, k=deps.settings.retrieval_k)
            passages[group["id"]] = [p.model_dump() for p in results]
        return {"passages": passages}

    return retrieve


def make_write_node(deps: PipelineDeps):
    """Node: draft every section on the first pass, then only those the reviewer failed."""

    def write(state: RFPState) -> dict[str, Any]:
        iteration = state.get("iteration", 0)
        if iteration == 0:
            targets = [g["id"] for g in state["groups"]]
        else:
            targets = groups_needing_revision(state, deps.settings.review_threshold)
        drafts = dict(state.get("drafts", {}))
        for group_id in targets:
            group, requirements, passages = _group_context(state, group_id)
            user = (
                f"Response section: {group.title}\n\n"
                f"Requirements to answer:\n{_requirements_block(requirements)}\n\n"
                f"Knowledge-base passages:\n{_passages_block(passages)}"
            )
            review = state.get("reviews", {}).get(group_id)
            if iteration > 0 and review and group_id in drafts:
                r = ReviewResult.model_validate(review)
                uncovered = ", ".join(r.uncovered_requirements) or "none"
                user += (
                    f"\n\nYour previous draft:\n{drafts[group_id]}\n\n"
                    f"Reviewer feedback (coverage {r.coverage:.2f}, "
                    f"groundedness {r.groundedness:.2f}):\n"
                    f"{r.feedback}\n"
                    f"Requirements not fully answered: {uncovered}\n"
                    f"Unsupported claims: {'; '.join(r.unsupported_claims) or 'none'}\n"
                    "Rewrite the section, fixing every point above."
                )
            response = deps.require_writer().complete(
                [
                    {"role": "system", "content": WRITER_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                tag="write",
            )
            drafts[group_id] = clean_draft(response.content)
        return {"drafts": drafts, "pending_groups": targets, "iteration": iteration + 1}

    return write


def make_review_node(deps: PipelineDeps):
    """Node: LLM-as-judge on the sections written in the latest pass, plus hard checks."""

    def review(state: RFPState) -> dict[str, Any]:
        reviews = dict(state.get("reviews", {}))
        for group_id in state.get("pending_groups", []):
            group, requirements, passages = _group_context(state, group_id)
            draft = state["drafts"][group_id]
            user = (
                f"Requirements:\n{_requirements_block(requirements)}\n\n"
                f"Knowledge-base passages:\n{_passages_block(passages)}\n\n"
                f"Draft section '{group.title}':\n{draft}"
            )
            try:
                result = deps.require_reviewer().complete_structured(
                    [
                        {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    ReviewResult,
                    temperature=0.0,
                    tag="review",
                )
            except StructuredOutputError as exc:
                logger.error("Review of %s failed validation: %s", group_id, exc)
                result = ReviewResult(
                    coverage=0.0,
                    groundedness=0.0,
                    feedback="The automatic review failed; please review this section manually.",
                )
            checked = apply_deterministic_checks(
                result, draft, requirements, {p.chunk_id for p in passages}
            )
            reviews[group_id] = checked.model_dump()
        history = list(state.get("review_history", []))
        history.append({g: ReviewResult.model_validate(r).score for g, r in reviews.items()})
        return {"reviews": reviews, "review_history": history}

    return review


def human_review(state: RFPState) -> dict[str, Any]:
    """Pause for a human decision (LangGraph interrupt), then apply their edits.

    Resume with `Command(resume={"action": "approve" | "reject", "edits": {group_id: text}})`.
    """
    decision = interrupt(
        {
            "type": "human_review",
            "run_id": state.get("run_id"),
            "groups": [g["id"] for g in state.get("groups", [])],
        }
    )
    decision = decision if isinstance(decision, dict) else {}
    if decision.get("action") != "approve":
        return {"outcome": "rejected"}
    drafts = dict(state.get("drafts", {}))
    for group_id, text in (decision.get("edits") or {}).items():
        if group_id in drafts and isinstance(text, str) and text.strip():
            drafts[group_id] = text.strip()
    return {"outcome": "approved", "drafts": drafts}


def make_export_node(deps: PipelineDeps):
    """Node: write the DOCX (response sections + compliance table + sources)."""

    def export(state: RFPState) -> dict[str, Any]:
        document = ParsedDocument.model_validate(state["document"])
        requirements = [Requirement.model_validate(r) for r in state.get("requirements", [])]
        groups = [ResponseGroup.model_validate(g) for g in state.get("groups", [])]
        reviews = {k: ReviewResult.model_validate(v) for k, v in state.get("reviews", {}).items()}
        passages = {
            p["chunk_id"]: RetrievedPassage.model_validate(p)
            for plist in state.get("passages", {}).values()
            for p in plist
        }
        drafts = state.get("drafts", {})
        rows = build_compliance_rows(requirements, groups, drafts, reviews)
        out_dir = deps.settings.runs_dir / "outputs"
        path = out_dir / f"{state.get('run_id') or Path(document.source).stem}.docx"
        export_docx(
            path,
            rfp_title=document.title,
            groups=groups,
            drafts=drafts,
            compliance=rows,
            passages=passages,
        )
        return {"output_path": str(path)}

    return export
