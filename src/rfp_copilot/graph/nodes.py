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

from rfp_copilot.compliance import requirement_segments, rows_from_state
from rfp_copilot.config import Settings
from rfp_copilot.export.docx_exporter import export_docx
from rfp_copilot.graph.state import RFPState
from rfp_copilot.ingestion.pdf_parser import parse_pdf
from rfp_copilot.ingestion.requirements import extract_requirements, group_requirements
from rfp_copilot.llm import BaseLLMClient, StructuredOutputError
from rfp_copilot.models import (
    CITATION_GROUP_RE,
    PLACEHOLDER_RE,
    ParsedDocument,
    Requirement,
    ResponseGroup,
    RetrievedPassage,
    ReviewResult,
    SectionAnswers,
    check_section_answers,
    extract_citations,
    placeholder,
)
from rfp_copilot.retrieval.retriever import HybridRetriever
from rfp_copilot.tracing import Tracer

logger = logging.getLogger(__name__)

_LEADING_HEADING_RE = re.compile(r"\A\s*(#{1,6}[^\n]*\n+)+")

WRITER_SYSTEM_PROMPT = """You answer RFP requirements for a technical proposal.
Return exactly one answer per requirement id listed, in the same order.

For each requirement:
- answer: ONLY facts stated in the knowledge-base passages, each fact followed by the id of
  its passage in square brackets, exactly as given, e.g. [hosting-security#2]. Copy figures
  exactly. Do not generalize beyond a passage ("all", "always", "fully"...) and do not merge
  facts from different passages into a new claim. Never invent clients, figures, features,
  certifications, people or dates. Leave the answer empty if no passage supports it.
- If the passages show that we offer LESS than required (for example an RPO of 24 hours when
  4 hours are required), state our documented value honestly with its citation; never
  claim we comply.
- missing_information: if the passages do not fully answer the requirement, name precisely
  what is missing (e.g. "experience integrating with RecTrac"); null if fully answered.
- Never answer several requirements together and never write ranges such as
  "REQ-030 through REQ-050".
- The buyer is named in the prompt. Refer to the buyer only by that name. Clients named in the
  passages are other organisations from past projects: never use their names for the buyer.
- Professional, concise tone. No titles, introductions or summaries."""

REVIEWER_SYSTEM_PROMPT = """You are a strict reviewer of proposal drafts (LLM-as-judge).
Given the RFP requirements, the knowledge-base passages and a draft section, score:
- coverage: fraction (0 to 1) of the requirements the draft fully and explicitly answers;
- groundedness: fraction (0 to 1) of the draft's factual claims supported by the cited passages.
List the ids of requirements not fully answered, quote unsupported claims, and give short,
concrete feedback the writer can act on.
[TO CONFIRM: ...] marks information the knowledge base lacks: it is not a claim, but the
requirement is not fully answered. Administrative requirements (deadlines, copies, forms,
page limits) are bid-team checklist items: ignore them when scoring.
non_compliant: list every requirement where the passages show that what we offer is worse
than what the RFP requires (a longer delay, a lower percentage, a missing certification...),
with what is required and what we offer. Do not count those as covered."""


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


_REPEATED_ID_RE = re.compile(r"(\*\*REQ-\d+\*\*:?\s*)\1+")


def clean_draft(text: str) -> str:
    """Drop headings the model put before the body (the DOCX adds its own title) and
    collapse a requirement id written twice in a row ("**REQ-1**: **REQ-1**:")."""
    text = _LEADING_HEADING_RE.sub("", text.strip()).strip()
    return _REPEATED_ID_RE.sub(r"\1", text)


def groups_needing_revision(state: RFPState, threshold: float) -> list[str]:
    """Ids of groups whose latest review scores below `threshold` and that a rewrite can fix.

    A section with no relevant passage is skipped: without evidence, a rewrite can only
    repeat "[TO CONFIRM]" (or invent), so it would waste calls.
    """
    passages = state.get("passages", {})
    return [
        group_id
        for group_id, review in state.get("reviews", {}).items()
        if ReviewResult.model_validate(review).score < threshold and passages.get(group_id)
    ]


_FIGURE_RE = re.compile(r"\d+(?:[.,]\d+)*\s?%?")
_REQ_ID_RE = re.compile(r"REQ-\d+")
# Claims end with their citation ("... [src#1]."), so split after each citation and at line
# breaks, not at every period: abbreviations such as "U.S." would cut a claim from its source.
_SENTENCE_RE = re.compile(r"(?<=\])[.;]?\s+|\n+")


def _figures(text: str) -> set[str]:
    return {m.replace(",", "").replace(" ", "").rstrip(".") for m in _FIGURE_RE.findall(text)}


def unsupported_figures(draft: str, passages: dict[str, str], requirement_text: str) -> list[str]:
    """Sentences stating a figure that is not in the passages they cite.

    Figures that come from the RFP itself (requirement text) are allowed. A figure in a
    sentence without any citation counts as unsupported. This catches the most damaging
    hallucinations (invented numbers, dates, SLAs) without any LLM call.
    """
    allowed = _figures(requirement_text)
    issues = []
    for sentence in _SENTENCE_RE.split(draft):
        cited = citations(sentence)
        claim = CITATION_GROUP_RE.sub(" ", _REQ_ID_RE.sub(" ", sentence))
        figures = _figures(claim) - allowed
        if not figures:
            continue
        source = " ".join(passages.get(c, "") for c in cited)
        missing = sorted(figures - _figures(source))
        if missing:
            issues.append(
                f"Figure(s) {', '.join(missing)} not found in the cited sources: "
                f'"{sentence.strip()[:140]}"'
            )
    return issues


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


_BUYER_NOUN_RE = re.compile(r"\bthe (City|Town|Village|County|Township|Borough)\b")


def buyer_name_issues(draft: str, issuer: str | None, passages: dict[str, str]) -> list[str]:
    """Sentences calling the buyer by the wrong kind of entity ("the Town" for a City).

    A sentence is only flagged when none of its cited passages uses that word, so references
    to past clients ("the County's auditor [proj-cedar-county#2]") are not flagged.
    """
    if not issuer:
        return []
    noun = issuer.split()[0].capitalize()
    issues = []
    for sentence in _SENTENCE_RE.split(draft):
        for wrong in set(_BUYER_NOUN_RE.findall(sentence)) - {noun}:
            sources = " ".join(passages.get(c, "") for c in citations(sentence))
            if wrong not in sources:
                issues.append(
                    f'The buyer is {issuer}, not "the {wrong}": "{sentence.strip()[:140]}"'
                )
    return issues


def apply_deterministic_checks(
    review: ReviewResult,
    draft: str,
    requirements: list[Requirement],
    passages: dict[str, str],
    issuer: str | None = None,
) -> ReviewResult:
    """Tighten an LLM review with checks that need no judgement.

    `passages` maps each retrieved chunk id to its text.

    * requirement ids without their own answer, or answered only by a [TO CONFIRM] placeholder
      with no citation, are uncovered;
    * citations that match no retrieved passage, figures absent from the cited passages, and
      a wrong name for the buyer are unsupported claims and cap groundedness at 0.5;
    * a draft with no citation at all gets groundedness 0, unless every requirement is
      administrative (bid-team checklist items);
    * uncovered ids and non-compliance findings outside this section are dropped; coverage is
      capped by the share of requirements actually answered.
    """
    group_ids = [r.id for r in requirements]
    segments = requirement_segments(draft)
    unanswered = [
        r.id
        for r in requirements
        if r.type != "administrative"
        and (
            r.id not in segments
            or (PLACEHOLDER_RE.search(segments[r.id]) and not citations(segments[r.id]))
        )
    ]
    uncovered = [i for i in review.uncovered_requirements if i in group_ids]
    uncovered += [i for i in unanswered if i not in uncovered]
    cited = citations(draft)
    invalid = [c for c in cited if c not in passages]
    figure_issues = unsupported_figures(draft, passages, " ".join(r.text for r in requirements))
    name_issues = buyer_name_issues(draft, issuer, passages)
    unsupported = (
        list(review.unsupported_claims)
        + [f"Citation [{c}] does not match any retrieved passage" for c in invalid]
        + figure_issues
        + name_issues
    )
    groundedness = review.groundedness
    needs_sources = any(r.type != "administrative" for r in requirements)
    if not cited and needs_sources:
        groundedness = 0.0
    elif invalid or figure_issues or name_issues:
        groundedness = min(groundedness, 0.5)
    answerable = [r.id for r in requirements if r.type != "administrative"]
    answered_share = 1 - len(unanswered) / len(answerable) if answerable else 1.0
    return review.model_copy(
        update={
            "coverage": min(review.coverage, answered_share),
            "groundedness": groundedness,
            "uncovered_requirements": uncovered,
            "unsupported_claims": unsupported,
            "non_compliant": [n for n in review.non_compliant if n.requirement_id in group_ids],
        }
    )


def _requirements_block(requirements: list[Requirement]) -> str:
    return "\n".join(f"- {r.id} ({r.type}, RFP p.{r.source_page}): {r.text}" for r in requirements)


def _passages_block(passages: list[RetrievedPassage]) -> str:
    if not passages:
        return "(no passages retrieved)"
    return "\n\n".join(f"[{p.chunk_id}] ({p.source} - {p.title})\n{p.text}" for p in passages)


def writer_user_prompt(
    title: str,
    requirements: list[Requirement],
    passages: list[RetrievedPassage],
    issuer: str | None = None,
) -> str:
    """The writer's first-pass prompt for one section (also used by the RAG evaluation)."""
    buyer = issuer or "the buyer (the organisation that issued this RFP)"
    return (
        f"Buyer: {buyer}\n"
        f"Response section: {title}\n\n"
        f"Requirements to answer (one answer each):\n{_requirements_block(requirements)}\n\n"
        f"Knowledge-base passages:\n{_passages_block(passages)}"
    )


def administrative_answer(requirement: Requirement) -> str:
    """Checklist line for an administrative item: only the bid team can confirm it."""
    return f"- **{requirement.id}**: {placeholder(f'bid team check: {requirement.text}')}"


def write_section(
    writer: BaseLLMClient,
    title: str,
    requirements: list[Requirement],
    passages: list[RetrievedPassage],
    *,
    issuer: str | None = None,
    revision_notes: str = "",
) -> str:
    """Draft one section: one validated answer per requirement, rendered as markdown bullets.

    Administrative requirements never go to the LLM (they become checklist lines). If the
    writer never returns valid answers, each requirement gets a placeholder rather than
    failing the run.
    """
    lines = {r.id: administrative_answer(r) for r in requirements if r.type == "administrative"}
    answerable = [r for r in requirements if r.type != "administrative"]
    if answerable:
        expected = [r.id for r in answerable]
        user = writer_user_prompt(title, answerable, passages, issuer) + revision_notes
        try:
            result = writer.complete_structured(
                [
                    {"role": "system", "content": WRITER_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                SectionAnswers,
                tag="write",
                check=lambda answers: check_section_answers(answers, expected),
            )
            lines.update({a.requirement_id: a.render() for a in result.answers})
        except StructuredOutputError as exc:
            logger.error("Writer output for '%s' never validated: %s", title, exc)
            failed = placeholder("the answer could not be generated; write it manually")
            lines.update({r.id: f"- **{r.id}**: {failed}" for r in answerable})
    return "\n".join(lines[r.id] for r in requirements)


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


def section_passages(
    retriever: HybridRetriever,
    queries: dict[str, str],
    *,
    budget: int,
    minimum: int,
    maximum: int,
) -> tuple[list[RetrievedPassage], dict[str, list[dict[str, Any]]]]:
    """Passages for a section, with a guaranteed share for every requirement.

    `queries` maps requirement id -> requirement text. Each requirement gets its own top
    `max(minimum, budget // len(queries))` passages, merged in requirement order without
    duplicates and capped at `maximum`: a single fused search would let well-matched
    requirements crowd out the others' evidence. Also returns, per requirement, the chunks
    found for it with their similarity (used to audit "to confirm" answers).
    """
    per_query = max(minimum, budget // max(1, len(queries)))
    merged: dict[str, RetrievedPassage] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    for req_id, query in queries.items():
        found = retriever.search(query, k=per_query)
        evidence[req_id] = [{"chunk_id": p.chunk_id, "similarity": p.similarity} for p in found]
        for passage in found:
            merged.setdefault(passage.chunk_id, passage)
    return list(merged.values())[:maximum], evidence


def make_retrieve_node(deps: PipelineDeps):
    """Node: hybrid retrieval per requirement, merged per response section."""

    def retrieve(state: RFPState) -> dict[str, Any]:
        if deps.retriever is None:
            raise RuntimeError("No retriever configured")
        settings = deps.settings
        by_id = {r["id"]: r for r in state["requirements"]}
        passages: dict[str, list[dict[str, Any]]] = {}
        evidence: dict[str, list[dict[str, Any]]] = {}
        for group in state["groups"]:
            # Administrative items are checklist lines: no evidence needed.
            queries = {
                i: by_id[i]["text"]
                for i in group["requirement_ids"]
                if by_id[i]["type"] != "administrative"
            }
            if not queries:
                passages[group["id"]] = []
                continue
            results, found = section_passages(
                deps.retriever,
                queries,
                budget=settings.retrieval_k,
                minimum=settings.retrieval_min_per_requirement,
                maximum=settings.retrieval_max_per_section,
            )
            passages[group["id"]] = [p.model_dump() for p in results]
            evidence.update(found)
        return {"passages": passages, "evidence": evidence}

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
        issuer = state.get("document", {}).get("issuer")
        for group_id in targets:
            group, requirements, passages = _group_context(state, group_id)
            notes = ""
            review = state.get("reviews", {}).get(group_id)
            if iteration > 0 and review and group_id in drafts:
                r = ReviewResult.model_validate(review)
                uncovered = ", ".join(r.uncovered_requirements) or "none"
                notes = (
                    f"\n\nYour previous answers:\n{drafts[group_id]}\n\n"
                    f"Reviewer feedback (coverage {r.coverage:.2f}, "
                    f"groundedness {r.groundedness:.2f}):\n"
                    f"{r.feedback}\n"
                    f"Requirements not fully answered: {uncovered}\n"
                    f"Unsupported claims: {'; '.join(r.unsupported_claims) or 'none'}\n"
                    "Answer again, fixing every point above."
                )
            drafts[group_id] = write_section(
                deps.require_writer(),
                group.title,
                requirements,
                passages,
                issuer=issuer,
                revision_notes=notes,
            )
        return {"drafts": drafts, "pending_groups": targets, "iteration": iteration + 1}

    return write


def make_review_node(deps: PipelineDeps):
    """Node: LLM-as-judge on the sections written in the latest pass, plus hard checks."""

    def review(state: RFPState) -> dict[str, Any]:
        reviews = dict(state.get("reviews", {}))
        for group_id in state.get("pending_groups", []):
            group, requirements, passages = _group_context(state, group_id)
            draft = state["drafts"][group_id]
            if all(r.type == "administrative" for r in requirements):
                # Checklist items only: nothing for an LLM to judge.
                reviews[group_id] = ReviewResult(
                    coverage=1.0, groundedness=1.0, feedback="Bid-team checklist items."
                ).model_dump()
                continue
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
                result,
                draft,
                requirements,
                {p.chunk_id: p.text for p in passages},
                issuer=state.get("document", {}).get("issuer"),
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
        groups = [ResponseGroup.model_validate(g) for g in state.get("groups", [])]
        passages = {
            p["chunk_id"]: RetrievedPassage.model_validate(p)
            for plist in state.get("passages", {}).values()
            for p in plist
        }
        drafts = state.get("drafts", {})
        rows = rows_from_state(state)
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
