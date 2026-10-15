"""Pydantic models shared across ingestion, retrieval, the graph and export."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# One or more citation ids in square brackets: [a#1] or [a#1, b#2].
CITATION_GROUP_RE = re.compile(r"\[([A-Za-z0-9_.\-]+#\d+(?:\s*[,;]\s*[A-Za-z0-9_.\-]+#\d+)*)\]")


def extract_citations(text: str) -> list[str]:
    """Citation ids found in `text` (`[a#1]` or `[a#1, b#2]`), in order of first appearance."""
    ids = (
        part.strip()
        for group in CITATION_GROUP_RE.findall(text)
        for part in re.split(r"[,;]", group)
    )
    return list(dict.fromkeys(ids))


RequirementType = Literal["mandatory", "desirable", "administrative"]
# addressed: answered from cited evidence; partial: cited evidence + a [TO CONFIRM] gap;
# not_covered: no evidence in the knowledge base (placeholder only); to_confirm: only a human
# can verify (administrative items, commitments without evidence); non_compliant: our
# documented value is worse than what the RFP requires.
ComplianceStatus = Literal["addressed", "partial", "not_covered", "to_confirm", "non_compliant"]

# "[TO CONFIRM: <missing information>]" placeholders (the legacy "to be confirmed" wording is
# still recognised so older drafts keep their status).
PLACEHOLDER_RE = re.compile(r"\[TO CONFIRM:\s*([^\]]*)\]|to be confirmed", re.IGNORECASE)
_RANGE_RE = re.compile(r"REQ-\d+\s*(?:-|–|—|to|through|thru|until)\s*(?:REQ-)?\d+", re.I)


def placeholder(missing: str) -> str:
    """The standard placeholder for information the knowledge base does not provide."""
    return f"[TO CONFIRM: {missing.strip().rstrip('.') or 'supporting evidence'}]"


class Section(BaseModel):
    """A titled span of an RFP, with the pages it covers (1-based)."""

    id: str
    title: str
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class ParsedDocument(BaseModel):
    """An RFP parsed into sections; `page_texts[i]` is the text of page i + 1."""

    source: str
    title: str
    page_texts: list[str]
    sections: list[Section]
    issuer: str | None = None  # e.g. "City of Ballwin", detected from the first pages

    @property
    def page_count(self) -> int:
        """Number of pages in the PDF."""
        return len(self.page_texts)


class ExtractedRequirement(BaseModel):
    """What the LLM returns for one requirement (ids are assigned by code)."""

    text: str = Field(min_length=5, description="The requirement, quoted or closely paraphrased.")
    type: RequirementType = Field(
        description=(
            "mandatory: must/shall/required; desirable: should/may/preferred/nice to have; "
            "administrative: submission format, deadlines, forms, eligibility paperwork."
        )
    )
    source_page: int = Field(ge=1, description="Page number shown in the [page N] marker.")

    @field_validator("type", mode="before")
    @classmethod
    def _normalise_type(cls, value: object) -> object:
        # Small models often vary the case ("Mandatory") or add spaces.
        return value.strip().lower() if isinstance(value, str) else value


class ExtractedRequirements(BaseModel):
    """LLM output schema for requirement extraction."""

    requirements: list[ExtractedRequirement]


class Requirement(ExtractedRequirement):
    """A requirement with a stable id (REQ-001, ...) and the RFP section it belongs to."""

    id: str
    section_id: str | None = None


class Chunk(BaseModel):
    """A knowledge-base passage. `id` is the citation key used as [id] in drafts."""

    id: str
    source: str  # file name inside the knowledge base
    title: str
    text: str


class RetrievedPassage(BaseModel):
    """A chunk returned by retrieval with its fused score."""

    chunk_id: str
    source: str
    title: str
    text: str
    score: float
    similarity: float | None = None  # best cosine similarity to the queries


class ResponseGroup(BaseModel):
    """Requirements answered together in one response section."""

    id: str
    title: str
    requirement_ids: list[str]


class RequirementAnswer(BaseModel):
    """The writer's answer to exactly one requirement."""

    requirement_id: str = Field(pattern=r"^REQ-\d{3,}$", description="One id, e.g. REQ-004.")
    answer: str = Field(
        default="",
        description=(
            "What the passages show for this requirement, each fact followed by its [source_id]. "
            "Empty if no passage supports it."
        ),
    )
    missing_information: str | None = Field(
        default=None,
        description=(
            "If the passages do not fully answer: the specific information that is missing "
            "(e.g. 'experience integrating with RecTrac'). Null if fully answered."
        ),
    )

    def render(self) -> str:
        """Markdown bullet for this answer, with a [TO CONFIRM: ...] placeholder if needed."""
        parts = [self.answer.strip()] if self.answer.strip() else []
        if self.missing_information or not parts:
            parts.append(placeholder(self.missing_information or ""))
        return f"- **{self.requirement_id}**: {' '.join(parts)}"


class SectionAnswers(BaseModel):
    """Writer output for one response section: one answer per requirement id."""

    answers: list[RequirementAnswer]


def check_section_answers(result: SectionAnswers, expected_ids: list[str]) -> None:
    """Raise ValueError (with a message for the model) unless every id is answered once.

    Rejects grouped answers such as "REQ-030 through REQ-050": each requirement needs its own
    answer, so that its compliance status and sources can be derived.
    """
    problems = []
    ids = [a.requirement_id for a in result.answers]
    missing = [i for i in expected_ids if i not in ids]
    extra = sorted({i for i in ids if i not in expected_ids})
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if missing:
        problems.append(f"missing answers for {', '.join(missing)}")
    if extra:
        problems.append(f"unknown requirement ids {', '.join(extra)}")
    if duplicated:
        problems.append(f"several answers for {', '.join(duplicated)}")
    for answer in result.answers:
        if _RANGE_RE.search(answer.answer):
            problems.append(
                f"{answer.requirement_id} refers to a range of requirements; answer each id "
                "separately"
            )
        if not answer.answer.strip() and not answer.missing_information:
            problems.append(
                f"{answer.requirement_id} has neither an answer nor missing_information"
            )
    if problems:
        raise ValueError("; ".join(problems))


class NonCompliance(BaseModel):
    """A requirement where our documented offer is worse than what the RFP requires."""

    requirement_id: str
    required: str = Field(description="What the RFP requires, e.g. 'RPO of 4 hours'.")
    offered: str = Field(description="What our passages state, e.g. 'RPO of 24 hours'.")


class ReviewResult(BaseModel):
    """LLM-as-judge output for one draft section."""

    coverage: float = Field(ge=0, le=1, description="Share of the requirements fully answered.")
    groundedness: float = Field(
        ge=0, le=1, description="Share of claims supported by the cited passages."
    )
    uncovered_requirements: list[str] = Field(
        default_factory=list, description="Ids of requirements not (fully) answered."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims not supported by the provided passages."
    )
    feedback: str = Field(default="", description="Concrete instructions to improve the draft.")
    non_compliant: list[NonCompliance] = Field(
        default_factory=list,
        description=(
            "Requirements where the passages show we offer LESS than required (e.g. RPO 24 h "
            "offered vs 4 h required). Empty if none."
        ),
    )

    @property
    def score(self) -> float:
        """Combined score used by the routing threshold."""
        return (self.coverage + self.groundedness) / 2


class ComplianceRow(BaseModel):
    """One row of the compliance table in the exported DOCX."""

    requirement_id: str
    requirement_text: str
    type: RequirementType
    status: ComplianceStatus
    where_addressed: str
    sources: list[str]  # chunk ids actually cited in this requirement's answer
    note: str | None = None  # missing information, bid-team check, or non-compliance warning
