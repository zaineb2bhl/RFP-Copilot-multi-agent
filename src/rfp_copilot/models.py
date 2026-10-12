"""Pydantic models shared across ingestion, retrieval, the graph and export."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# One or more citation ids in square brackets: [a#1] or [a#1, b#2].
_CITATION_GROUP_RE = re.compile(r"\[([A-Za-z0-9_.\-]+#\d+(?:\s*[,;]\s*[A-Za-z0-9_.\-]+#\d+)*)\]")


def extract_citations(text: str) -> list[str]:
    """Citation ids found in `text` (`[a#1]` or `[a#1, b#2]`), in order of first appearance."""
    ids = (
        part.strip()
        for group in _CITATION_GROUP_RE.findall(text)
        for part in re.split(r"[,;]", group)
    )
    return list(dict.fromkeys(ids))


RequirementType = Literal["mandatory", "desirable", "administrative"]
ComplianceStatus = Literal["addressed", "partially_addressed", "not_addressed"]


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


class ResponseGroup(BaseModel):
    """Requirements answered together in one response section."""

    id: str
    title: str
    requirement_ids: list[str]


class DraftSection(BaseModel):
    """LLM output for one response section."""

    content: str = Field(
        description="Markdown response text with inline citations like [source_id]."
    )


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
    sources: list[str]
