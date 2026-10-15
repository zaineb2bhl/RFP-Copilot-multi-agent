"""Hand-annotated gold data: RFP requirements, and retrieval queries with relevant sources."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from rfp_copilot.models import RequirementType

GOLD_SUFFIX = ".gold.json"


class GoldRequirement(BaseModel):
    """One requirement as annotated by a human."""

    id: str
    text: str
    type: RequirementType
    source_page: int = Field(ge=1)


class GoldFile(BaseModel):
    """Annotation of one RFP: `rfp` is the PDF path relative to the repository root."""

    rfp: str
    description: str = ""
    annotator: str = ""
    requirements: list[GoldRequirement]

    def rfp_path(self, root: Path) -> Path:
        """Absolute path of the annotated PDF."""
        return (root / self.rfp).resolve()


class RagQuery(BaseModel):
    """A retrieval query with the knowledge-base files that contain the answer."""

    id: str
    query: str
    relevant_sources: list[str] = Field(min_length=1)


class RagQuerySet(BaseModel):
    """A set of labelled retrieval queries for one knowledge base."""

    description: str = ""
    queries: list[RagQuery]


def load_gold_files(gold_dir: Path | str) -> list[tuple[Path, GoldFile]]:
    """All `*.gold.json` files in `gold_dir`, sorted by name."""
    paths = sorted(Path(gold_dir).glob(f"*{GOLD_SUFFIX}"))
    return [(p, GoldFile.model_validate_json(p.read_text(encoding="utf-8"))) for p in paths]


def load_rag_queries(path: Path | str) -> list[RagQuery]:
    """Queries from a `RagQuerySet` JSON file."""
    return RagQuerySet.model_validate_json(Path(path).read_text(encoding="utf-8")).queries


def gold_name(path: Path) -> str:
    """Short document name of a gold file (`sample_rfp` for `sample_rfp.gold.json`)."""
    return path.name.removesuffix(GOLD_SUFFIX)
