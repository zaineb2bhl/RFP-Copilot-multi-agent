"""Graph state.

Values are plain JSON-compatible dicts/lists (model dumps) rather than Pydantic objects, so
the SQLite checkpointer can always serialise them; nodes validate them back into models.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

RunOutcome = Literal["approved", "rejected"]


class RFPState(TypedDict, total=False):
    """State carried through the pipeline for one RFP run."""

    run_id: str
    rfp_path: str
    document: dict[str, Any]  # ParsedDocument
    requirements: list[dict[str, Any]]  # Requirement
    groups: list[dict[str, Any]]  # ResponseGroup
    passages: dict[str, list[dict[str, Any]]]  # group id -> RetrievedPassage list
    drafts: dict[str, str]  # group id -> markdown
    reviews: dict[str, dict[str, Any]]  # group id -> ReviewResult
    pending_groups: list[str]  # groups written in the latest write pass
    iteration: int  # number of write passes done
    review_history: list[dict[str, float]]  # per pass: group id -> score
    outcome: RunOutcome
    output_path: str
