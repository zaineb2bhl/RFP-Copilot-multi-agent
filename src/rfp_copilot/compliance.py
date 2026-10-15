"""Deterministic compliance status, sources and notes for every requirement.

The status of a requirement is derived only from its own answer in the (possibly human-edited)
draft, the passages that were retrieved for its section, and the reviewer's non-compliance
findings. No LLM is involved, so the compliance matrix can be audited and tested.
"""

from __future__ import annotations

import re

from rfp_copilot.models import (
    PLACEHOLDER_RE,
    ComplianceRow,
    ComplianceStatus,
    NonCompliance,
    Requirement,
    ResponseGroup,
    ReviewResult,
    extract_citations,
)

_ANSWER_START_RE = re.compile(r"^\s*(?:[-*+]\s*)?\*\*(REQ-\d+)\*\*")


def requirement_segments(draft: str) -> dict[str, str]:
    """Map each requirement id to its own answer text in a draft.

    An answer starts at a line beginning with its bold id ("- **REQ-004**: ...") and runs
    until the next such line. Ids only mentioned inline (older free-text drafts) get the
    lines that mention them.
    """
    segments: dict[str, list[str]] = {}
    current: str | None = None
    for line in draft.splitlines():
        match = _ANSWER_START_RE.match(line)
        if match:
            current = match.group(1)
            segments.setdefault(current, []).append(line)
        elif current and line.strip():
            segments[current].append(line)
    result = {k: "\n".join(v) for k, v in segments.items()}
    for line in draft.splitlines():
        for req_id in set(re.findall(r"REQ-\d+", line)) - set(result):
            result[req_id] = line
    return result


def missing_information(segment: str) -> str | None:
    """The text of the first [TO CONFIRM: ...] placeholder, if any."""
    match = PLACEHOLDER_RE.search(segment)
    if not match:
        return None
    return (match.group(1) or "").strip() or "information to be confirmed by the bid team"


def derive_status(
    requirement: Requirement,
    segment: str | None,
    retrieved_ids: set[str],
    non_compliance: NonCompliance | None,
) -> tuple[ComplianceStatus, list[str], str | None]:
    """Status, cited sources and note for one requirement (see `ComplianceStatus`).

    Rules, in order:
    1. administrative items (deadlines, copies, forms, page limits...) -> `to_confirm`, with a
       checklist line: only the bid team can verify them;
    2. a reviewer finding that our offer is below the requirement -> `non_compliant`;
    3. no answer, or a placeholder with no cited evidence -> `not_covered`;
    4. cited evidence plus a placeholder -> `partial`;
    5. an answer with no cited evidence (e.g. a commitment typed by a human) -> `to_confirm`;
    6. otherwise -> `addressed`.
    Sources are only the retrieved chunks actually cited in this requirement's answer.
    """
    if requirement.type == "administrative":
        return (
            "to_confirm",
            [],
            f"Bid team check: {requirement.text} (RFP p.{requirement.source_page})",
        )
    cited = [c for c in extract_citations(segment or "") if c in retrieved_ids]
    if non_compliance is not None:
        note = (
            f"NON-COMPLIANT: the RFP requires {non_compliance.required}; "
            f"our documented offer is {non_compliance.offered}."
        )
        return "non_compliant", cited, note
    gap = missing_information(segment or "")
    if not segment or (not cited and gap):
        return "not_covered", [], gap or "No evidence in the knowledge base."
    if gap:
        return "partial", cited, gap
    if not cited:
        return "to_confirm", [], "Answer has no cited evidence: a person must confirm it."
    return "addressed", cited, None


def build_compliance_rows(
    requirements: list[Requirement],
    groups: list[ResponseGroup],
    drafts: dict[str, str],
    reviews: dict[str, ReviewResult],
    retrieved: dict[str, set[str]],
) -> list[ComplianceRow]:
    """One compliance row per requirement. `retrieved` maps group id -> retrieved chunk ids.

    `where_addressed` uses the section numbering of the exported document (1.N).
    """
    group_of = {r: (i, g) for i, g in enumerate(groups, start=1) for r in g.requirement_ids}
    rows: list[ComplianceRow] = []
    for req in requirements:
        number, group = group_of.get(req.id, (None, None))
        draft = drafts.get(group.id, "") if group else ""
        segment = requirement_segments(draft).get(req.id) if draft else None
        review = reviews.get(group.id) if group else None
        finding = next(
            (n for n in (review.non_compliant if review else []) if n.requirement_id == req.id),
            None,
        )
        status, sources, note = derive_status(
            req, segment, retrieved.get(group.id, set()) if group else set(), finding
        )
        rows.append(
            ComplianceRow(
                requirement_id=req.id,
                requirement_text=req.text,
                type=req.type,
                status=status,
                where_addressed=f"1.{number} {group.title}" if group and segment else "-",
                sources=sources,
                note=note,
            )
        )
    return rows


def rows_from_state(values: dict) -> list[ComplianceRow]:
    """Compliance rows computed from a checkpointed graph state (plain dicts)."""
    return build_compliance_rows(
        [Requirement.model_validate(r) for r in values.get("requirements", [])],
        [ResponseGroup.model_validate(g) for g in values.get("groups", [])],
        values.get("drafts", {}),
        {k: ReviewResult.model_validate(v) for k, v in values.get("reviews", {}).items()},
        {g: {p["chunk_id"] for p in plist} for g, plist in values.get("passages", {}).items()},
    )
