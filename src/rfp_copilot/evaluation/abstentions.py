"""Audit of possible false abstentions: "[TO CONFIRM]" answers despite strong evidence.

The writer abstains (placeholder, no citation) when it finds no support. If a chunk retrieved
for that very requirement scored above a similarity threshold, the abstention may be wrong:
the evidence was there and the writer did not use it. These cases are listed for a human to
review; the tool cannot decide alone whether the chunk really answers the requirement.
"""

from __future__ import annotations

from typing import Any

from rfp_copilot.compliance import rows_from_state


def possible_false_abstentions(
    values: dict[str, Any], min_similarity: float
) -> list[dict[str, Any]]:
    """`not_covered` requirements with a retrieved chunk scoring >= `min_similarity`.

    `values` is a checkpointed graph state. Each item lists the requirement, what the draft
    says is missing, and the qualifying chunks (best first) with a text snippet.
    """
    chunks = {p["chunk_id"]: p for plist in values.get("passages", {}).values() for p in plist}
    evidence = values.get("evidence", {})
    requirements = {r["id"]: r for r in values.get("requirements", [])}
    group_of = {r: g["id"] for g in values.get("groups", []) for r in g["requirement_ids"]}
    findings = []
    for row in rows_from_state(values):
        if row.status != "not_covered":
            continue
        found = evidence.get(row.requirement_id)
        if found is None:  # runs made before per-requirement evidence: use the section's chunks
            group_chunks = values.get("passages", {}).get(group_of.get(row.requirement_id, ""), [])
            found = [
                {"chunk_id": p["chunk_id"], "similarity": p.get("similarity")} for p in group_chunks
            ]
        strong = sorted(
            (e for e in found if (e.get("similarity") or 0.0) >= min_similarity),
            key=lambda e: -(e["similarity"] or 0.0),
        )
        if not strong:
            continue
        best = chunks.get(strong[0]["chunk_id"], {})
        findings.append(
            {
                "requirement_id": row.requirement_id,
                "requirement": requirements.get(row.requirement_id, {}).get("text", ""),
                "missing_according_to_draft": row.note,
                "best_chunk": strong[0]["chunk_id"],
                "best_similarity": round(strong[0]["similarity"], 3),
                "best_chunk_text": (best.get("text") or "")[:300],
                "other_chunks": [e["chunk_id"] for e in strong[1:]],
            }
        )
    return findings
