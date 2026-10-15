"""List possible false abstentions of a run, for manual review.

    python eval/false_abstentions.py <run_id> [--min-similarity 0.70]

A possible false abstention is a requirement answered only by a "[TO CONFIRM: ...]"
placeholder although a chunk retrieved for it scored at or above the similarity threshold
(default: MIN_RELEVANCE, the retrieval floor). Review each one: if the chunk answers the
requirement, the writer abstained wrongly; if not, the knowledge base really lacks it.
Writes eval/results/false_abstentions_<run_id>.{md,json}. No LLM call.
"""

from __future__ import annotations

import argparse

from _common import RESULTS_DIR, setup

from rfp_copilot.evaluation.abstentions import possible_false_abstentions
from rfp_copilot.evaluation.store import write_report
from rfp_copilot.pipeline import PipelineService

COLUMNS = [
    ("requirement_id", "Requirement"),
    ("requirement", "Text"),
    ("missing_according_to_draft", "Missing (per draft)"),
    ("best_chunk", "Best chunk"),
    ("best_similarity", "Similarity"),
    ("best_chunk_text", "Chunk text"),
    ("verdict", "Your verdict (fill in)"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("--min-similarity", type=float, help="default: MIN_RELEVANCE")
    args = parser.parse_args()

    settings = setup()
    threshold = args.min_similarity if args.min_similarity is not None else settings.min_relevance
    values = PipelineService(settings).get_snapshot(args.run_id).values
    findings = possible_false_abstentions(values, threshold)
    not_covered = sum(
        1 for r in values.get("requirements", []) if r.get("type") != "administrative"
    )
    _, md = write_report(
        RESULTS_DIR,
        f"false_abstentions_{args.run_id}",
        title=f"Possible false abstentions: run {args.run_id}",
        sections=[("To review by hand", [{**f, "verdict": ""} for f in findings], COLUMNS)],
        meta={
            "run_id": args.run_id,
            "similarity_threshold": threshold,
            "non_administrative_requirements": not_covered,
            "possible_false_abstentions": len(findings),
        },
        notes=[
            "Each row: the draft says [TO CONFIRM] although a chunk retrieved for this "
            "requirement scored above the threshold.",
            "Verdict: 'false abstention' if the chunk answers the requirement (writer error), "
            "'correct' if it does not (knowledge-base gap).",
        ],
    )
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
