"""(a) Requirement extraction precision / recall / F1 against hand-annotated gold files.

    python eval/run_extraction_eval.py [--config-name NAME] [--limit N] [--force]

Uses the LLM configured in `.env`. Each document's result is stored in eval/results/items and
reused on later runs (use --force to recompute). Writes eval/results/extraction.{md,json}.
"""

from __future__ import annotations

import argparse
import sys

from _common import GOLD_DIR, RESULTS_DIR, default_config_name, eval_settings, setup, store

from rfp_copilot.evaluation.experiments import eval_extraction_doc, summarize_extraction
from rfp_copilot.evaluation.gold import load_gold_files
from rfp_copilot.evaluation.metrics import DEFAULT_MATCH_THRESHOLD
from rfp_copilot.evaluation.store import write_report

DOC_COLUMNS = [
    ("document", "Document"),
    ("model", "Model"),
    ("gold", "Gold"),
    ("predicted", "Pred"),
    ("precision", "P"),
    ("recall", "R"),
    ("f1", "F1"),
    ("type_accuracy", "Type acc"),
    ("page_accuracy", "Page acc"),
    ("tokens", "Tokens"),
    ("cached_calls", "Cached calls"),
    ("cost_usd", "Cost $"),
    ("wall_s", "Time (s)"),
]
SUMMARY_COLUMNS = [
    ("config", "Config"),
    ("model", "Model"),
    ("documents", "Docs"),
    ("precision", "P"),
    ("recall", "R"),
    ("f1", "F1"),
    ("type_accuracy", "Type acc"),
    ("page_accuracy", "Page acc"),
    ("tokens_per_doc", "Tokens/doc"),
    ("cached_share", "Cached share"),
    ("cost_per_doc", "Cost $/doc"),
    ("wall_s_per_doc", "Time/doc (s)"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold-dir", default=str(GOLD_DIR))
    parser.add_argument("--config-name", help="label for this run (default: the model name)")
    parser.add_argument("--limit", type=int, help="evaluate at most N documents")
    parser.add_argument("--force", action="store_true", help="recompute stored results")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    settings = eval_settings(setup(args.verbose))
    config = args.config_name or default_config_name(settings)
    gold_files = load_gold_files(args.gold_dir)[: args.limit]
    if not gold_files:
        print(f"No *.gold.json files in {args.gold_dir}", file=sys.stderr)
        return 1
    rows = []
    for path, gold in gold_files:
        print(f"[{config}] {path.name} ...", flush=True)
        rows.append(
            eval_extraction_doc(
                settings,
                path,
                gold,
                store(),
                config,
                root=RESULTS_DIR.parents[1],
                force=args.force,
                threshold=args.threshold,
            )
        )
    summary = summarize_extraction(rows, config)
    _, md = write_report(
        RESULTS_DIR,
        "extraction",
        title="Requirement extraction",
        sections=[
            ("Summary (micro-averaged)", [summary], SUMMARY_COLUMNS),
            ("Per document", rows, DOC_COLUMNS),
        ],
        meta={"config": config, "match_threshold": args.threshold},
        notes=[
            "A predicted requirement matches a gold one when the Dice similarity of their "
            "content words is >= the threshold (one-to-one, greedy).",
            "Type/page accuracy are computed over matched requirements only.",
            "Time includes throttling waits; cost is 0 on free tiers (tokens are still counted).",
            "Calls served from the LLM cache take ~0 s: check the cached columns before reading "
            "times, or run with LLM_CACHE_ENABLED=false --force for pure latency.",
        ],
    )
    print(md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
