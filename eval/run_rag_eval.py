"""(b) RAG quality: retrieval vs labelled sources, LLM-judged context precision and faithfulness.

    python eval/run_rag_eval.py [--config-name NAME] [--limit N]
                                [--no-generation] [--no-judge] [--force]

For each labelled query: hybrid retrieval (top RETRIEVAL_K), source-level hit / precision /
recall / MRR, LLM-judged context precision, an answer generated with the pipeline's writer
prompt, citation validity, and LLM-judged faithfulness. The judge is the reviewer profile
from `.env`. Writes eval/results/rag.{md,json}.
"""

from __future__ import annotations

import argparse
import sys

from _common import RAG_QUERIES, RESULTS_DIR, default_config_name, eval_settings, setup, store

from rfp_copilot.evaluation.experiments import (
    build_eval_retriever,
    eval_rag_query,
    make_judge,
    model_label,
    summarize_rag,
)
from rfp_copilot.evaluation.gold import load_rag_queries
from rfp_copilot.evaluation.store import write_report

SUMMARY_COLUMNS = [
    ("config", "Config"),
    ("queries", "Queries"),
    ("hit", "Hit@k"),
    ("precision", "Src precision@k"),
    ("recall", "Src recall@k"),
    ("reciprocal_rank", "MRR"),
    ("context_precision", "Ctx precision (judge)"),
    ("context_avg_precision", "Ctx AP (judge)"),
    ("faithfulness", "Faithfulness (judge)"),
    ("citation_validity", "Citation validity"),
    ("gen_tokens", "Gen tokens/query"),
    ("gen_cached_share", "Gen cached share"),
    ("generation_s", "Gen time (s)"),
    ("retrieval_s", "Retrieval time (s)"),
]
QUERY_COLUMNS = [
    ("query_id", "Query"),
    ("hit", "Hit"),
    ("precision", "Src P"),
    ("recall", "Src R"),
    ("reciprocal_rank", "RR"),
    ("context_precision", "Ctx P"),
    ("faithfulness", "Faith."),
    ("claims", "Claims"),
    ("citation_validity", "Cit. valid"),
    ("gen_tokens", "Gen tokens"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", default=str(RAG_QUERIES))
    parser.add_argument("--config-name", help="label for this run (default: the model name)")
    parser.add_argument("--limit", type=int, help="evaluate at most N queries")
    parser.add_argument("--no-generation", action="store_true", help="retrieval metrics only")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judged metrics")
    parser.add_argument("--force", action="store_true", help="recompute stored results")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    base = setup(args.verbose)
    settings = eval_settings(base)
    config = args.config_name or default_config_name(settings)
    queries = load_rag_queries(args.queries)[: args.limit]
    judge = None if args.no_judge else make_judge(base)
    retriever, rerank_usage = build_eval_retriever(settings)
    rows = []
    for query in queries:
        print(f"[{config}] {query.id} ...", flush=True)
        rows.append(
            eval_rag_query(
                settings,
                retriever,
                rerank_usage,
                query,
                store(),
                config,
                judge=judge,
                generate=not args.no_generation,
                force=args.force,
            )
        )
    _, md = write_report(
        RESULTS_DIR,
        "rag",
        title="RAG quality",
        sections=[
            ("Summary (means over queries)", [summarize_rag(rows, config)], SUMMARY_COLUMNS),
            ("Per query", rows, QUERY_COLUMNS),
        ],
        meta={
            "config": config,
            "writer": model_label(settings),
            "judge": "none" if judge is None else model_label(base, "judge"),
            "k": settings.retrieval_k,
            "chunk_size": settings.chunk_size,
            "rerank": settings.rerank_enabled,
        },
        notes=[
            "Src precision/recall/MRR use the hand-labelled relevant files (no LLM).",
            "Ctx precision / AP and faithfulness are LLM-judge estimates, not ground truth.",
            "Citation validity = share of [ids] in the answer that were actually retrieved.",
            "Generation calls served from the LLM cache take ~0 s (see the cached share).",
        ],
    )
    print(md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
