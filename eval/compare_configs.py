"""(c) Configuration comparisons, with (d) tokens / cost / latency for each.

    python eval/compare_configs.py [--experiments chunk_size,rerank,reviewer_loop,model]
                                   [--limit N] [--no-judge] [--force]

Experiments:
- chunk_size:    chunks of 400 vs 800 characters (RAG eval)
- rerank:        hybrid retrieval with vs without listwise LLM reranking (RAG eval)
- reviewer_loop: full pipeline with 1 write pass vs write/review loop (2 passes)
- model:         writer = local Ollama qwen2.5:3b vs Gemini (extraction + RAG eval)

The judge stays fixed (reviewer profile from `.env`) across configs. Results are stored per
item, so an interrupted comparison resumes. Writes eval/results/compare_<experiment>.{md,json}.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from typing import Any

from _common import (
    GOLD_DIR,
    RAG_QUERIES,
    RESULTS_DIR,
    eval_settings,
    setup,
    store,
)

from rfp_copilot.config import ConfigError, Settings
from rfp_copilot.evaluation.experiments import (
    Judge,
    build_eval_retriever,
    eval_extraction_doc,
    eval_pipeline_doc,
    eval_rag_query,
    make_judge,
    model_label,
    summarize_extraction,
    summarize_pipeline,
    summarize_rag,
)
from rfp_copilot.evaluation.gold import load_gold_files, load_rag_queries
from rfp_copilot.evaluation.store import write_report

logger = logging.getLogger("compare_configs")

RAG_COLUMNS = [
    ("config", "Config"),
    ("hit", "Hit@k"),
    ("precision", "Src P@k"),
    ("recall", "Src R@k"),
    ("reciprocal_rank", "MRR"),
    ("context_precision", "Ctx P (judge)"),
    ("faithfulness", "Faithfulness (judge)"),
    ("citation_validity", "Citation validity"),
    ("gen_tokens", "Gen tokens/query"),
    ("gen_cached_share", "Gen cached share"),
    ("rerank_tokens", "Rerank tokens/query"),
    ("retrieval_s", "Retrieval (s)"),
    ("generation_s", "Generation (s)"),
]
EXTRACTION_COLUMNS = [
    ("config", "Config"),
    ("model", "Model"),
    ("precision", "P"),
    ("recall", "R"),
    ("f1", "F1"),
    ("type_accuracy", "Type acc"),
    ("tokens_per_doc", "Tokens/doc"),
    ("cached_share", "Cached share"),
    ("cost_per_doc", "Cost $/doc"),
    ("wall_s_per_doc", "Time/doc (s)"),
]
PIPELINE_COLUMNS = [
    ("config", "Config"),
    ("documents", "Docs"),
    ("write_passes", "Write passes"),
    ("reviewer_score", "Reviewer score"),
    ("faithfulness", "Faithfulness (judge)"),
    ("addressed_share", "Addressed share"),
    ("tokens", "Tokens/doc"),
    ("cost_usd", "Cost $/doc"),
    ("wall_s", "Time/doc (s)"),
    ("cached_share", "Cached share"),
]


def model_configs(base: Settings) -> dict[str, dict[str, Any]]:
    """Writer = Gemini (from .env if it is the Gemini profile) vs local Ollama 3B."""
    gemini_model = base.llm_model if base.llm_provider == "gemini" else None
    return {
        "gemini": {"llm_provider": "gemini", "llm_model": gemini_model},
        "ollama_qwen2.5-3b": {
            "llm_provider": "ollama",
            "llm_model": "qwen2.5:3b",
            "llm_api_key": None,
            "llm_base_url": None,
            "llm_max_rpm": 0,
            # CPU inference runs at about 6 output tokens/s on an 8 GB laptop: allow long calls,
            # do not retry timeouts several times, and keep extraction batches small.
            "llm_timeout_s": 600,
            "llm_max_retries": 1,
            "extraction_batch_chars": 3000,
        },
    }


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "chunk_size": {
        "kinds": ["rag"],
        "configs": lambda base: {
            "chunk_400": {"chunk_size": 400, "chunk_overlap": 60},
            "chunk_800": {"chunk_size": 800, "chunk_overlap": 120},
        },
    },
    "rerank": {
        "kinds": ["rag"],
        "configs": lambda base: {
            "no_rerank": {"rerank_enabled": False},
            "rerank": {"rerank_enabled": True},
        },
    },
    "reviewer_loop": {
        "kinds": ["pipeline"],
        "configs": lambda base: {
            "single_pass": {"max_iterations": 1},
            "review_loop": {"max_iterations": 2},
        },
    },
    "model": {"kinds": ["extraction", "rag"], "configs": model_configs},
}


def _run_rag(
    settings: Settings, config: str, judge: Judge | None, args: argparse.Namespace
) -> dict[str, Any]:
    retriever, rerank_usage = build_eval_retriever(settings)
    rows = [
        eval_rag_query(
            settings, retriever, rerank_usage, q, store(), config, judge=judge, force=args.force
        )
        for q in load_rag_queries(RAG_QUERIES)[: args.limit]
    ]
    return summarize_rag(rows, config)


def _run_extraction(
    settings: Settings, config: str, judge: Judge | None, args: argparse.Namespace
) -> dict[str, Any]:
    rows = [
        eval_extraction_doc(
            settings, p, g, store(), config, root=RESULTS_DIR.parents[1], force=args.force
        )
        for p, g in load_gold_files(GOLD_DIR)[: args.limit]
    ]
    return summarize_extraction(rows, config)


def _run_pipeline(
    settings: Settings, config: str, judge: Judge | None, args: argparse.Namespace
) -> dict[str, Any]:
    rows = [
        eval_pipeline_doc(
            settings,
            p,
            g,
            store(),
            config,
            root=RESULTS_DIR.parents[1],
            judge=judge,
            force=args.force,
        )
        for p, g in load_gold_files(GOLD_DIR)[: args.limit]
    ]
    return summarize_pipeline(rows, config)


RUNNERS: dict[str, tuple[Callable[..., dict[str, Any]], list[tuple[str, str]], str]] = {
    "rag": (_run_rag, RAG_COLUMNS, "RAG quality"),
    "extraction": (_run_extraction, EXTRACTION_COLUMNS, "Requirement extraction"),
    "pipeline": (_run_pipeline, PIPELINE_COLUMNS, "Full pipeline"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--limit", type=int, help="at most N queries / documents per config")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judged metrics")
    parser.add_argument("--force", action="store_true", help="recompute stored results")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    base = setup(args.verbose)
    judge = None if args.no_judge else make_judge(base)
    names = [n.strip() for n in args.experiments.split(",") if n.strip()]
    unknown = set(names) - set(EXPERIMENTS)
    if unknown:
        parser.error(f"unknown experiments: {', '.join(sorted(unknown))}")

    for name in names:
        experiment = EXPERIMENTS[name]
        sections = []
        skipped = []
        for kind in experiment["kinds"]:
            runner, columns, heading = RUNNERS[kind]
            rows = []
            for config, overrides in experiment["configs"](base).items():
                label = f"{name}/{config}"
                print(f"[{label}] {kind} ...", flush=True)
                try:
                    settings = eval_settings(base, **overrides)
                    row = runner(settings, f"{name}__{config}", judge, args)
                except ConfigError as exc:
                    skipped.append(f"{config}: skipped ({exc})")
                    continue
                rows.append({**row, "config": config, "model": model_label(settings)})
            sections.append((heading, rows, columns))
        _, md = write_report(
            RESULTS_DIR,
            f"compare_{name}",
            title=f"Config comparison: {name}",
            sections=sections,
            meta={
                "judge": "none" if judge is None else model_label(base, "judge"),
                "limit": args.limit or "all",
            },
            notes=[
                *skipped,
                "Judge metrics are LLM estimates from one fixed judge model.",
                "Times include free-tier throttling waits. Calls served from the LLM cache take "
                "~0 s (see the cached share); for pure latency run with LLM_CACHE_ENABLED=false "
                "and --force.",
            ],
        )
        print(md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
