"""Shared helpers for the evaluation scripts (paths, settings, logging, config overrides)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from rfp_copilot.config import Settings, get_settings, resolve_embedding_profile
from rfp_copilot.evaluation.store import ResultStore

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "eval" / "results"
GOLD_DIR = ROOT / "data" / "eval"
RAG_QUERIES = GOLD_DIR / "rag_queries.json"


def setup(verbose: bool = False) -> Settings:
    """Run from the repo root (so `.env` and relative paths resolve) and configure logging."""
    os.chdir(ROOT)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "chromadb", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    get_settings.cache_clear()
    return get_settings()


def store() -> ResultStore:
    """Per-item results (gitignored); reports go to `eval/results/*.md|json`."""
    return ResultStore(RESULTS_DIR / "items")


def eval_settings(base: Settings, **overrides: Any) -> Settings:
    """Settings for one evaluated config.

    Vectors go to a separate Chroma directory, one collection per (embedding model, chunk
    size, overlap), so evaluations never touch the app's index and configs never collide.
    """
    settings = base.model_copy(update=overrides)
    model = re.sub(r"[^A-Za-z0-9._-]+", "-", resolve_embedding_profile(settings).model)
    collection = f"eval_{model}_c{settings.chunk_size}_o{settings.chunk_overlap}"
    return settings.model_copy(
        update={"chroma_dir": ROOT / ".cache" / "eval_chroma", "chroma_collection": collection}
    )


def default_config_name(settings: Settings) -> str:
    """A label like `gemini-3.5-flash-lite` for the configured writer model."""
    from rfp_copilot.evaluation.experiments import model_label

    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_label(settings).split(":", 1)[-1])
