"""Shared fixtures. Tests never touch the network, Ollama, or the developer's `.env`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rfp_copilot.config import Settings, get_settings
from rfp_copilot.llm import MockLLM
from rfp_copilot.models import ParsedDocument
from rfp_copilot.retrieval.embeddings import HashEmbedder
from rfp_copilot.retrieval.retriever import HybridRetriever, build_retriever
from rfp_copilot.samples import write_sample_rfp
from tests.fakes import make_pipeline_llm

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "knowledge_base"
_ENV_PREFIXES = ("LLM_", "REVIEWER_LLM_", "EMBEDDING_", "LANGFUSE_")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Clear project env vars and run from an empty directory (so no `.env` is loaded)."""
    for name in list(os.environ):
        if name.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline settings: mock LLM and hash embeddings, all state under tmp_path."""
    return Settings(
        llm_provider="mock",
        embedding_provider="mock",
        llm_cache_enabled=False,
        embedding_cache_enabled=False,
        knowledge_base_dir=KB_DIR,
        chroma_dir=tmp_path / "chroma",
        runs_dir=tmp_path / "runs",
        min_relevance=0.0,  # the 0.67 floor is calibrated for nomic-embed-text, not hash vectors
    )


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic sample RFP rendered once per test session."""
    return write_sample_rfp(tmp_path_factory.mktemp("rfp") / "sample_rfp.pdf")


@pytest.fixture(scope="session")
def sample_document(sample_pdf: Path) -> ParsedDocument:
    from rfp_copilot.ingestion.pdf_parser import parse_pdf

    return parse_pdf(sample_pdf)


@pytest.fixture
def retriever(settings: Settings) -> HybridRetriever:
    """Hybrid retriever over the real synthetic knowledge base, with hash embeddings."""
    return build_retriever(settings, embedder=HashEmbedder())


@pytest.fixture
def pipeline_llm() -> MockLLM:
    """One MockLLM playing extractor, writer and reviewer (see tests.fakes)."""
    return MockLLM(make_pipeline_llm())
