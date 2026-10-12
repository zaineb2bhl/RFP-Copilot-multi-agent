"""Tests for chunking, RRF fusion, BM25, embeddings, the vector store and hybrid retrieval."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rfp_copilot.config import EmbeddingProfile, Settings
from rfp_copilot.models import Chunk
from rfp_copilot.retrieval.bm25 import BM25Index
from rfp_copilot.retrieval.chunking import chunk_markdown, load_knowledge_base
from rfp_copilot.retrieval.embeddings import (
    EmbeddingCache,
    HashEmbedder,
    OpenAICompatEmbedder,
    build_embedder,
)
from rfp_copilot.retrieval.fusion import reciprocal_rank_fusion
from rfp_copilot.retrieval.retriever import HybridRetriever, sync_index
from rfp_copilot.retrieval.vector_store import EmbeddingModelMismatchError, VectorStore
from tests.conftest import KB_DIR

DOC = """# Project Alpha

Intro paragraph about the project.

## Approach

First paragraph of the approach. It has two sentences.

Second paragraph of the approach.

## Results

Results paragraph.
"""


# -- chunking -----------------------------------------------------------------------------


def test_chunks_follow_headings_and_ids() -> None:
    chunks = chunk_markdown(DOC, "alpha.md", chunk_size=800)
    assert [c.id for c in chunks] == ["alpha#1", "alpha#2", "alpha#3"]
    assert chunks[1].title == "Project Alpha > Approach"
    assert "Second paragraph" in chunks[1].text
    assert chunks[0].title == "Project Alpha"


def test_long_sections_are_split_with_overlap() -> None:
    body = " ".join(f"Sentence number {i} talks about delivery." for i in range(60))
    chunks = chunk_markdown(f"# Big\n\n{body}", "big.md", chunk_size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + 60 + 2 for c in chunks)
    tail = chunks[0].text[-30:]
    assert tail.split()[-1] in chunks[1].text  # overlap carries the end of the previous chunk


def test_knowledge_base_loads() -> None:
    chunks = load_knowledge_base(KB_DIR)
    assert len({c.source for c in chunks}) >= 10
    assert len({c.id for c in chunks}) == len(chunks)


def test_missing_kb_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_knowledge_base(tmp_path / "nope")


# -- fusion & BM25 -------------------------------------------------------------------------


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert set(ids[:2]) == {"a", "b"}
    assert ids[2:] == ["c", "d"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_single_list_keeps_order_and_ties_are_deterministic() -> None:
    assert [d for d, _ in reciprocal_rank_fusion([["x", "y", "z"]])] == ["x", "y", "z"]
    assert [d for d, _ in reciprocal_rank_fusion([["p"], ["q"]])] == ["p", "q"]


def test_bm25_finds_keyword_match() -> None:
    chunks = [
        Chunk(id="a#1", source="a.md", title="A", text="ISO 27001 certification and security"),
        Chunk(id="b#1", source="b.md", title="B", text="Agile sprints and steering committee"),
    ]
    assert BM25Index(chunks).search("ISO 27001", 5)[0][0] == "a#1"
    assert BM25Index(chunks).search("the and of", 5) == []


# -- embeddings ----------------------------------------------------------------------------


class FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model: str, input: list[str]):
        self.batches.append(list(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(len(t)), 1.0]) for i, t in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)))  # out of order on purpose


def test_embedder_batches_caches_and_prefixes(tmp_path: Path) -> None:
    api = FakeEmbeddingsAPI()
    profile = EmbeddingProfile(
        provider="ollama", base_url="x", api_key="k", model="nomic-embed-text"
    )
    embedder = OpenAICompatEmbedder(
        profile, batch_size=2, cache=EmbeddingCache(tmp_path / "e.sqlite"), client=api
    )
    vectors = embedder.embed_documents(["a", "bb", "ccc"])
    assert [b for b in api.batches] == [
        ["search_document: a", "search_document: bb"],
        ["search_document: ccc"],
    ]
    assert vectors[2][0] == len("search_document: ccc")
    embedder.embed_documents(["a", "bb", "ccc"])
    assert len(api.batches) == 2  # all from cache
    embedder.embed_query("q")
    assert api.batches[-1] == ["search_query: q"]


def test_hash_embedder_similarity() -> None:
    emb = HashEmbedder()
    a, b, c = emb.embed_documents(["cloud hosting azure", "azure cloud hosting region", "pizza"])
    dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))  # noqa: E731
    assert dot(a, b) > dot(a, c)


def test_build_embedder_mock_profile() -> None:
    assert isinstance(build_embedder(Settings(embedding_provider="mock")), HashEmbedder)


# -- vector store & hybrid retrieval ----------------------------------------------------


def test_vector_store_refuses_other_embedding_model(tmp_path: Path) -> None:
    VectorStore(tmp_path, "kb_test", "model-a")
    with pytest.raises(EmbeddingModelMismatchError, match="Re-index"):
        VectorStore(tmp_path, "kb_test", "model-b")
    assert VectorStore(tmp_path, "kb_test", "model-b", reset=True).count() == 0


def test_sync_index_is_incremental(tmp_path: Path) -> None:
    emb = HashEmbedder()
    store = VectorStore(tmp_path, "kb_test", emb.model)
    chunks = [Chunk(id=f"d#{i}", source="d.md", title="D", text=f"text {i}") for i in range(3)]
    assert sync_index(chunks, emb, store)["embedded"] == 3
    assert sync_index(chunks, emb, store)["embedded"] == 0
    changed = [chunks[0].model_copy(update={"text": "new text"}), chunks[1]]
    stats = sync_index(changed, emb, store)
    assert (stats["embedded"], stats["deleted"]) == (1, 1)
    assert store.count() == 2


def test_hybrid_retriever_finds_relevant_sources(retriever: HybridRetriever) -> None:
    results = retriever.search("ISO/IEC 27001 certification of the supplier", k=5)
    assert len(results) == 5
    assert any(
        r.source in {"company-profile.md", "method-security-privacy.md"} for r in results[:3]
    )
    sla = retriever.search_many(["response time for critical incidents", "99.9% availability"], k=3)
    assert sla[0].source == "method-support-sla.md"
