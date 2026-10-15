"""Hybrid retriever (BM25 + embeddings fused with RRF) and knowledge-base index sync."""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Sequence

from rfp_copilot.config import Settings
from rfp_copilot.llm import UsageTracker, build_llm
from rfp_copilot.models import Chunk, RetrievedPassage
from rfp_copilot.retrieval.bm25 import BM25Index
from rfp_copilot.retrieval.chunking import load_knowledge_base
from rfp_copilot.retrieval.embeddings import Embedder, build_embedder
from rfp_copilot.retrieval.fusion import reciprocal_rank_fusion
from rfp_copilot.retrieval.rerank import LLMReranker
from rfp_copilot.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _content_hash(chunk: Chunk) -> str:
    return hashlib.sha256(f"{chunk.title}\n{chunk.text}".encode()).hexdigest()[:16]


def _embedding_text(chunk: Chunk) -> str:
    return f"{chunk.title}\n{chunk.text}"


def sync_index(chunks: list[Chunk], embedder: Embedder, store: VectorStore) -> dict[str, int]:
    """Make the vector store match `chunks`: embed new/changed chunks, delete removed ones."""
    indexed = store.content_hashes()
    wanted = {c.id: _content_hash(c) for c in chunks}
    to_embed = [c for c in chunks if indexed.get(c.id) != wanted[c.id]]
    to_delete = [id_ for id_ in indexed if id_ not in wanted]
    if to_embed:
        vectors = embedder.embed_documents([_embedding_text(c) for c in to_embed])
        store.upsert([c.id for c in to_embed], vectors, [wanted[c.id] for c in to_embed])
    store.delete(to_delete)
    stats = {
        "chunks": len(chunks),
        "embedded": len(to_embed),
        "deleted": len(to_delete),
        "unchanged": len(chunks) - len(to_embed),
    }
    logger.info("Index sync: %s", stats)
    return stats


class HybridRetriever:
    """Searches the knowledge base with BM25 and vector similarity, fused with RRF."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        store: VectorStore,
        *,
        candidates: int = 20,
        rrf_k: int = 60,
        reranker: LLMReranker | None = None,
        rerank_candidates: int = 12,
        min_relevance: float = 0.0,
    ) -> None:
        self.chunks = {c.id: c for c in chunks}
        self.min_relevance = min_relevance
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.embedder = embedder
        self.store = store
        self.bm25 = BM25Index(chunks)
        self.candidates = candidates
        self.rrf_k = rrf_k

    def _rankings(self, query: str, vector: list[float]) -> list[list[str]]:
        keyword = [id_ for id_, _ in self.bm25.search(query, self.candidates)]
        semantic = [
            id_
            for id_, _ in self.store.query(vector, self.candidates)
            if id_ in self.chunks  # ignore stale vectors not in the current knowledge base
        ]
        return [keyword, semantic]

    def _similarities(self, ids: list[str], query_vectors: list[list[float]]) -> dict[str, float]:
        """Best cosine similarity of each chunk to any of the queries."""
        stored = self.store.embeddings(ids)
        return {
            id_: max(_cosine(stored[id_], qv) for qv in query_vectors)
            for id_ in ids
            if id_ in stored
        }

    def search(self, query: str, k: int = 6) -> list[RetrievedPassage]:
        """Top-`k` passages for one query."""
        return self.search_many([query], k)

    def search_many(self, queries: Sequence[str], k: int = 6) -> list[RetrievedPassage]:
        """Top-`k` passages for several queries (e.g. one per requirement), fused with RRF.

        Passages whose best cosine similarity to the queries is below `min_relevance` are
        dropped, so an off-topic requirement gets no passages (and the writer says "to be
        confirmed") instead of loosely related ones it would stretch into claims. With a
        reranker, the top `rerank_candidates` remaining passages are reordered by the LLM.
        """
        texts = [q for q in queries if q.strip()]
        vectors = [self.embedder.embed_query(q) for q in texts]
        rankings = [r for q, v in zip(texts, vectors, strict=True) for r in self._rankings(q, v)]
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)[: self.candidates]
        similarity = self._similarities([id_ for id_, _ in fused], vectors) if vectors else {}
        kept = [(i, s) for i, s in fused if similarity.get(i, 0.0) >= self.min_relevance]
        depth = max(k, self.rerank_candidates) if self.reranker else k
        passages = [
            RetrievedPassage(
                chunk_id=id_,
                source=self.chunks[id_].source,
                title=self.chunks[id_].title,
                text=self.chunks[id_].text,
                score=score,
                similarity=round(similarity.get(id_, 0.0), 4),
            )
            for id_, score in kept[:depth]
        ]
        if self.reranker:
            query = "\n".join(q for q in queries if q.strip())
            return self.reranker.rerank(query, passages, k)
        return passages


def build_retriever(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    rebuild: bool = False,
    usage: UsageTracker | None = None,
) -> HybridRetriever:
    """Load and chunk the knowledge base, sync the Chroma index, and return a retriever.

    With RERANK_ENABLED=true the writer LLM reranks fused candidates (calls tracked in `usage`).
    """
    chunks = load_knowledge_base(
        settings.knowledge_base_dir, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    if not chunks:
        raise ValueError(f"No .md/.txt documents found in {settings.knowledge_base_dir}")
    embedder = embedder or build_embedder(settings)
    store = VectorStore(
        settings.chroma_dir, settings.chroma_collection, embedder.model, reset=rebuild
    )
    sync_index(chunks, embedder, store)
    return HybridRetriever(
        chunks,
        embedder,
        store,
        candidates=settings.retrieval_candidates,
        rrf_k=settings.rrf_k,
        reranker=LLMReranker(build_llm(settings, "writer", usage=usage))
        if settings.rerank_enabled
        else None,
        rerank_candidates=settings.rerank_candidates,
        min_relevance=settings.min_relevance,
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
