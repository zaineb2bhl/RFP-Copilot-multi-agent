"""Hybrid retriever (BM25 + embeddings fused with RRF) and knowledge-base index sync."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

from rfp_copilot.config import Settings
from rfp_copilot.models import Chunk, RetrievedPassage
from rfp_copilot.retrieval.bm25 import BM25Index
from rfp_copilot.retrieval.chunking import load_knowledge_base
from rfp_copilot.retrieval.embeddings import Embedder, build_embedder
from rfp_copilot.retrieval.fusion import reciprocal_rank_fusion
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
    ) -> None:
        self.chunks = {c.id: c for c in chunks}
        self.embedder = embedder
        self.store = store
        self.bm25 = BM25Index(chunks)
        self.candidates = candidates
        self.rrf_k = rrf_k

    def _rankings(self, query: str) -> list[list[str]]:
        keyword = [id_ for id_, _ in self.bm25.search(query, self.candidates)]
        semantic = [
            id_
            for id_, _ in self.store.query(self.embedder.embed_query(query), self.candidates)
            if id_ in self.chunks  # ignore stale vectors not in the current knowledge base
        ]
        return [keyword, semantic]

    def search(self, query: str, k: int = 6) -> list[RetrievedPassage]:
        """Top-`k` passages for one query."""
        return self.search_many([query], k)

    def search_many(self, queries: Sequence[str], k: int = 6) -> list[RetrievedPassage]:
        """Top-`k` passages for several queries (e.g. one per requirement), fused with RRF."""
        rankings = [r for q in queries if q.strip() for r in self._rankings(q)]
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)[:k]
        return [
            RetrievedPassage(
                chunk_id=id_,
                source=self.chunks[id_].source,
                title=self.chunks[id_].title,
                text=self.chunks[id_].text,
                score=score,
            )
            for id_, score in fused
        ]


def build_retriever(
    settings: Settings, *, embedder: Embedder | None = None, rebuild: bool = False
) -> HybridRetriever:
    """Load and chunk the knowledge base, sync the Chroma index, and return a retriever."""
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
    )
