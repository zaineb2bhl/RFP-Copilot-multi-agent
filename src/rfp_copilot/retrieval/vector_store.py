"""Persistent ChromaDB collection that remembers which embedding model built it."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_KEY = "embedding_model"


class EmbeddingModelMismatchError(RuntimeError):
    """The collection was built with a different embedding model than the configured one."""


class VectorStore:
    """Thin wrapper over a Chroma collection with cosine distance and caller-supplied vectors."""

    def __init__(
        self, persist_dir: Path | str, collection: str, embedding_model: str, *, reset: bool = False
    ) -> None:
        """Open (or create) the collection; `reset=True` drops it first, e.g. to switch models."""
        self.embedding_model = embedding_model
        self.collection_name = collection
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        if reset:
            with contextlib.suppress(NotFoundError):
                self._client.delete_collection(collection)
        try:
            self._collection = self._client.get_collection(collection, embedding_function=None)
        except NotFoundError:
            self._collection = self._create()
            return
        indexed_with = (self._collection.metadata or {}).get(EMBEDDING_MODEL_KEY)
        if indexed_with != embedding_model:
            raise EmbeddingModelMismatchError(
                f"Chroma collection '{collection}' in {persist_dir} was built with embedding model "
                f"'{indexed_with}', but EMBEDDING_MODEL is '{embedding_model}'. Vectors from "
                "different models are not comparable. Re-index with:\n"
                "    python -m rfp_copilot.cli index --rebuild"
            )

    def _create(self) -> chromadb.Collection:
        return self._client.create_collection(
            self.collection_name,
            metadata={EMBEDDING_MODEL_KEY: self.embedding_model},
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def content_hashes(self) -> dict[str, str]:
        """Map of chunk id -> content hash for everything currently indexed."""
        result = self._collection.get(include=["metadatas"])
        return {
            id_: (meta or {}).get("hash", "")
            for id_, meta in zip(result["ids"], result["metadatas"] or [], strict=False)
        }

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        hashes: Sequence[str],
    ) -> None:
        """Insert or replace vectors (documents live in the knowledge base, not here)."""
        if ids:
            self._collection.upsert(
                ids=list(ids),
                embeddings=[list(e) for e in embeddings],
                metadatas=[{"hash": h} for h in hashes],
            )

    def delete(self, ids: Sequence[str]) -> None:
        """Remove vectors by id."""
        if ids:
            self._collection.delete(ids=list(ids))

    def count(self) -> int:
        """Number of indexed vectors."""
        return self._collection.count()

    def query(self, embedding: Sequence[float], k: int) -> list[tuple[str, float]]:
        """Return up to `k` (id, cosine similarity) pairs, best first."""
        n = min(k, self.count())
        if n == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(embedding)], n_results=n, include=["distances"]
        )
        return [
            (id_, 1.0 - distance)
            for id_, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]
