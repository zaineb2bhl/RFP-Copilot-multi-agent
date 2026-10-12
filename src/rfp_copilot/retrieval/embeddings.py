"""Embedding clients: OpenAI-compatible (Ollama, Gemini, OpenAI) and an offline hash embedder.

Embeddings are batched and cached on disk by a hash of (provider, model, text), so
re-indexing an unchanged knowledge base costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import openai

from rfp_copilot.config import EmbeddingProfile, Settings, resolve_embedding_profile
from rfp_copilot.llm import RetryPolicy, get_rate_limiter

Vector = list[float]


class EmbeddingCache:
    """SQLite cache of embedding vectors keyed by sha256(provider, model, text)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
            )

    @staticmethod
    def make_key(provider: str, model: str, text: str) -> str:
        """Deterministic key for one text."""
        return hashlib.sha256(f"{provider}\x00{model}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, Vector]:
        """Return the cached vectors among `keys`."""
        found: dict[str, Vector] = {}
        with sqlite3.connect(self.path, timeout=30) as conn:
            for start in range(0, len(keys), 500):  # stay under SQLite's variable limit
                batch = list(keys[start : start + 500])
                marks = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({marks})", batch
                ).fetchall()
                found.update({key: json.loads(vector) for key, vector in rows})
        return found

    def set_many(self, items: dict[str, Vector]) -> None:
        """Store vectors."""
        with sqlite3.connect(self.path, timeout=30) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                [(k, json.dumps(v)) for k, v in items.items()],
            )


class Embedder(ABC):
    """Turns texts into vectors. Documents and queries may be embedded differently."""

    provider: str
    model: str

    @abstractmethod
    def _embed(self, texts: list[str]) -> list[Vector]:
        """Embed texts exactly as given."""

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed passages to be indexed."""
        return self._embed([self._document_prefix + t for t in texts])

    def embed_query(self, text: str) -> Vector:
        """Embed a search query."""
        return self._embed([self._query_prefix + text])[0]

    @property
    def _document_prefix(self) -> str:
        # nomic-embed-text was trained with task prefixes and retrieves noticeably better with them.
        return "search_document: " if self.model.startswith("nomic-embed") else ""

    @property
    def _query_prefix(self) -> str:
        return "search_query: " if self.model.startswith("nomic-embed") else ""


class OpenAICompatEmbedder(Embedder):
    """Embeddings through an OpenAI-compatible `/embeddings` endpoint, batched and cached."""

    def __init__(
        self,
        profile: EmbeddingProfile,
        *,
        batch_size: int = 32,
        cache: EmbeddingCache | None = None,
        retry: RetryPolicy | None = None,
        client: Any | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.provider = profile.provider
        self.model = profile.model
        self.batch_size = batch_size
        self.cache = cache
        self.retry = retry or RetryPolicy()
        self._client = client or openai.OpenAI(
            base_url=profile.base_url, api_key=profile.api_key, timeout=timeout_s, max_retries=0
        )

    def _embed(self, texts: list[str]) -> list[Vector]:
        keys = [EmbeddingCache.make_key(self.provider, self.model, t) for t in texts]
        vectors: dict[str, Vector] = self.cache.get_many(keys) if self.cache else {}
        missing = list(dict.fromkeys(k for k in keys if k not in vectors))
        text_by_key = dict(zip(keys, texts, strict=True))
        for start in range(0, len(missing), self.batch_size):
            batch_keys = missing[start : start + self.batch_size]
            batch = [text_by_key[k] for k in batch_keys]
            response = self.retry.run(
                lambda batch=batch: self._client.embeddings.create(model=self.model, input=batch),
                f"{self.provider} embeddings",
            )
            new = {
                k: list(item.embedding)
                for k, item in zip(
                    batch_keys, sorted(response.data, key=lambda d: d.index), strict=True
                )
            }
            if self.cache:
                self.cache.set_many(new)
            vectors.update(new)
        return [vectors[k] for k in keys]


class HashEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder: offline, for tests and the `mock` profile.

    Texts that share words get similar vectors, which is enough to exercise retrieval logic.
    """

    def __init__(self, dim: int = 256) -> None:
        self.provider = "mock"
        self.model = f"mock-hash-embedding-{dim}" if dim != 256 else "mock-hash-embedding"
        self.dim = dim

    def _embed(self, texts: list[str]) -> list[Vector]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                vec[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


def build_embedder(settings: Settings) -> Embedder:
    """Build the configured embedder (cached on disk unless EMBEDDING_CACHE_ENABLED=false)."""
    profile = resolve_embedding_profile(settings)
    if profile.provider == "mock":
        return HashEmbedder()
    cache = (
        EmbeddingCache(settings.embedding_cache_path) if settings.embedding_cache_enabled else None
    )
    retry = RetryPolicy(
        settings.llm_max_retries,
        settings.llm_backoff_base_s,
        settings.llm_backoff_max_s,
        rate_limiter=get_rate_limiter(
            profile.provider,
            profile.base_url,
            0 if profile.provider == "ollama" else settings.llm_max_rpm,
        ),
    )
    return OpenAICompatEmbedder(
        profile,
        batch_size=settings.embedding_batch_size,
        cache=cache,
        retry=retry,
        timeout_s=settings.llm_timeout_s,
    )
