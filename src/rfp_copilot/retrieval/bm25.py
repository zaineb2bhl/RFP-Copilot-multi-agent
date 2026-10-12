"""Keyword retrieval with BM25 (rank_bm25), rebuilt in memory from the chunks."""

from __future__ import annotations

import re

from rank_bm25 import BM25Plus

from rfp_copilot.models import Chunk

_STOPWORDS_TEXT = (
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were will with shall must should may can our your their we you they all any each"
)
_STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without English stopwords."""
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """BM25+ over chunk title + text.

    BM25+ is used instead of Okapi BM25 because Okapi's IDF is zero or negative for terms that
    occur in half the documents or more, which in a small knowledge base drops real matches.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._ids = [c.id for c in chunks]
        corpus = [tokenize(f"{c.title} {c.text}") for c in chunks]
        self._vocab = [set(doc) for doc in corpus]
        self._bm25 = BM25Plus(corpus) if corpus else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to `k` (chunk id, score) pairs sharing a term with the query, best first."""
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        query_terms = set(tokens)
        ranked = sorted(
            (
                (id_, float(score))
                for id_, score, vocab in zip(self._ids, scores, self._vocab, strict=True)
                if vocab & query_terms  # BM25+ gives every document a positive floor score
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]
