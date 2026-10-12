"""Reciprocal Rank Fusion (Cormack et al., 2009)."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank(d)), rank from 1.

    Only ranks matter, so BM25 scores and cosine similarities need no normalisation. Ties are
    broken by first appearance, which keeps the output deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    order = {doc_id: i for i, doc_id in enumerate(scores)}
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))
