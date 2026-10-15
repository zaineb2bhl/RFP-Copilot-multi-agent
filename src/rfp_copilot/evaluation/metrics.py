"""Deterministic metrics: requirement extraction P/R/F1 and source-level retrieval scores."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel

from rfp_copilot.evaluation.gold import GoldRequirement
from rfp_copilot.models import Requirement
from rfp_copilot.textsim import dice

DEFAULT_MATCH_THRESHOLD = 0.5


def text_similarity(a: str, b: str) -> float:
    """Dice coefficient between the content-word sets of two texts (0 to 1)."""
    return dice(a, b)


def match_requirements(
    predicted: Sequence[str], gold: Sequence[str], threshold: float = DEFAULT_MATCH_THRESHOLD
) -> list[tuple[int, int, float]]:
    """One-to-one matching of predicted to gold texts: (pred index, gold index, similarity).

    Pairs with similarity >= `threshold` are taken greedily, best first, so each prediction
    and each gold requirement is used at most once. Ties break on index order.
    """
    pairs = sorted(
        (
            (sim, pi, gi)
            for pi, p in enumerate(predicted)
            for gi, g in enumerate(gold)
            if (sim := text_similarity(p, g)) >= threshold
        ),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches = []
    for sim, pi, gi in pairs:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matches.append((pi, gi, sim))
    return sorted(matches, key=lambda m: m[1])


class ExtractionScores(BaseModel):
    """Requirement extraction quality against a gold annotation."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    type_accuracy: float | None  # among matched requirements
    page_accuracy: float | None  # among matched requirements
    type_correct: int = 0
    page_correct: int = 0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_extraction(
    predicted: Sequence[Requirement],
    gold: Sequence[GoldRequirement],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> ExtractionScores:
    """Precision/recall/F1 of extracted requirements, plus type and page accuracy of matches."""
    matches = match_requirements([p.text for p in predicted], [g.text for g in gold], threshold)
    tp = len(matches)
    type_correct = sum(predicted[pi].type == gold[gi].type for pi, gi, _ in matches)
    page_correct = sum(predicted[pi].source_page == gold[gi].source_page for pi, gi, _ in matches)
    precision, recall, f1 = _prf(tp, len(predicted) - tp, len(gold) - tp)
    return ExtractionScores(
        tp=tp,
        fp=len(predicted) - tp,
        fn=len(gold) - tp,
        precision=precision,
        recall=recall,
        f1=f1,
        type_accuracy=type_correct / tp if tp else None,
        page_accuracy=page_correct / tp if tp else None,
        type_correct=type_correct,
        page_correct=page_correct,
    )


def micro_average(scores: Iterable[ExtractionScores]) -> ExtractionScores:
    """Pool counts over documents (micro-averaged P/R/F1 and accuracies)."""
    scores = list(scores)
    tp, fp, fn = (sum(getattr(s, f) for s in scores) for f in ("tp", "fp", "fn"))
    type_correct = sum(s.type_correct for s in scores)
    page_correct = sum(s.page_correct for s in scores)
    precision, recall, f1 = _prf(tp, fp, fn)
    return ExtractionScores(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        type_accuracy=type_correct / tp if tp else None,
        page_accuracy=page_correct / tp if tp else None,
        type_correct=type_correct,
        page_correct=page_correct,
    )


class RetrievalScores(BaseModel):
    """Source-level retrieval quality for one query (relevance labelled per file)."""

    hit: float  # 1 if any retrieved chunk comes from a relevant source
    precision: float  # share of retrieved chunks from relevant sources
    recall: float  # share of relevant sources present among the retrieved chunks
    reciprocal_rank: float  # 1 / rank of the first relevant chunk (0 if none)


def score_retrieval(retrieved_sources: Sequence[str], relevant: set[str]) -> RetrievalScores:
    """Score a ranked list of retrieved chunk sources against the relevant source files."""
    flags = [s in relevant for s in retrieved_sources]
    first = next((i for i, f in enumerate(flags, start=1) if f), None)
    return RetrievalScores(
        hit=1.0 if first else 0.0,
        precision=sum(flags) / len(flags) if flags else 0.0,
        recall=len(set(retrieved_sources) & relevant) / len(relevant) if relevant else 0.0,
        reciprocal_rank=1 / first if first else 0.0,
    )


def average_precision(relevance: Sequence[bool]) -> float:
    """Rank-aware context precision (as in RAGAS): mean of precision@i at relevant ranks."""
    hits = 0
    total = 0.0
    for i, relevant in enumerate(relevance, start=1):
        if relevant:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def mean(values: Iterable[float | None]) -> float | None:
    """Mean of the non-None values, or None if there are none."""
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None
