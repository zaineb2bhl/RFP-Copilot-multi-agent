"""Lexical text similarity shared by requirement deduplication and evaluation matching."""

from __future__ import annotations

import re

from rfp_copilot.retrieval.bm25 import tokenize


def normalise(text: str) -> str:
    """Lowercase alphanumerics separated by single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def dice(a: str, b: str) -> float:
    """Dice coefficient between the content-word sets of two texts (0 to 1)."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


def near_duplicates(a: str, b: str, *, threshold: float = 0.85, min_contained: int = 40) -> bool:
    """True if two requirement texts say the same thing.

    Either their content words overlap strongly, or the shorter text (at least
    `min_contained` characters, to avoid matching tiny fragments) is contained in the longer
    one, e.g. a summary-table entry that repeats the start of a main-body requirement.
    """
    if dice(a, b) >= threshold:
        return True
    short, long_ = sorted((normalise(a), normalise(b)), key=len)
    return len(short) >= min_contained and short in long_
