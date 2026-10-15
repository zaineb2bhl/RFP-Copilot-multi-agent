"""Simple LLM-as-judge metrics: context precision and faithfulness.

RAGAS was not added: it pulls in LangChain integrations and many judge calls per sample,
which does not fit the free-tier budget. These judges use one structured call each.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rfp_copilot.evaluation.metrics import average_precision
from rfp_copilot.llm import BaseLLMClient, StructuredOutputError
from rfp_copilot.models import RetrievedPassage

CONTEXT_JUDGE_PROMPT = """You judge retrieval quality for an RFP response assistant.
A passage is RELEVANT if it contains concrete evidence that helps answer the requirement
(experience, certification, method, figure, person). Topic overlap alone is not enough.
Return the ids of the relevant passages only."""

FAITHFULNESS_JUDGE_PROMPT = """You check whether an answer is faithful to its sources.
List every factual claim the answer makes about the bidder (experience, figures,
certifications, people, methods, commitments backed by past practice). Ignore pure statements
of intent and "[TO CONFIRM: ...]" placeholders. For each claim, set supported=true only if the
passages state it (paraphrase is fine); otherwise supported=false."""


class RelevantPassages(BaseModel):
    """Context judge output."""

    relevant_ids: list[str] = Field(description="Ids of passages that are relevant.")


class Claim(BaseModel):
    """One factual claim and whether the passages support it."""

    claim: str
    supported: bool


class FaithfulnessVerdict(BaseModel):
    """Faithfulness judge output."""

    claims: list[Claim]


def _passages(passages: list[RetrievedPassage]) -> str:
    return "\n\n".join(f"[{p.chunk_id}] {p.title}\n{p.text}" for p in passages)


def judge_context_precision(
    judge: BaseLLMClient, query: str, passages: list[RetrievedPassage]
) -> dict[str, float | list[str] | None]:
    """LLM-judged precision@k and rank-aware average precision of the retrieved passages."""
    if not passages:
        return {"context_precision": None, "context_avg_precision": None, "relevant_ids": []}
    messages = [
        {"role": "system", "content": CONTEXT_JUDGE_PROMPT},
        {"role": "user", "content": f"Requirement:\n{query}\n\nPassages:\n{_passages(passages)}"},
    ]
    try:
        verdict = judge.complete_structured(
            messages, RelevantPassages, temperature=0.0, tag="judge_context"
        )
    except StructuredOutputError:
        return {"context_precision": None, "context_avg_precision": None, "relevant_ids": []}
    relevant = set(verdict.relevant_ids)
    flags = [p.chunk_id in relevant for p in passages]
    return {
        "context_precision": sum(flags) / len(flags),
        "context_avg_precision": average_precision(flags),
        "relevant_ids": [p.chunk_id for p in passages if p.chunk_id in relevant],
    }


def judge_faithfulness(
    judge: BaseLLMClient, answer: str, passages: list[RetrievedPassage]
) -> dict[str, float | int | None]:
    """Share of the answer's factual claims supported by the passages (None if no claims)."""
    messages = [
        {"role": "system", "content": FAITHFULNESS_JUDGE_PROMPT},
        {"role": "user", "content": f"Passages:\n{_passages(passages)}\n\nAnswer:\n{answer}"},
    ]
    try:
        verdict = judge.complete_structured(
            messages, FaithfulnessVerdict, temperature=0.0, tag="judge_faithfulness"
        )
    except StructuredOutputError:
        return {"faithfulness": None, "claims": 0, "supported_claims": 0}
    supported = sum(c.supported for c in verdict.claims)
    total = len(verdict.claims)
    return {
        "faithfulness": supported / total if total else None,
        "claims": total,
        "supported_claims": supported,
    }
