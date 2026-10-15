"""Optional listwise LLM reranker (no cross-encoder: no torch on this machine).

One structured LLM call per query orders the fused candidates by relevance. Ids the model
omits or invents are handled deterministically, so the output is always a permutation-prefix
of the input.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from rfp_copilot.llm import BaseLLMClient, LLMError
from rfp_copilot.models import RetrievedPassage

logger = logging.getLogger(__name__)

RERANK_SYSTEM_PROMPT = """You rank knowledge-base passages by how useful they are to answer an
RFP requirement with concrete, verifiable evidence. Return the ids of the passages, most useful
first. Include only ids from the list."""

_SNIPPET_CHARS = 400


class RerankResult(BaseModel):
    """LLM output: passage ids, most relevant first."""

    ranked_ids: list[str] = Field(description="Passage ids ordered from most to least useful.")


class LLMReranker:
    """Reorders retrieved passages with one LLM call."""

    def __init__(self, llm: BaseLLMClient) -> None:
        self.llm = llm

    def rerank(
        self, query: str, passages: list[RetrievedPassage], k: int
    ) -> list[RetrievedPassage]:
        """Return the `k` most relevant passages; falls back to the input order on failure."""
        if len(passages) <= 1:
            return passages[:k]
        listing = "\n\n".join(
            f"[{p.chunk_id}] {p.title}\n{p.text[:_SNIPPET_CHARS]}" for p in passages
        )
        messages = [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": f"Requirement:\n{query}\n\nPassages:\n{listing}"},
        ]
        try:
            result = self.llm.complete_structured(
                messages, RerankResult, temperature=0.0, tag="rerank"
            )
        except LLMError as exc:
            logger.warning("Reranking failed, keeping fused order: %s", exc)
            return passages[:k]
        by_id = {p.chunk_id: p for p in passages}
        ordered = [by_id[i] for i in dict.fromkeys(result.ranked_ids) if i in by_id]
        ordered += [p for p in passages if p not in ordered]  # omitted ids keep fused order
        return ordered[:k]
