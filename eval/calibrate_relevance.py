"""Calibrate MIN_RELEVANCE (the retrieval relevance floor) for the configured embedding model.

    python eval/calibrate_relevance.py

For each labelled query it computes the cosine similarity between the query and every chunk,
split into chunks from relevant files and from other files, and also the best similarity of
off-domain queries (which should retrieve nothing). It prints the distributions and, for
candidate thresholds, how many queries keep at least one relevant chunk and how many
off-domain queries would wrongly get passages. Pick the threshold to put in .env.
"""

from __future__ import annotations

import math
import statistics

from _common import RAG_QUERIES, setup

from rfp_copilot.evaluation.gold import load_rag_queries
from rfp_copilot.retrieval.retriever import build_retriever

# Requirements a municipal-website company cannot answer: the floor should reject them.
OFF_DOMAIN = [
    "Provide preventive maintenance of HVAC equipment in city buildings.",
    "Supply ten police patrol vehicles with emergency lighting packages.",
    "Asphalt resurfacing and pothole repair of residential streets.",
    "Actuarial valuation services for the municipal employees pension fund.",
    "Janitorial services for city hall, five days a week.",
    "Provide armored car cash pickup services for city facilities.",
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)


def main() -> None:
    settings = setup()
    retriever = build_retriever(settings)
    ids = list(retriever.chunks)
    vectors = retriever.store.embeddings(ids)
    queries = load_rag_queries(RAG_QUERIES)

    best_relevant, best_other, off_domain = [], [], []
    for q in queries:
        qv = retriever.embedder.embed_query(q.query)
        sims = {i: cosine(qv, vectors[i]) for i in ids}
        relevant = [s for i, s in sims.items() if retriever.chunks[i].source in q.relevant_sources]
        other = [s for i, s in sims.items() if retriever.chunks[i].source not in q.relevant_sources]
        best_relevant.append(max(relevant))
        best_other.append(max(other))
    for text in OFF_DOMAIN:
        qv = retriever.embedder.embed_query(text)
        off_domain.append(max(cosine(qv, vectors[i]) for i in ids))

    def describe(name: str, values: list[float]) -> None:
        median = statistics.median(values)
        print(f"{name:42} min {min(values):.3f}  median {median:.3f}  max {max(values):.3f}")

    print(
        f"embedding model: {retriever.embedder.model}, {len(ids)} chunks, {len(queries)} queries\n"
    )
    describe("best relevant chunk per query", best_relevant)
    describe("best non-relevant chunk per query", best_other)
    describe("best chunk for off-domain queries", off_domain)
    print("\nthreshold | queries keeping a relevant chunk | off-domain queries with passages")
    for t in [x / 100 for x in range(40, 81, 5)]:
        kept = sum(s >= t for s in best_relevant)
        leaked = sum(s >= t for s in off_domain)
        queries_kept = f"{kept:>2}/{len(best_relevant)}"
        print(f"   {t:.2f}   | {queries_kept:<32} | {leaked}/{len(off_domain)}")


if __name__ == "__main__":
    main()
