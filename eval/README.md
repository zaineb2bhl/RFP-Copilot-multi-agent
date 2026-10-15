# Evaluation

Run every script from the repo root with the venv's Python. All of them read `.env`, go
through the project's LLM wrapper (RPM throttle, retry, SQLite cache), and store each
computed item in `eval/results/items/` (gitignored). A re-run skips stored items, so an
interrupted or quota-limited evaluation continues where it stopped; `--force` recomputes.
Reports are written as `eval/results/<name>.md` (tables) and `.json` (all rows plus metadata).

| Script | Measures | Report |
|---|---|---|
| `run_extraction_eval.py` | (a) requirement extraction vs gold | `extraction.md` |
| `run_rag_eval.py` | (b) retrieval and generation quality | `rag.md` |
| `compare_configs.py` | (c) config comparisons | `compare_<experiment>.md` |
| `calibrate_relevance.py` | similarity of relevant vs off-domain passages, to set `MIN_RELEVANCE` | printed |
| `false_abstentions.py <run_id>` | `[TO CONFIRM]` answers although a chunk retrieved for that requirement scored above the threshold, for manual review (no LLM) | `false_abstentions_<run_id>.md` |
| `measure_run.py <run_id>` | judge-estimated unsupported claims of one run (resumable) | printed |

Every report includes (d): tokens, cost (0 on free tiers) and wall time per document or query.

## Data

- `data/eval/*.gold.json`: hand-annotated requirements, one file per RFP (format and
  guidelines in `data/eval/README.md`). `sample_rfp.gold.json` covers the synthetic sample
  RFP (generate it with `python -m rfp_copilot.cli sample-rfp`).
- `data/eval/rag_queries.json`: 24 municipal-website RFP queries over the synthetic knowledge
  base, each labelled with the files that contain the evidence.

## Metrics

**(a) Extraction**
- **Matching:** a predicted requirement matches a gold one when the Dice similarity of their
  content words (stopwords removed) is at least 0.5 (`--threshold`). Matching is one-to-one
  and greedy, best pair first.
- **Precision, recall and F1** are micro-averaged over documents.
- **Type accuracy and page accuracy** are computed over matched pairs only.

**(b) RAG**: for each query, the top `RETRIEVAL_K` chunks are retrieved and scored.
- **Labelled metrics (no LLM):** source-level hit@k, precision@k (share of chunks from a
  relevant file), recall@k (share of relevant files found) and MRR.
- **Context precision (LLM judge):** the share of retrieved chunks the judge rates as
  relevant evidence, plus a rank-aware average precision in the style of RAGAS.
- **Answer generation:** an answer is generated with the pipeline's exact writer prompts.
  - *Citation validity:* the share of cited ids that were actually retrieved.
  - *Faithfulness (LLM judge):* the share of the answer's factual claims supported by the
    passages.
- **The judge** is `EVAL_JUDGE_MODEL` from `.env` (otherwise the reviewer profile,
  otherwise the main LLM). It is built from the base settings, so it stays the same across compared configs.

RAGAS was not used: it brings LangChain integrations and makes several judge calls per sample,
which the free-tier budget cannot afford. Each judge here is a single structured call.

**(c) Experiments** (`--experiments` takes a comma-separated subset)

| Experiment | Configs | Evaluated with |
|---|---|---|
| `chunk_size` | 400/60 vs 800/120 characters (size/overlap) | RAG eval |
| `rerank` | without vs with listwise LLM reranking | RAG eval |
| `reviewer_loop` | `MAX_ITERATIONS=1` vs `2` | full pipeline on the gold RFPs (auto-approved): judge faithfulness of final drafts, the pipeline reviewer's own score, addressed share |
| `model` | writer = Gemini (from `.env`) vs Ollama `qwen2.5:3b` | extraction + RAG eval |

Each config indexes into its own Chroma collection under `.cache/eval_chroma`, so the app's
index is never modified. Embeddings are cached on disk, so re-indexing is cheap.

## Practical notes

- **Budget, for the 24 queries:** the RAG eval makes about 3 LLM calls per query (answer plus
  2 judge calls), plus 1 rerank call when reranking is on. At `LLM_MAX_RPM=10` that is roughly
  6 minutes per config. Use `--limit` for a quick check and `--no-judge` for labelled metrics
  only.
- **The reviewer loop only matters when sections fail review.** On the easy synthetic RFP,
  every section may pass on the first try, which makes both configs identical. Use real RFPs,
  or raise `REVIEW_THRESHOLD` for this experiment.
- **Latency includes throttling waits.** Stored items keep the numbers measured on their first
  run, but that first run can itself be served from the LLM cache (~0 s), for example when the
  app already sent the same prompt. Reports therefore show the cached-call share next to the
  times. For pure latency, run with `LLM_CACHE_ENABLED=false` and `--force` (this spends
  quota).
- **Judge metrics are LLM estimates** and should be read with that in mind. The labelled
  metrics are exact with respect to the annotations.
