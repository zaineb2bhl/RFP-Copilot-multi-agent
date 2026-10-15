# RFP Copilot

A multi-agent pipeline that takes a Request for Proposal (PDF) and produces a draft technical
response as DOCX. Every claim cites a knowledge-base source, and the output includes a
requirements compliance matrix. It runs for free: Gemini's free tier by default, or fully
offline with Ollama. It is designed for a laptop with 8 GB of RAM: no torch, no transformers.

**Domain: municipal website RFPs.** The tool is specialised in RFPs from cities and counties
for website redesign, CMS, hosting and support. Answer quality depends on the knowledge base
covering what RFPs in that domain ask for. `data/knowledge_base/` describes a fictional
government-website company, Northbridge Civic Web, in 23 files: CMS features and modules,
accessibility, hosting and security, integrations, migration, training, support, four past
projects, references and CVs. To use the tool for real, replace it with your own company's
documents in the same domain.

> **Data policy:** the Gemini free tier may use prompts and responses to improve Google's
> models. Only use **public RFPs** and **synthetic** knowledge-base data with that profile.

## Architecture

```mermaid
flowchart LR
    PDF[/RFP PDF/] --> ingest
    subgraph graph [LangGraph pipeline, SQLite checkpointer]
        ingest[ingest<br/>PyMuPDF sections + pages] --> extract[extract_requirements<br/>structured LLM output]
        extract --> retrieve[retrieve<br/>BM25+ and vectors, RRF]
        retrieve --> write[write<br/>cited drafts]
        write --> review[review<br/>LLM judge + hard checks]
        review -- "score < threshold<br/>and passes left" --> write
        review -- otherwise --> human[human_review<br/>interrupt]
        human -- approve + edits --> export[export_docx<br/>sections + compliance matrix]
        human -- reject --> stop((end))
    end
    KB[(knowledge base<br/>markdown)] --> chroma[(Chroma)]
    KB --> bm25[BM25+]
    chroma --> retrieve
    bm25 --> retrieve
    export --> DOCX[/DOCX/]

    llm{{rfp_copilot.llm<br/>cache, throttle, backoff,<br/>validation retries, usage}}
    extract -.-> llm
    write -.-> llm
    review -.-> llm
    llm -.-> providers[Gemini / Ollama / OpenAI]

    api[FastAPI] --> graph
    ui[Streamlit] --> api
    mcp[MCP server<br/>stdio] --> chroma
    mcp --> db[(SQLite<br/>runs + requirements)]
    graph --> db
```

All LLM calls go through one wrapper (`rfp_copilot.llm`). Routing after review is plain
Python, never an LLM decision. The writer and the reviewer can use different models.

## Install from scratch

Requires Python 3.11+. [uv](https://docs.astral.sh/uv/) is recommended but optional.

**Windows (PowerShell)**

```powershell
git clone <repo-url> rfp-copilot; cd rfp-copilot
py -3.11 -m venv .venv                 # or: uv venv --python 3.11 .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
copy .env.example .env                 # then edit .env (see below)
.venv\Scripts\python.exe -m pytest     # offline: no API key or Ollama needed
```

**Linux / macOS**

```bash
git clone <repo-url> rfp-copilot && cd rfp-copilot
python3.11 -m venv .venv               # or: uv venv --python 3.11 .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env                   # then edit .env (see below)
.venv/bin/python -m pytest
```

Optional extra: `tracing` (Langfuse), e.g. `pip install -e ".[dev,tracing]"`.
`requirements-lock.txt` pins the exact versions this project was tested with (this
project's dependencies only):
`pip install -r requirements-lock.txt && pip install -e . --no-deps`.

## Configure an LLM profile

Profiles are switched with environment variables only (see `.env.example`).

**Gemini (default, free tier).** Use it for real runs and evaluations.
1. Create a key at [AI Studio](https://aistudio.google.com/apikey).
2. In `.env`: `LLM_PROVIDER=gemini`, `LLM_API_KEY=<key>`, `LLM_MODEL=gemini-3.5-flash-lite`,
   `LLM_MAX_RPM=10` (keep it at or below your free-tier quota).
3. Embeddings stay local: `ollama pull nomic-embed-text` (or set `EMBEDDING_PROVIDER=gemini`,
   `EMBEDDING_MODEL=gemini-embedding-001`).

Model availability changes over time. `gemini-2.5-flash` was already unavailable to new keys
when this was built. List the models your key can use with
`GET https://generativelanguage.googleapis.com/v1beta/openai/models`. On the free tier,
`gemini-3.8-flash` measured roughly 40 s per call, versus about 1 s for
`gemini-3.5-flash-lite`.

**Ollama (offline, dev, fallback).** This runs fully on your machine.
1. `ollama pull qwen2.5:3b` (about 2 GB of RAM) and `ollama pull nomic-embed-text`.
2. In `.env`: `LLM_PROVIDER=ollama`, `LLM_MODEL=qwen2.5:3b`, `LLM_MAX_RPM=0`, plus
   `LLM_TIMEOUT_S=600`, `LLM_MAX_RETRIES=1` and `EXTRACTION_BATCH_CHARS=3000`. On CPU, the
   3B model produced about 6 tokens/s here, and one 3,000-character extraction batch took
   about 77 s. A full run takes many minutes.
3. Ollama's OpenAI-compatible endpoint cannot set `num_ctx` per request. If long prompts get
   truncated, start the server with a larger default, e.g. `OLLAMA_CONTEXT_LENGTH=8192`
   (uses more RAM).

**OpenAI (optional, paid):** `LLM_PROVIDER=openai`, `LLM_API_KEY=sk-...`,
`LLM_MODEL=gpt-4.1-mini`.

**Mixing models:** `REVIEWER_LLM_PROVIDER` / `REVIEWER_LLM_MODEL` / `REVIEWER_LLM_API_KEY`
give the reviewer (LLM-as-judge) its own model, e.g. a local writer with a Gemini reviewer.

## Run it

All commands are run from the repo root with the venv's Python (`.venv\Scripts\python.exe` on
Windows, `.venv/bin/python` on Linux/macOS, shown below as `python`).

1. **Index the knowledge base** (23 markdown files). Re-running only embeds new or changed
   chunks:
   ```
   python -m rfp_copilot.cli index            # add --rebuild after changing EMBEDDING_MODEL
   ```
2. **Run headless** on the synthetic sample RFP:
   ```
   python -m rfp_copilot.cli sample-rfp       # writes data/rfps/sample_rfp.pdf
   python -m rfp_copilot.cli run data/rfps/sample_rfp.pdf --auto-approve
   ```
   The DOCX is written to `runs/outputs/<run_id>.docx`.
3. **Or use the UI**: start the API, then Streamlit, in two terminals:
   ```
   python -m uvicorn rfp_copilot.api.app:app --port 8010
   python -m streamlit run app/streamlit_app.py
   ```
   Upload a PDF, wait for the drafts, edit them, approve, then download the DOCX. API docs
   are at http://localhost:8010/docs.

Runtime state lives in `runs/` (uploads, LangGraph checkpoints, the SQLite app database, DOCX
outputs), `chroma_db/` (vectors) and `.cache/` (LLM and embedding caches). All of them are
gitignored and safe to delete.

## How it works

- **ingest:** PyMuPDF splits the PDF into sections, detecting headings from typography (larger
  or bold lines). Section text keeps `[page N]` markers so that pages can be cited.
- **extract_requirements:** structured extraction in batches sized for a 4k-token local model.
  Requirements are numbered `REQ-001`… and typed mandatory, desirable or administrative.
  Each page is checked against the PDF text, duplicates are dropped, and requirements are
  grouped by RFP section, with at most 6 per response section.
- **retrieve:** each requirement is searched on its own, with BM25+ and Chroma vector search
  fused by Reciprocal Rank Fusion.
  - Every requirement gets at least `RETRIEVAL_MIN_PER_REQUIREMENT` passages (default 2), and
    a section is capped at `RETRIEVAL_MAX_PER_SECTION` (default 10). Well-matched
    requirements therefore cannot crowd out the others' evidence.
  - A **relevance floor** (`MIN_RELEVANCE`, cosine similarity) drops passages that are not
    about the requirement, so the writer is never handed loosely related text to stretch.
  - An optional LLM reranker (`RERANK_ENABLED=true`) can reorder candidates.
- **write:** the writer returns one structured answer per requirement id (Pydantic): an
  answer built only from the retrieved passages, plus what is missing, if anything.
  - Every fact cites its passage (`[hosting-security#2]`) and figures are copied exactly.
  - Answers are validated: exactly the expected ids, each once, and no grouped answers such as
    "REQ-030 through REQ-050". An invalid answer is retried with the error message.
  - Gaps are written as `[TO CONFIRM: <missing information>]`.
  - The buyer is called by the issuer name detected in the RFP (e.g. "City of Ballwin"),
    never by a past client's name.
  - Administrative items (deadlines, copies, forms, page limits) never go to the LLM: they
    become bid-team checklist lines.
- **review:** an LLM judge scores coverage and groundedness (0–1). Deterministic checks then
  tighten the scores:
  - requirement ids the draft never mentions count as uncovered;
  - citations to passages that were not retrieved count as unsupported;
  - **figure check:** any number, date or percentage that is absent from the passages its
    sentence cites (and absent from the RFP itself) counts as unsupported;
  - each of the last two caps groundedness at 0.5, and a section with no citation at all gets
    groundedness 0, unless it contains only administrative requirements.
- **routing:** back to `write` (only for failing sections, with the reviewer's feedback) while
  any section scores below `REVIEW_THRESHOLD` (default 0.75) and fewer than `MAX_ITERATIONS`
  write passes (default 2) have run. Sections with no relevant passage are not rewritten,
  because without evidence a rewrite could only repeat "to be confirmed" or invent.
- **human_review:** a LangGraph `interrupt()` with a SQLite checkpointer. The run survives
  restarts, and the reviewer can approve (with edits) or reject.
- **compliance (deterministic):** each requirement's status comes from its own answer, with
  no LLM involved:
  - `addressed`: cited evidence;
  - `partial`: cited evidence plus a `[TO CONFIRM]` gap;
  - `not_covered`: a placeholder with no evidence;
  - `to_confirm`: an administrative item, or a commitment without evidence that only a person
    can verify;
  - `non_compliant`: the reviewer found that our documented offer is worse than required
    (e.g. RPO 24 h vs 4 h), shown as a visible warning.

  Sources are only the chunks cited in that answer, never chunks retrieved but unused.
- **export_docx:** writes the warnings for the bid team, the response sections, the
  compliance matrix (requirement | status | where addressed | sources | notes), the bid-team
  checklist and the cited sources.

## MCP server

A stdio MCP server (official Python SDK) exposes the same retrieval code and run database:

| Tool | Returns |
|---|---|
| `search_past_projects(query, k=5)` | knowledge-base passages (past projects, CVs, methodologies) with citation ids |
| `get_requirement_status(run_id)` | every requirement of a run with status, where addressed and sources |
| `list_runs(limit=20)` | recent runs with status and current stage |

Example client configuration (Claude Desktop `claude_desktop_config.json`, or a project
`.mcp.json` for Claude Code). Adjust the paths:

```json
{
  "mcpServers": {
    "rfp-copilot": {
      "command": "C:\\path\\to\\rfp-copilot\\.venv\\Scripts\\python.exe",
      "args": ["-m", "rfp_copilot.mcp_server", "--project-dir", "C:\\path\\to\\rfp-copilot"]
    }
  }
}
```

`--project-dir` is needed because MCP clients start servers from their own working
directory, while `.env`, `data/`, `runs/` and `chroma_db/` are relative to the repo.

## Tracing (optional, Langfuse)

Install the extra (`pip install -e ".[tracing]"`) and set `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` (plus `LANGFUSE_HOST` if you are not on the EU cloud). Each pipeline run
becomes a trace (session id = run id) with one span per graph node and one generation per LLM
call, recording prompt, output, model, token usage and cache hits. Without keys, tracing does
nothing. Langfuse errors are logged and never fail a run.

## Docker

> **Dockerfile and compose provided; not verified locally.** The image build (`docker build`,
> without running the containers) is checked in CI. `docker compose up` has not been tested.

```
docker compose up --build        # UI: http://localhost:8501, API: http://localhost:8010
```

Two services are built from one image: `api` (FastAPI) and `ui` (Streamlit). Ollama stays on
the host, and the container reaches it through `host.docker.internal` (set by
`OLLAMA_BASE_URL` in `compose.yaml`), so start `ollama serve` on the host first. Configuration
comes from `.env`. Runs, vectors and caches live in named volumes, and the knowledge base is
mounted read-only. The image installs the exact versions in `requirements-lock.txt`.

## Evaluation

The scripts in `eval/` compute real numbers from gold data in `data/eval/`. They respect the
free-tier throttle, reuse the LLM cache, and are resumable: each item is stored in
`eval/results/items/` and skipped on re-runs unless you pass `--force`. See
[`eval/README.md`](eval/README.md) for metric definitions, and
[`data/eval/README.md`](data/eval/README.md) for how to annotate public RFPs.

```
python eval/run_extraction_eval.py      # (a) requirement extraction P/R/F1 vs gold
python eval/run_rag_eval.py             # (b) retrieval + judged context precision / faithfulness
python eval/compare_configs.py          # (c) chunk size, reranking, reviewer loop, local vs Gemini
```

Every report includes (d) tokens, cost and time per document or query. Reports are written
to `eval/results/*.md` and `*.json`.

### Results

_Not run yet. These tables will be filled from `eval/results/` after the scripts have been run.
No number in this README is estimated or invented._

**Requirement extraction** (`eval/results/extraction.md`)

| Config | Docs | Precision | Recall | F1 | Type acc. | Tokens/doc | Time/doc (s) |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

**RAG quality** (`eval/results/rag.md`)

| Config | Hit@k | Src precision@k | MRR | Context precision (judge) | Faithfulness (judge) | Citation validity |
|---|---|---|---|---|---|---|
| | | | | | | |

**Config comparisons** (`eval/results/compare_*.md`)

| Experiment | Config A | Config B | Key metric A | Key metric B | Tokens A | Tokens B |
|---|---|---|---|---|---|---|
| chunk_size | 400 chars | 800 chars | | | | |
| rerank | off | on | | | | |
| reviewer_loop | 1 pass | write/review loop | | | | |
| model | Ollama qwen2.5:3b | Gemini Flash-Lite | | | | |

## Design decisions and trade-offs

- **One domain, one knowledge base.** A generic knowledge base produced many "to be confirmed"
  points and stretched citations on real RFPs: the retriever always returned something, and
  the writer made it fit. Specialising in municipal-website RFPs, with a knowledge base that
  covers what those RFPs ask for, fixes the cause rather than the symptom. The knowledge base
  was written from typical requirements in this domain, not copied from the downloaded RFPs,
  so those RFPs remain a fair test.
- **Hallucination guards are deterministic where possible.** A calibrated relevance floor
  (no passage is better than an off-topic one), a per-requirement evidence budget, and a
  figure check that needs no LLM catch the most damaging errors cheaply. The LLM judge comes
  on top of these.
- **The relevance floor is calibrated, not guessed.** `eval/calibrate_relevance.py` measures,
  on the labelled queries plus off-domain queries, where on-topic and off-topic passages
  separate. For `nomic-embed-text`, 0.67 kept a relevant passage for 24/24 queries and
  returned nothing for 6/6 off-domain queries. The margin is thin and specific to the
  model: re-run the script after changing `EMBEDDING_MODEL` or the knowledge base.
- **One OpenAI-compatible client for every provider.** Gemini, Ollama and OpenAI all expose
  the OpenAI API, so a single thin wrapper adds the cache, throttle, backoff, structured
  output and usage tracking. Provider quirks are handled by data rather than code paths: the
  structured-output mode falls back from JSON schema to JSON mode to prompt-only JSON.
- **Validation retries include the error.** Small local models often return near-miss JSON.
  The retry sends back their previous answer together with the Pydantic error, and up to 2
  retries are made.
- **Hybrid retrieval with RRF.** BM25 catches exact terms such as ISO numbers, product names
  and SLA figures; embeddings catch paraphrases. RRF fuses the two by rank only, so no score
  normalisation is needed. BM25+ replaced Okapi BM25 because Okapi gives zero IDF to terms
  that appear in half the documents, which drops real matches in a small knowledge base.
- **Retrieval once per requirement, fused per section.** Short, specific queries retrieve
  better than one long section-level query.
- **An LLM judge plus deterministic checks.** The judge is lenient and noisy. Code-level checks
  (uncovered requirement ids, unknown citations, missing citations) are cheap and reliable,
  so they cap the judge's scores. Routing uses only these numbers, never an LLM decision.
- **Only failing sections are rewritten,** with the previous draft and the feedback. This
  limits free-tier usage.
- **Graph state holds plain JSON dicts,** not Pydantic objects, so the SQLite checkpointer
  can always serialise them. Nodes validate dicts back into models.
- **Reranking with an LLM, not a cross-encoder.** A cross-encoder would need torch. The
  listwise LLM reranker costs one call per query and is off by default; whether it helps is
  measured by `compare_configs.py`.
- **The Chroma collection records its embedding model,** and the app refuses to mix vectors
  from different models, with a message telling you to re-index.
- **Evaluation stores what it measured.** Each item is saved with the latency and tokens
  measured on its first run, so re-runs do not overwrite real timings. The LLM cache can
  still serve a first run (for example, prompts already sent by the app), so every report
  shows the share of cached calls next to the times. Run with `LLM_CACHE_ENABLED=false
  --force` to get pure latency, at the cost of quota.

## Limitations

- **Scanned PDFs are not supported** (no OCR). Heading detection relies on typography and can
  merge or split sections on unusual layouts. Tables in RFPs are read as plain text.
- **Requirement extraction quality depends on the model.** A 3B local model misses or merges
  requirements more often. One RFP sentence may contain several requirements, and the gold
  annotation decides how they are split.
- **The tool is specialised.** Outside municipal-website RFPs, most requirements will come out
  "to be confirmed": by design, the relevance floor refuses off-topic evidence.
- **The figure check is lexical.** It catches invented numbers, but it cannot catch a wrong
  claim that contains no figure, or a figure reworded ("ninety-nine percent"). The LLM judge
  covers those, imperfectly.
- **Judge metrics are estimates.** Faithfulness, context precision and the reviewer score are
  produced by an LLM and can be wrong. In this project, `gemini-3.5-flash-lite` as a judge
  rated every claim of a run as supported, while `gemini-3.8-flash` found unsupported ones.
  Use the stronger model as the evaluation judge (`EVAL_JUDGE_MODEL`, which leaves the
  pipeline's own reviewer unchanged). Source-level retrieval metrics and extraction P/R/F1
  use human labels, but they are only as good as the annotations.
- **Gold data is small.** The sample RFP is synthetic and was written alongside the knowledge
  base, so it is easier than real RFPs. Real evaluation needs the public RFPs listed in
  `data/eval/README.md`, annotated by hand.
- **Free tiers are slow and limited.** The RPM throttle makes full runs take minutes, and
  daily quotas can stop a long evaluation. Because the scripts are resumable, re-running
  continues where they stopped.
- **The API runs pipelines in-process** (FastAPI background tasks). A restart loses runs that
  are in progress, though runs waiting for review survive thanks to the checkpointer. There
  is no authentication: it is a local tool.
- **Docker is only build-checked in CI;** the containers have not been run.

## Licensing note

PyMuPDF is licensed under AGPL-3.0, so distributing this project carries AGPL obligations.
