# RFP Copilot

A multi-agent pipeline that takes a Request for Proposal (PDF) and produces a draft technical
response as DOCX, with cited sources and a requirements compliance table. It runs for free:
Gemini's free tier by default, or fully offline with Ollama.

> **Status:** the pipeline runs end to end (ingestion, extraction, hybrid retrieval,
> write/review loop, human review, DOCX export, API, UI), with an MCP server, optional
> Langfuse tracing and Docker. Still to come: evaluation scripts (Phase 6).

> **Data policy:** the Gemini free tier may use prompts and responses to improve Google's
> models. Only use **public RFPs** and **synthetic** knowledge-base data with that profile.

## Install from scratch

Requires Python 3.11+. [uv](https://docs.astral.sh/uv/) is recommended but optional.

**Windows (PowerShell)**

```powershell
git clone <repo-url> rfp-copilot; cd rfp-copilot
py -3.11 -m venv .venv                 # or: uv venv --python 3.11 .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
copy .env.example .env                 # then edit .env
.venv\Scripts\python.exe -m pytest
```

**Linux / macOS**

```bash
git clone <repo-url> rfp-copilot && cd rfp-copilot
python3.11 -m venv .venv               # or: uv venv --python 3.11 .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env                   # then edit .env
.venv/bin/python -m pytest
```

Optional extras: `tracing` (Langfuse). For example: `pip install -e ".[dev,tracing]"`.
`requirements-lock.txt` pins the exact versions this project was tested with (this
project's dependencies only):
`pip install -r requirements-lock.txt && pip install -e . --no-deps`.

## Run it

All commands below are run from the repo root with the venv's Python
(`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on Linux/macOS, shown as `python`).

1. **Configure** `.env` (copy `.env.example`). Default: Gemini for the LLM and local Ollama
   for embeddings (`ollama pull nomic-embed-text`).
2. **Index the knowledge base** (`data/knowledge_base/`, 15 synthetic markdown files about the
   fictional company Northbridge Digital). Re-running only embeds new or changed chunks:
   ```
   python -m rfp_copilot.cli index            # add --rebuild after changing EMBEDDING_MODEL
   ```
3. **Try it headless** on the synthetic sample RFP:
   ```
   python -m rfp_copilot.cli sample-rfp       # writes data/rfps/sample_rfp.pdf
   python -m rfp_copilot.cli run data/rfps/sample_rfp.pdf --auto-approve
   ```
   The DOCX lands in `runs/outputs/<run_id>.docx`.
4. **Or use the UI**: start the API, then Streamlit, in two terminals:
   ```
   python -m uvicorn rfp_copilot.api.app:app --port 8010
   python -m streamlit run app/streamlit_app.py
   ```
   Upload a PDF, wait for the drafts, edit them, approve, and download the DOCX.
   API docs: http://localhost:8010/docs.

Runtime state lives in `runs/` (uploads, LangGraph checkpoints, the SQLite app database, DOCX
outputs), `chroma_db/` (vectors) and `.cache/` (LLM and embedding caches). All of them are
gitignored and safe to delete.

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

`--project-dir` matters because MCP clients start servers from their own working directory,
and the project's `.env`, `data/`, `runs/` and `chroma_db/` are relative to the repo.

## Tracing (optional, Langfuse)

Install the extra (`pip install -e ".[tracing]"`) and set `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY` (and `LANGFUSE_HOST` if not on the EU cloud). Each pipeline run becomes
a trace (session id = run id) with one span per graph node and one generation per LLM call:
prompt, output, model, token usage, and whether it came from the cache. Without keys, tracing
is a no-op. Langfuse errors are logged and never fail a run.

## Docker

> **Dockerfile and compose provided; not verified locally.** The image build (`docker build`,
> without running the containers) is checked in CI. `docker compose up` has not been tested.

```
docker compose up --build        # UI: http://localhost:8501, API: http://localhost:8010
```

There are two services built from one image: `api` (FastAPI) and `ui` (Streamlit). Ollama stays
on the host. The container reaches it via `host.docker.internal` (set through
`OLLAMA_BASE_URL` in `compose.yaml`), so start `ollama serve` on the host first. Configuration
comes from `.env`. Runs, vectors and caches live in named volumes, and the knowledge base is
mounted read-only from `data/knowledge_base`. The image installs the exact versions in
`requirements-lock.txt`.

## How it works

```
ingest -> extract_requirements -> retrieve -> write -> review --(score < threshold and passes left)--> write
                                                          \--(otherwise)--> human_review (interrupt) -> export_docx
```

- **ingest:** PyMuPDF parses sections from typography (larger or bold headings) and keeps
  `[page N]` markers.
- **extract_requirements:** batched structured extraction (`REQ-001`...). Pages are checked
  against the PDF text, duplicates are dropped, and requirements are grouped by RFP section.
- **retrieve:** for each section, BM25+ and Chroma vector search run once per requirement,
  and the results are fused with Reciprocal Rank Fusion.
- **write:** drafts each section only from retrieved passages, citing `[source_id]` (e.g.
  `[proj-city-portal#2]`).
- **review:** an LLM judge scores coverage and groundedness (0-1). Deterministic checks then
  tighten the scores: requirement ids missing from the draft count as uncovered, citations
  to non-retrieved passages count as unsupported, and a draft without citations gets
  groundedness 0.
- **routing:** plain Python: back to `write` (only failing sections, with the feedback) while
  any section scores below `REVIEW_THRESHOLD` (default 0.75) and fewer than `MAX_ITERATIONS`
  write passes (default 2) have run.
- **human_review:** LangGraph `interrupt()` with a SQLite checkpointer. Approve with edits,
  or reject.
- **export_docx:** response sections, a compliance matrix (requirement | status | where
  addressed | sources) and the list of cited sources.

## LLM profiles

You switch profiles with environment variables only (see `.env.example`):

| Profile | Use | Setup |
|---|---|---|
| `gemini` (default) | real runs, evals | free API key from [AI Studio](https://aistudio.google.com/apikey); set `LLM_API_KEY`, `LLM_MAX_RPM` |
| `ollama` | dev, offline, fallback | `ollama pull qwen2.5:3b` and `ollama pull nomic-embed-text` |
| `openai` | optional, paid | `LLM_API_KEY=sk-...`, `LLM_MODEL=gpt-4.1-mini` |

The reviewer (LLM-as-judge) can use a different model through `REVIEWER_LLM_*` variables,
for example a local writer with a Gemini reviewer.

Every call goes through `rfp_copilot.llm`. It adds a SQLite response cache (`.cache/`,
disable with `LLM_CACHE_ENABLED=false`), a requests-per-minute throttle, exponential backoff
on 429/5xx, structured output with Pydantic validation and retries, and token, latency and
cost tracking.

**Ollama context window:** Ollama's OpenAI-compatible endpoint cannot set `num_ctx` per
request. If prompts get truncated, raise the server default, e.g.
`OLLAMA_CONTEXT_LENGTH=8192` (more RAM), before starting `ollama serve`.

## Licensing note

PyMuPDF is licensed under AGPL-3.0, so distributing this project carries AGPL obligations.

## Evaluation results

_Not yet run._ Results will be filled in from `eval/results/` once the evaluation scripts
(Phase 6) have been run on real data.
