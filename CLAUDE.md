# CLAUDE.md

Guidance for AI coding assistants working in this repo.

## What this is
RFP Copilot: a LangGraph pipeline that turns an RFP PDF into a cited DOCX draft response and a
requirements compliance table. Nodes: ingest -> extract_requirements -> retrieve -> write ->
review -> (deterministic loop back to write) -> human_review (interrupt) -> export_docx.

## Commands (Windows; use `.venv/bin/python` on Linux/macOS)
- Install: `uv pip install --python .venv\Scripts\python.exe -e ".[dev,tracing]"`
- Tests: `.venv\Scripts\python.exe -m pytest`
- Lint/format: `.venv\Scripts\python.exe -m ruff check .` / `... -m ruff format .`
- Regenerate lock: `uv pip compile pyproject.toml --extra dev --extra tracing --universal -o requirements-lock.txt`
- Index KB: `python -m rfp_copilot.cli index [--rebuild]`
- Headless run: `python -m rfp_copilot.cli run data/rfps/sample_rfp.pdf --auto-approve`
- Eval: `python eval/run_extraction_eval.py`, `python eval/run_rag_eval.py`, `python eval/compare_configs.py [--experiments ...] [--limit N]` (resumable; `--force` recomputes)
- MCP server (stdio): `python -m rfp_copilot.mcp_server --project-dir .`
- Docker: `docker compose up --build` (api + ui; Ollama stays on the host)
- API: `python -m uvicorn rfp_copilot.api.app:app --port 8010`; UI: `python -m streamlit run app/streamlit_app.py`

## Layout
- `config.py` (env settings), `llm.py` (the only LLM entry point), `models.py` (shared Pydantic models)
- `ingestion/` PDF -> sections with `[page N]` markers; requirement extraction + grouping
- `retrieval/` chunking, embeddings (+ disk cache), Chroma store (model-name guard), BM25+, RRF, hybrid retriever
- `graph/` state (plain dicts for the checkpointer), nodes (prompts + deterministic checks), graph wiring
- `compliance.py` deterministic per-requirement status/sources/notes (never an LLM); `textsim.py` dedup similarity
- `export/` DOCX (warnings, sections, matrix, checklist, sources); `db/` SQLAlchemy run/requirement summary; `pipeline.py` run lifecycle
- `mcp_server/` FastMCP tools reusing retrieval + DB; `tracing.py` optional Langfuse (no-op without keys)
- `evaluation/` gold models, metrics (P/R/F1, retrieval), LLM judges, resumable store, runners; `eval/*.py` are thin CLIs
- `retrieval/rerank.py` optional listwise LLM reranker (RERANK_ENABLED)
- `api/app.py` FastAPI (`create_app(service)` for tests); `app/streamlit_app.py` UI over the API
- Tests use `tests/fakes.py::make_pipeline_llm` (rule-based extractor/writer/reviewer) and `HashEmbedder`.

## Domain
Municipal-website RFPs only (city/county website redesign, CMS, hosting, support). The knowledge base
(`data/knowledge_base/`, fictional Northbridge Civic Web) must cover what those RFPs ask; do not copy
requirements from the downloaded evaluation RFPs into it (keeps the evaluation fair).
Hallucination guards: relevance floor (`MIN_RELEVANCE`, recalibrate with `eval/calibrate_relevance.py`),
per-requirement retrieval budget, figure check in `graph/nodes.py::unsupported_figures`.

## Hard constraints
- 8 GB RAM, limited disk, zero budget. NEVER add torch, transformers, sentence-transformers,
  Docling or LangChain integrations. Justify any new dependency and declare it in pyproject.toml
  with a version range, then regenerate `requirements-lock.txt`.
- ALL LLM calls go through `rfp_copilot.llm` (`get_llm(role)` / `build_llm(settings, role)`).
  Embeddings go through `rfp_copilot.retrieval.embeddings`. Never instantiate the OpenAI SDK
  anywhere else. Configuration comes only from env vars
  (`rfp_copilot.config.Settings`).
- Graph routing is deterministic Python, never decided by an LLM.
- Tests must run offline: use `MockLLM` and fakes, with no network and no Ollama.
- Never fabricate metrics. Eval scripts compute real numbers; leave README tables empty until run.
- Only public RFPs and synthetic data (the Gemini free tier may train on inputs).

## Conventions
- Python 3.11+, type hints everywhere, Pydantic models for data crossing module boundaries,
  docstrings on public functions. Small modules, ruff (line length 100).
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`...).
- Work proceeds in phases (see README); stop after each phase for review.
