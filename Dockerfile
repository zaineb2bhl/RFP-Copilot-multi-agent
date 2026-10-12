# One image for both services (API and Streamlit UI). Ollama is NOT containerised:
# the container reaches the host's Ollama through host.docker.internal (see compose.yaml).
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

WORKDIR /app

# Dependencies first (cached layer): exact versions from the lock file.
COPY requirements-lock.txt ./
RUN uv pip install --system -r requirements-lock.txt

# Then the project itself, without re-resolving dependencies.
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-deps .

COPY app ./app
COPY data/knowledge_base ./data/knowledge_base

RUN useradd --create-home --uid 1000 app \
    && mkdir -p runs chroma_db .cache \
    && chown -R app:app /app
USER app

EXPOSE 8010 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8010/health', timeout=4)" || exit 1

CMD ["uvicorn", "rfp_copilot.api.app:app", "--host", "0.0.0.0", "--port", "8010"]
