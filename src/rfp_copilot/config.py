"""Application settings, loaded only from environment variables (and an optional `.env`)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["gemini", "ollama", "openai", "mock"]
EmbeddingProvider = Literal["ollama", "gemini", "openai", "mock"]
LLMRole = Literal["writer", "reviewer", "judge"]
StructuredMode = Literal["auto", "json_schema", "json_object", "prompt"]

PROVIDER_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434/v1",
    "openai": "https://api.openai.com/v1",
    "mock": "mock://",
}
DEFAULT_LLM_MODELS: dict[str, str] = {
    "gemini": "gemini-3.5-flash-lite",
    "ollama": "qwen2.5:3b",
    "openai": "gpt-4.1-mini",
    "mock": "mock-model",
}
DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "ollama": "nomic-embed-text",
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
    "mock": "mock-hash-embedding",
}
# Providers that need no real API key (the OpenAI SDK still requires a non-empty string).
KEYLESS_PROVIDERS = {"ollama", "mock"}


class ConfigError(RuntimeError):
    """Raised when the configuration is incomplete or inconsistent."""


class Settings(BaseSettings):
    """All runtime configuration. Field `foo_bar` is read from env var `FOO_BAR`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Main (writer) LLM
    llm_provider: LLMProvider = "gemini"
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_temperature: float = 0.2

    # Optional reviewer overrides (fall back to the LLM_* values)
    reviewer_llm_provider: LLMProvider | None = None
    reviewer_llm_base_url: str | None = None
    reviewer_llm_api_key: SecretStr | None = None
    reviewer_llm_model: str | None = None

    # Optional evaluation judge (eval/ scripts only; falls back to the reviewer). Keeping it
    # separate lets evaluations use a strong, slow, low-quota model without slowing the app.
    eval_judge_provider: LLMProvider | None = None
    eval_judge_model: str | None = None
    eval_judge_api_key: SecretStr | None = None

    # Where the Ollama provider lives (both LLM and embeddings). In Docker use
    # http://host.docker.internal:11434/v1 so the container reaches Ollama on the host.
    ollama_base_url: str = PROVIDER_BASE_URLS["ollama"]

    # Embeddings
    embedding_provider: EmbeddingProvider = "ollama"
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None

    # Free-tier protection
    llm_max_rpm: int = 0  # 0 disables throttling
    llm_max_retries: int = 5
    llm_backoff_base_s: float = 2.0
    llm_backoff_max_s: float = 60.0
    llm_timeout_s: float = 120.0
    llm_cache_enabled: bool = True
    llm_cache_path: Path = Path(".cache/llm_cache.sqlite")
    llm_structured_mode: StructuredMode = "auto"
    llm_structured_max_retries: int = 2

    # Optional cost override (USD per million tokens) for providers not in the price table
    llm_price_input_per_mtok: float | None = None
    llm_price_output_per_mtok: float | None = None

    embedding_batch_size: int = 32
    embedding_cache_enabled: bool = True
    embedding_cache_path: Path = Path(".cache/embeddings.sqlite")

    # Knowledge base and retrieval
    knowledge_base_dir: Path = Path("data/knowledge_base")
    chroma_dir: Path = Path("chroma_db")
    chroma_collection: str = "knowledge_base"
    chunk_size: int = 800  # characters
    chunk_overlap: int = 120  # characters
    retrieval_k: int = 6  # passage budget per section, spread over its requirements
    retrieval_min_per_requirement: int = 2  # every requirement gets at least this many
    retrieval_max_per_section: int = 10
    retrieval_candidates: int = 20  # per retriever, before RRF fusion
    rrf_k: int = 60
    # Relevance floor (cosine similarity, embedding-model specific; calibrate with
    # eval/calibrate_relevance.py). 0.67 separates on- and off-domain queries for nomic-embed-text.
    min_relevance: float = 0.67
    rerank_enabled: bool = False  # listwise LLM reranking of fused candidates (1 call/query)
    rerank_candidates: int = 12

    # Pipeline
    extraction_batch_chars: int = 6000  # RFP text per extraction call (fits a 4k-token context)
    max_requirements_per_section: int = 6
    review_threshold: float = 0.75  # min (coverage + groundedness) / 2 per section
    max_iterations: int = 2  # write passes, including the first one

    # Storage
    runs_dir: Path = Path("runs")  # uploads, checkpoints, DOCX outputs, SQLite DB

    # Observability (optional)
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


class LLMProfile(BaseModel):
    """Fully resolved connection parameters for one LLM role."""

    provider: LLMProvider
    base_url: str
    api_key: str
    model: str


class EmbeddingProfile(BaseModel):
    """Fully resolved connection parameters for the embedding model."""

    provider: EmbeddingProvider
    base_url: str
    api_key: str
    model: str


def default_base_url(settings: Settings, provider: str) -> str:
    """Provider's default endpoint (Ollama's is configurable through OLLAMA_BASE_URL)."""
    return settings.ollama_base_url if provider == "ollama" else PROVIDER_BASE_URLS[provider]


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value is not None else None


def _resolve_key(provider: str, *candidates: str | None) -> str:
    for key in candidates:
        if key:
            return key
    if provider in KEYLESS_PROVIDERS:
        return provider
    raise ConfigError(
        f"No API key configured for provider '{provider}'. "
        "Set LLM_API_KEY (or the role-specific *_API_KEY) in your environment or .env file."
    )


def resolve_llm_profile(settings: Settings, role: LLMRole = "writer") -> LLMProfile:
    """Return the provider, URL, key and model to use for `role`.

    The reviewer inherits every unset field from the writer. The writer's API key is
    only reused when both roles use the same provider, so a Gemini key is never sent
    to OpenAI (or the reverse).
    """
    provider = settings.llm_provider
    base_url = settings.llm_base_url
    model = settings.llm_model
    key_candidates: list[str | None] = [_secret(settings.llm_api_key)]

    if role in ("reviewer", "judge"):
        reviewer_provider = settings.reviewer_llm_provider or provider
        if reviewer_provider != provider:
            # Different provider: do not inherit URL, model or key from the writer.
            base_url, model, key_candidates = None, None, []
        provider = reviewer_provider
        base_url = settings.reviewer_llm_base_url or base_url
        model = settings.reviewer_llm_model or model
        key_candidates = [_secret(settings.reviewer_llm_api_key), *key_candidates]

    if role == "judge":
        # Evaluation judge: inherits the reviewer, overridable without touching the pipeline.
        judge_provider = settings.eval_judge_provider or provider
        if judge_provider != provider:
            base_url, model, key_candidates = None, None, []
        provider = judge_provider
        model = settings.eval_judge_model or model
        key_candidates = [_secret(settings.eval_judge_api_key), *key_candidates]

    return LLMProfile(
        provider=provider,
        base_url=base_url or default_base_url(settings, provider),
        api_key=_resolve_key(provider, *key_candidates),
        model=model or DEFAULT_LLM_MODELS[provider],
    )


def runs_db_url(settings: Settings) -> str:
    """SQLAlchemy URL of the application database."""
    return f"sqlite:///{(settings.runs_dir / 'app.sqlite').as_posix()}"


def resolve_embedding_profile(settings: Settings) -> EmbeddingProfile:
    """Return the resolved embedding configuration."""
    provider = settings.embedding_provider
    candidates = [_secret(settings.embedding_api_key)]
    if provider == settings.llm_provider:
        candidates.append(_secret(settings.llm_api_key))
    return EmbeddingProfile(
        provider=provider,
        base_url=settings.embedding_base_url or default_base_url(settings, provider),
        api_key=_resolve_key(provider, *candidates),
        model=settings.embedding_model or DEFAULT_EMBEDDING_MODELS[provider],
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached; call `get_settings.cache_clear()` to reload)."""
    return Settings()
