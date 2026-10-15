"""Provider-agnostic LLM client.

Every LLM call in the project goes through `BaseLLMClient`. It adds, on top of any
OpenAI-compatible endpoint (Gemini, Ollama, OpenAI):

* a local SQLite response cache, so re-running tests and evals does not burn quota;
* a requests-per-minute throttle shared by all clients of the same endpoint;
* exponential backoff on 429 / 5xx / connection errors (honouring `Retry-After`);
* structured output: JSON schema or JSON mode when the provider supports it, otherwise
  prompt-only JSON, always followed by Pydantic validation and up to N retries that feed
  the validation error back to the model;
* per-call tracking of tokens, latency and cost.

`MockLLM` implements the same interface without any network access, for tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import openai
from pydantic import BaseModel, Field, ValidationError

from rfp_copilot.config import (
    LLMProfile,
    LLMRole,
    Settings,
    StructuredMode,
    get_settings,
    resolve_llm_profile,
)
from rfp_copilot.tracing import Tracer, get_tracer

logger = logging.getLogger(__name__)

Message = dict[str, str]
T = TypeVar("T", bound=BaseModel)

# USD per million tokens (input, output). Used only when no env override is set.
PRICES_PER_MTOK: dict[tuple[str, str], tuple[float, float]] = {
    ("openai", "gpt-4.1-mini"): (0.40, 1.60),
    ("openai", "gpt-4.1-nano"): (0.10, 0.40),
    ("openai", "gpt-4.1"): (2.00, 8.00),
}
# Assumes the Gemini *free* tier. Set LLM_PRICE_*_PER_MTOK if you use a paid tier.
FREE_PROVIDERS = {"gemini", "ollama", "mock"}

# Order in which structured-output strategies are tried when a provider rejects one.
_STRUCTURED_FALLBACK: dict[str, str] = {"json_schema": "json_object", "json_object": "prompt"}


class LLMError(RuntimeError):
    """A provider call failed (after retries, if the error was retryable)."""


class ResponseFormatUnsupportedError(LLMError):
    """The provider rejected the requested `response_format`."""


class QuotaExhaustedError(LLMError):
    """The provider's quota is exhausted for longer than it is worth waiting (e.g. daily)."""


class StructuredOutputError(LLMError):
    """The model never produced output that validates against the requested schema."""

    def __init__(self, message: str, *, raw_outputs: list[str], errors: list[str]) -> None:
        super().__init__(message)
        self.raw_outputs = raw_outputs
        self.errors = errors


class LLMResponse(BaseModel):
    """Text returned by one chat completion, with its usage metadata."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


class LLMCallRecord(BaseModel):
    """One tracked LLM call."""

    provider: str
    model: str
    tag: str | None = None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    cached: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UsageTracker:
    """Thread-safe in-memory log of LLM calls; can be shared by several clients."""

    def __init__(self) -> None:
        self._records: list[LLMCallRecord] = []
        self._lock = threading.Lock()

    def record(self, record: LLMCallRecord) -> None:
        """Append one call record."""
        with self._lock:
            self._records.append(record)

    @property
    def records(self) -> list[LLMCallRecord]:
        """A copy of all records so far."""
        with self._lock:
            return list(self._records)

    def summary(self) -> dict[str, float | int]:
        """Aggregate counts, tokens, cost and latency over all records."""
        records = self.records
        return {
            "calls": len(records),
            "cached_calls": sum(r.cached for r in records),
            "prompt_tokens": sum(r.prompt_tokens for r in records),
            "completion_tokens": sum(r.completion_tokens for r in records),
            "cost_usd": round(sum(r.cost_usd for r in records), 6),
            "latency_s": round(sum(r.latency_s for r in records), 3),
        }

    def reset(self) -> None:
        """Forget all records."""
        with self._lock:
            self._records.clear()


def compute_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    price_override: tuple[float, float] | None = None,
) -> float:
    """Return the USD cost of a call; 0 for free providers and unknown models."""
    if price_override is not None:
        price_in, price_out = price_override
    elif provider in FREE_PROVIDERS:
        return 0.0
    else:
        price_in, price_out = PRICES_PER_MTOK.get((provider, model), (0.0, 0.0))
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


class RateLimiter:
    """Sliding-window limiter: at most `max_per_minute` acquisitions in any 60 s window.

    `max_per_minute <= 0` disables throttling. `clock` and `sleep` are injectable for tests.
    """

    def __init__(
        self,
        max_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_per_minute = max_per_minute
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a call is allowed; return the number of seconds waited."""
        if self.max_per_minute <= 0:
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.max_per_minute:
                    self._calls.append(now)
                    return waited
                wait = 60.0 - (now - self._calls[0])
            logger.info("Throttling: waiting %.1fs to respect %d RPM", wait, self.max_per_minute)
            self._sleep(wait)
            waited += wait


_LIMITERS: dict[tuple[str, str, int], RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_rate_limiter(provider: str, base_url: str, max_per_minute: int) -> RateLimiter:
    """Return the limiter shared by all clients of the same endpoint (e.g. writer + reviewer)."""
    key = (provider, base_url, max_per_minute)
    with _LIMITERS_LOCK:
        if key not in _LIMITERS:
            _LIMITERS[key] = RateLimiter(max_per_minute)
        return _LIMITERS[key]


class ResponseCache:
    """SQLite cache of LLM responses keyed by a hash of provider, model, messages and params."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def make_key(
        provider: str, model: str, messages: Sequence[Message], params: dict[str, Any]
    ) -> str:
        """Deterministic SHA-256 key for a request."""
        payload = json.dumps(
            {"provider": provider, "model": model, "messages": list(messages), "params": params},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached value for `key`, or None."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store `value` under `key` (overwriting)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), datetime.now(UTC).isoformat()),
            )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> str:
    """Extract the JSON payload from a model answer (strips code fences and chatter).

    Small local models often wrap JSON in markdown fences or add a sentence around it.
    """
    text = text.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if text.startswith(("{", "[")):
        return text
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start : end + 1] if end > start else text[start:]


def _schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "Respond ONLY with a single JSON object (no markdown, no commentary) that conforms "
        "to this JSON schema:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )


def _with_schema_instruction(messages: Sequence[Message], schema: type[BaseModel]) -> list[Message]:
    instruction = _schema_instruction(schema)
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = f"{msgs[0]['content']}\n\n{instruction}"
    else:
        msgs.insert(0, {"role": "system", "content": instruction})
    return msgs


def _response_format(schema: type[BaseModel], mode: str) -> dict[str, Any] | None:
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
    if mode == "json_object":
        return {"type": "json_object"}
    return None


class BaseLLMClient(ABC):
    """Common logic (cache, usage tracking, structured output) for every LLM client."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        cache: ResponseCache | None = None,
        usage: UsageTracker | None = None,
        temperature: float = 0.2,
        structured_mode: StructuredMode = "auto",
        structured_max_retries: int = 2,
        price_override: tuple[float, float] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.cache = cache
        self.usage = usage if usage is not None else UsageTracker()
        self.temperature = temperature
        self.structured_mode: str = "json_schema" if structured_mode == "auto" else structured_mode
        self.structured_max_retries = structured_max_retries
        self.price_override = price_override
        self.tracer = tracer or Tracer()

    @abstractmethod
    def _call(self, messages: list[Message], params: dict[str, Any]) -> LLMResponse:
        """Perform one uncached provider call. `params` values may be None (= omit)."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> LLMResponse:
        """Return the model's answer to `messages`, using the cache when enabled.

        `tag` labels the call in usage records (e.g. "extract_requirements").
        """
        msgs = [dict(m) for m in messages]
        params: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        key = ResponseCache.make_key(self.provider, self.model, msgs, params)
        trace_params = {k: v for k, v in params.items() if v is not None and k != "response_format"}
        with self.tracer.generation(
            tag or "llm",
            model=self.model,
            input=msgs,
            metadata={"provider": self.provider, "params": trace_params},
        ) as generation:
            if self.cache is not None and (hit := self.cache.get(key)) is not None:
                response = LLMResponse(**{**hit, "cached": True, "latency_s": 0.0})
            else:
                response = self._call(msgs, params)
                if self.cache is not None:
                    self.cache.set(key, response.model_dump(exclude={"cached", "latency_s"}))
            generation.end(
                response.content,
                {"input": response.prompt_tokens, "output": response.completion_tokens},
                {"cached": response.cached, "latency_s": round(response.latency_s, 3)},
            )
        self._track(response, tag)
        return response

    def complete_structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tag: str | None = None,
        check: Callable[[T], None] | None = None,
    ) -> T:
        """Return the answer parsed and validated as `schema`.

        Uses the client's structured mode (JSON schema -> JSON mode -> prompt-only, falling
        back automatically if the provider rejects a format). On a validation failure the
        model is asked again, with its previous answer and the validation error, up to
        `structured_max_retries` times.

        `check` adds context-dependent validation (e.g. "exactly these requirement ids"): it
        raises ValueError with a message for the model, which triggers the same retry.
        """
        msgs = _with_schema_instruction(messages, schema)
        raw_outputs: list[str] = []
        errors: list[str] = []
        for _attempt in range(self.structured_max_retries + 1):
            response = self._complete_with_format_fallback(
                msgs, schema, temperature=temperature, max_tokens=max_tokens, tag=tag
            )
            raw_outputs.append(response.content)
            try:
                result = schema.model_validate_json(extract_json(response.content))
                if check is not None:
                    check(result)
                return result
            except ValidationError as exc:
                error = str(exc)
            except ValueError as exc:
                error = f"Invalid content: {exc}"
            errors.append(error)
            logger.warning("Structured output for %s failed validation: %s", schema.__name__, error)
            msgs = [
                *msgs,
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": (
                        "Your previous answer was not valid for the required JSON schema.\n"
                        f"Validation errors:\n{error}\n"
                        "Return ONLY the corrected JSON object."
                    ),
                },
            ]
        raise StructuredOutputError(
            f"No valid {schema.__name__} after {len(raw_outputs)} attempts",
            raw_outputs=raw_outputs,
            errors=errors,
        )

    def _complete_with_format_fallback(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> LLMResponse:
        while True:
            try:
                return self.complete(
                    messages,
                    response_format=_response_format(schema, self.structured_mode),
                    **kwargs,
                )
            except ResponseFormatUnsupportedError:
                fallback = _STRUCTURED_FALLBACK.get(self.structured_mode)
                if fallback is None:
                    raise
                logger.warning(
                    "%s rejected response_format=%s; falling back to %s",
                    self.provider,
                    self.structured_mode,
                    fallback,
                )
                self.structured_mode = fallback

    def _track(self, response: LLMResponse, tag: str | None) -> None:
        cost = (
            0.0
            if response.cached
            else compute_cost(
                self.provider,
                self.model,
                response.prompt_tokens,
                response.completion_tokens,
                self.price_override,
            )
        )
        self.usage.record(
            LLMCallRecord(
                provider=self.provider,
                model=self.model,
                tag=tag,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_s=response.latency_s,
                cost_usd=cost,
                cached=response.cached,
            )
        )


_RETRYABLE_ERRORS = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)
_FORMAT_ERROR_HINTS = ("response_format", "json_schema", "json schema", "schema", "json")
R = TypeVar("R")


class RetryPolicy:
    """Throttle + exponential backoff around provider calls (chat and embeddings).

    Retries on 429, 5xx and connection errors, honouring `Retry-After`. Any other SDK error
    is raised immediately as `LLMError`. `sleep` and `rng` are injectable for tests.
    """

    def __init__(
        self,
        max_retries: int = 5,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        *,
        rate_limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.rate_limiter = rate_limiter or RateLimiter(0)
        self._sleep = sleep
        self._rng = rng or random.Random()

    def delay(self, attempt: int, error: Exception | None = None) -> float:
        """Seconds to wait before retry number `attempt` (0-based).

        Uses the server's `Retry-After` header when present, else exponential backoff with
        jitter in [d/2, d] where d = base * 2**attempt, capped at `backoff_max_s`.
        """
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            return min(retry_after, self.backoff_max_s)
        delay = min(self.backoff_max_s, self.backoff_base_s * (2**attempt))
        return delay / 2 + self._rng.uniform(0, delay / 2)

    def run(self, fn: Callable[[], R], label: str) -> R:
        """Call `fn` (acquiring the rate limiter before each attempt) with retries."""
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.acquire()
            try:
                return fn()
            except _RETRYABLE_ERRORS as exc:
                wait = _server_retry_delay(exc)
                if wait is not None and wait > QUOTA_FAIL_FAST_S:
                    # A daily quota: retrying within this run cannot succeed.
                    raise QuotaExhaustedError(
                        f"{label}: free-tier quota exhausted; the provider asks to retry in "
                        f"{wait / 3600:.1f} h. Use another model or try again later."
                    ) from exc
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"{label} call failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                delay = self.delay(attempt, exc)
                logger.warning(
                    "%s error (%s); retry %d/%d in %.1fs",
                    label,
                    type(exc).__name__,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                self._sleep(delay)
            except openai.APIError as exc:
                raise LLMError(f"{label} call failed: {exc}") from exc
        raise AssertionError("unreachable")  # pragma: no cover


class OpenAICompatClient(BaseLLMClient):
    """Client for any OpenAI-compatible chat endpoint (Gemini, Ollama, OpenAI)."""

    def __init__(
        self,
        profile: LLMProfile,
        *,
        max_retries: int = 5,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        timeout_s: float = 120.0,
        rate_limiter: RateLimiter | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(profile.provider, profile.model, **kwargs)
        self.base_url = profile.base_url
        self.retry = RetryPolicy(
            max_retries,
            backoff_base_s,
            backoff_max_s,
            rate_limiter=rate_limiter,
            sleep=sleep,
            rng=rng,
        )
        # The SDK's own retries are disabled: RetryPolicy controls backoff and counts attempts.
        self._client = client or openai.OpenAI(
            base_url=profile.base_url, api_key=profile.api_key, timeout=timeout_s, max_retries=0
        )

    def backoff_delay(self, attempt: int, error: Exception | None = None) -> float:
        """Seconds to wait before retry number `attempt` (see `RetryPolicy.delay`)."""
        return self.retry.delay(attempt, error)

    def _call(self, messages: list[Message], params: dict[str, Any]) -> LLMResponse:
        request = {"model": self.model, "messages": messages}
        request.update({k: v for k, v in params.items() if v is not None})

        def attempt() -> LLMResponse:
            start = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(**request)
            except openai.BadRequestError as exc:
                message = str(exc).lower()
                if "response_format" in request and any(h in message for h in _FORMAT_ERROR_HINTS):
                    raise ResponseFormatUnsupportedError(str(exc)) from exc
                raise LLMError(f"{self.provider} rejected the request: {exc}") from exc
            usage = getattr(completion, "usage", None)
            return LLMResponse(
                content=completion.choices[0].message.content or "",
                model=getattr(completion, "model", None) or self.model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_s=time.perf_counter() - start,
            )

        return self.retry.run(attempt, self.provider)


QUOTA_FAIL_FAST_S = 300.0
_RETRY_DELAY_RE = re.compile(r"retryDelay'?\"?\s*:\s*'?\"?(\d+(?:\.\d+)?)s")


def _server_retry_delay(error: Exception) -> float | None:
    """Seconds the provider asks to wait, from the Retry-After header or the error body.

    Gemini reports daily-quota exhaustion as a 429 with `retryDelay: '31879s'` in the body.
    """
    header = _retry_after_seconds(error)
    if header is not None:
        return header
    match = _RETRY_DELAY_RE.search(str(error))
    return float(match.group(1)) if match else None


def _retry_after_seconds(error: Exception | None) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


MockHandler = Callable[[list[Message]], str]


class MockLLM(BaseLLMClient):
    """Offline client for tests: replays scripted responses and records every call.

    `responses` is either a list (consumed in order; items may be strings or handlers
    called with the messages) or a single handler used for every call.
    """

    def __init__(
        self,
        responses: Sequence[str | MockHandler] | MockHandler | None = None,
        *,
        model: str = "mock-model",
        **kwargs: Any,
    ) -> None:
        super().__init__("mock", model, **kwargs)
        self._handler: MockHandler | None = None
        self._queue: deque[str | MockHandler] = deque()
        if callable(responses):
            self._handler = responses
        else:
            self._queue.extend(responses or [])
        self.calls: list[list[Message]] = []
        self.params: list[dict[str, Any]] = []

    def _call(self, messages: list[Message], params: dict[str, Any]) -> LLMResponse:
        self.calls.append(messages)
        self.params.append(params)
        if self._handler is not None:
            item: str | MockHandler = self._handler
        elif self._queue:
            item = self._queue.popleft()
        else:
            raise LLMError("MockLLM has no scripted response left")
        content = item(messages) if callable(item) else item
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_chars // 4,
            completion_tokens=len(content) // 4,
        )


def build_llm(
    settings: Settings,
    role: LLMRole = "writer",
    *,
    usage: UsageTracker | None = None,
) -> BaseLLMClient:
    """Build the client for `role` from `settings` (no global state besides rate limiters)."""
    profile = resolve_llm_profile(settings, role)
    price_override = None
    if (
        settings.llm_price_input_per_mtok is not None
        or settings.llm_price_output_per_mtok is not None
    ):
        price_override = (
            settings.llm_price_input_per_mtok or 0.0,
            settings.llm_price_output_per_mtok or 0.0,
        )
    common: dict[str, Any] = {
        "cache": ResponseCache(settings.llm_cache_path) if settings.llm_cache_enabled else None,
        "usage": usage,
        "temperature": settings.llm_temperature,
        "structured_mode": settings.llm_structured_mode,
        "structured_max_retries": settings.llm_structured_max_retries,
        "price_override": price_override,
        "tracer": get_tracer(settings),
    }
    if profile.provider == "mock":
        return MockLLM(lambda _messages: "{}", model=profile.model, **common)
    return OpenAICompatClient(
        profile,
        max_retries=settings.llm_max_retries,
        backoff_base_s=settings.llm_backoff_base_s,
        backoff_max_s=settings.llm_backoff_max_s,
        timeout_s=settings.llm_timeout_s,
        rate_limiter=get_rate_limiter(profile.provider, profile.base_url, settings.llm_max_rpm),
        **common,
    )


def get_llm(role: LLMRole = "writer", *, usage: UsageTracker | None = None) -> BaseLLMClient:
    """Build the client for `role` from the process-wide settings."""
    return build_llm(get_settings(), role, usage=usage)
