"""Optional Langfuse tracing that degrades to no-ops.

Tracing is enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set and the
`langfuse` package (extra `[tracing]`) is installed. Tracing errors are logged and swallowed:
observability must never break a pipeline run.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from rfp_copilot.config import Settings

logger = logging.getLogger(__name__)


class Generation:
    """Handle for one traced LLM call; `end` records its output and token usage."""

    def end(self, output: str, usage: dict[str, int], metadata: dict[str, Any]) -> None:
        """Record the result (no-op by default)."""


class Tracer:
    """No-op tracer. Subclasses send spans and generations to a backend."""

    enabled = False

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Iterator[None]:
        """Trace a block of work (a pipeline run or a graph node)."""
        yield

    @contextlib.contextmanager
    def generation(
        self, name: str, *, model: str, input: Any, metadata: dict[str, Any] | None = None
    ) -> Iterator[Generation]:
        """Trace one LLM call."""
        yield Generation()

    def flush(self) -> None:
        """Send buffered events (no-op by default)."""


class _LangfuseGeneration(Generation):
    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def end(self, output: str, usage: dict[str, int], metadata: dict[str, Any]) -> None:
        try:
            self._observation.update(output=output, usage_details=usage, metadata=metadata)
        except Exception:  # never let tracing break a call
            logger.warning("Langfuse generation update failed", exc_info=True)


class LangfuseTracer(Tracer):
    """Sends spans and generations to Langfuse (v3 SDK, OpenTelemetry based)."""

    enabled = True

    def __init__(self, client: Any) -> None:
        self._client = client

    @contextlib.contextmanager
    def _safe(self, factory: Any) -> Iterator[Any]:
        """Enter a Langfuse context manager, falling back to None if Langfuse fails."""
        try:
            manager = factory()
            observation = manager.__enter__()
        except Exception:
            logger.warning("Langfuse tracing failed; continuing without it", exc_info=True)
            yield None
            return
        try:
            yield observation
        except BaseException as exc:
            with contextlib.suppress(Exception):
                manager.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            with contextlib.suppress(Exception):
                manager.__exit__(None, None, None)

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Iterator[None]:
        factory = lambda: self._client.start_as_current_span(  # noqa: E731
            name=name, input=input, metadata=metadata
        )
        with self._safe(factory) as observation:
            if observation is not None and session_id:
                with contextlib.suppress(Exception):
                    observation.update_trace(name=name, session_id=session_id)
            yield

    @contextlib.contextmanager
    def generation(
        self, name: str, *, model: str, input: Any, metadata: dict[str, Any] | None = None
    ) -> Iterator[Generation]:
        factory = lambda: self._client.start_as_current_generation(  # noqa: E731
            name=name, model=model, input=input, metadata=metadata
        )
        with self._safe(factory) as observation:
            yield _LangfuseGeneration(observation) if observation is not None else Generation()

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            logger.warning("Langfuse flush failed", exc_info=True)


@lru_cache(maxsize=4)
def _langfuse_tracer(public_key: str, secret_key: str, host: str) -> Tracer:
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning(
            "LANGFUSE_* keys are set but langfuse is not installed: pip install -e .[tracing]"
        )
        return Tracer()
    try:
        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except Exception:
        logger.warning("Could not initialise Langfuse; tracing disabled", exc_info=True)
        return Tracer()
    logger.info("Langfuse tracing enabled (%s)", host)
    return LangfuseTracer(client)


def get_tracer(settings: Settings) -> Tracer:
    """The tracer for these settings: Langfuse if fully configured, otherwise a no-op."""
    secret = settings.langfuse_secret_key.get_secret_value() if settings.langfuse_secret_key else ""
    if not (settings.langfuse_public_key and secret):
        return Tracer()
    return _langfuse_tracer(settings.langfuse_public_key, secret, settings.langfuse_host)
