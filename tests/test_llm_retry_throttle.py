"""Tests for exponential backoff, Retry-After handling and the RPM throttle."""

from __future__ import annotations

import random
from typing import Any

import openai
import pytest

from rfp_copilot.config import LLMProfile
from rfp_copilot.llm import LLMError, OpenAICompatClient, RateLimiter
from tests.fakes import REQUEST, FakeOpenAI, completion, status_error

PROFILE = LLMProfile(provider="gemini", base_url="http://test/v1", api_key="k", model="m")
MSGS = [{"role": "user", "content": "hi"}]


def _client(fake: FakeOpenAI, sleeps: list[float], **kwargs: Any) -> OpenAICompatClient:
    return OpenAICompatClient(
        PROFILE, client=fake, sleep=sleeps.append, rng=random.Random(0), **kwargs
    )


def test_retries_on_429_and_5xx_then_succeeds() -> None:
    fake = FakeOpenAI(
        [
            status_error(openai.RateLimitError, 429),
            status_error(openai.InternalServerError, 503),
            completion("done"),
        ]
    )
    sleeps: list[float] = []
    response = _client(fake, sleeps, backoff_base_s=1.0).complete(MSGS)
    assert response.content == "done"
    assert (response.prompt_tokens, response.completion_tokens) == (7, 3)
    assert len(fake.requests) == 3
    # attempt 0 -> delay in [0.5, 1], attempt 1 -> delay in [1, 2]
    assert 0.5 <= sleeps[0] <= 1.0
    assert 1.0 <= sleeps[1] <= 2.0


def test_retries_on_connection_error() -> None:
    fake = FakeOpenAI([openai.APIConnectionError(request=REQUEST), completion()])
    sleeps: list[float] = []
    assert _client(fake, sleeps).complete(MSGS).content == "ok"
    assert len(sleeps) == 1


def test_gives_up_after_max_retries() -> None:
    fake = FakeOpenAI([status_error(openai.RateLimitError, 429) for _ in range(3)])
    sleeps: list[float] = []
    with pytest.raises(LLMError, match="after 3 attempts"):
        _client(fake, sleeps, max_retries=2).complete(MSGS)
    assert len(fake.requests) == 3
    assert len(sleeps) == 2


def test_does_not_retry_client_errors() -> None:
    fake = FakeOpenAI([status_error(openai.AuthenticationError, 401)])
    sleeps: list[float] = []
    with pytest.raises(LLMError):
        _client(fake, sleeps).complete(MSGS)
    assert len(fake.requests) == 1
    assert sleeps == []


def test_honours_retry_after_header_with_cap() -> None:
    client = _client(FakeOpenAI([]), [], backoff_max_s=30.0)
    assert client.backoff_delay(
        0, status_error(openai.RateLimitError, 429, {"retry-after": "7"})
    ) == pytest.approx(7.0)
    assert client.backoff_delay(
        0, status_error(openai.RateLimitError, 429, {"retry-after": "500"})
    ) == pytest.approx(30.0)


def test_exponential_backoff_is_capped() -> None:
    client = _client(FakeOpenAI([]), [], backoff_base_s=2.0, backoff_max_s=10.0)
    assert 5.0 <= client.backoff_delay(10) <= 10.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_blocks_when_window_is_full() -> None:
    clock = FakeClock()
    limiter = RateLimiter(2, clock=clock, sleep=clock.sleep)
    assert limiter.acquire() == 0.0
    clock.now = 10.0
    assert limiter.acquire() == 0.0
    clock.now = 20.0
    waited = limiter.acquire()  # third call must wait until the first leaves the 60s window
    assert waited == pytest.approx(40.0)
    assert clock.now == pytest.approx(60.0)


def test_rate_limiter_disabled_when_zero() -> None:
    clock = FakeClock()
    limiter = RateLimiter(0, clock=clock, sleep=clock.sleep)
    for _ in range(100):
        limiter.acquire()
    assert clock.sleeps == []


def test_client_acquires_limiter_once_per_attempt() -> None:
    clock = FakeClock()
    limiter = RateLimiter(1, clock=clock, sleep=clock.sleep)
    fake = FakeOpenAI([status_error(openai.RateLimitError, 429), completion()])
    _client(fake, [], rate_limiter=limiter).complete(MSGS)
    assert clock.sleeps == [pytest.approx(60.0)]
