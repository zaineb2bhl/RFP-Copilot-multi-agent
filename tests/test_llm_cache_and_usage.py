"""Tests for the response cache, usage tracking and cost computation."""

from __future__ import annotations

from rfp_copilot.config import Settings
from rfp_copilot.llm import (
    MockLLM,
    OpenAICompatClient,
    ResponseCache,
    UsageTracker,
    build_llm,
    compute_cost,
)

MSGS = [{"role": "user", "content": "hello"}]


def test_cache_key_is_deterministic_and_sensitive() -> None:
    key = ResponseCache.make_key("ollama", "m", MSGS, {"temperature": 0.2})
    assert key == ResponseCache.make_key("ollama", "m", MSGS, {"temperature": 0.2})
    assert key != ResponseCache.make_key("gemini", "m", MSGS, {"temperature": 0.2})
    assert key != ResponseCache.make_key("ollama", "m2", MSGS, {"temperature": 0.2})
    assert key != ResponseCache.make_key("ollama", "m", MSGS, {"temperature": 0.7})
    other = [{"role": "user", "content": "hello!"}]
    assert key != ResponseCache.make_key("ollama", "m", other, {"temperature": 0.2})


def test_cache_hit_skips_provider_call(tmp_path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    llm = MockLLM(["first answer"], cache=cache)
    assert llm.complete(MSGS).content == "first answer"
    second = llm.complete(MSGS)  # the mock has no responses left: must come from the cache
    assert second.content == "first answer"
    assert second.cached is True
    assert len(llm.calls) == 1


def test_cache_persists_across_instances(tmp_path) -> None:
    path = tmp_path / "cache.sqlite"
    MockLLM(["persisted"], cache=ResponseCache(path)).complete(MSGS)
    fresh = MockLLM([], cache=ResponseCache(path))
    assert fresh.complete(MSGS).content == "persisted"


def test_without_cache_every_call_hits_provider() -> None:
    llm = MockLLM(["a", "b"])
    assert [llm.complete(MSGS).content for _ in range(2)] == ["a", "b"]


def test_cache_disabled_via_settings(tmp_path) -> None:
    settings = Settings(llm_provider="mock", llm_cache_enabled=False)
    assert build_llm(settings).cache is None
    settings = Settings(llm_provider="mock", llm_cache_path=tmp_path / "c.sqlite")
    assert build_llm(settings).cache is not None


def test_usage_tracking_counts_tokens_and_cache_hits(tmp_path) -> None:
    usage = UsageTracker()
    llm = MockLLM(["x" * 40], cache=ResponseCache(tmp_path / "c.sqlite"), usage=usage)
    llm.complete(MSGS, tag="t1")
    llm.complete(MSGS, tag="t1")
    summary = usage.summary()
    assert summary["calls"] == 2
    assert summary["cached_calls"] == 1
    assert summary["completion_tokens"] == 20  # 10 tokens per call, cached calls included
    assert summary["cost_usd"] == 0.0
    assert {r.tag for r in usage.records} == {"t1"}


def test_shared_tracker_between_clients() -> None:
    usage = UsageTracker()
    MockLLM(["a"], usage=usage).complete(MSGS)
    MockLLM(["b"], usage=usage).complete(MSGS)
    assert usage.summary()["calls"] == 2


def test_compute_cost() -> None:
    assert compute_cost("gemini", "gemini-3.5-flash-lite", 10_000, 10_000) == 0.0
    assert compute_cost("ollama", "qwen2.5:3b", 10_000, 10_000) == 0.0
    assert compute_cost("openai", "gpt-4.1-mini", 1_000_000, 1_000_000) == 2.0
    assert compute_cost("gemini", "x", 1_000_000, 0, price_override=(0.1, 0.4)) == 0.1


def test_build_llm_uses_reviewer_profile() -> None:
    settings = Settings(
        llm_provider="ollama",
        reviewer_llm_provider="gemini",
        reviewer_llm_api_key="g-key",
        llm_cache_enabled=False,
    )
    writer, reviewer = build_llm(settings, "writer"), build_llm(settings, "reviewer")
    assert isinstance(writer, OpenAICompatClient) and isinstance(reviewer, OpenAICompatClient)
    assert (writer.provider, writer.model) == ("ollama", "qwen2.5:3b")
    assert (reviewer.provider, reviewer.model) == ("gemini", "gemini-3.5-flash-lite")
