"""Tests for settings and profile resolution."""

from __future__ import annotations

import pytest

from rfp_copilot.config import (
    ConfigError,
    Settings,
    get_settings,
    resolve_embedding_profile,
    resolve_llm_profile,
)


def test_ollama_profile_needs_no_key() -> None:
    profile = resolve_llm_profile(Settings(llm_provider="ollama"))
    assert profile.base_url == "http://localhost:11434/v1"
    assert profile.model == "qwen2.5:3b"
    assert profile.api_key == "ollama"


def test_gemini_without_key_fails_loudly() -> None:
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        resolve_llm_profile(Settings(llm_provider="gemini"))


def test_env_vars_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_API_KEY", "g-key")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash-lite")
    profile = resolve_llm_profile(get_settings())
    assert profile.provider == "gemini"
    assert profile.model == "gemini-2.5-flash-lite"
    assert profile.api_key == "g-key"
    assert "generativelanguage.googleapis.com" in profile.base_url


def test_reviewer_inherits_writer_when_unset() -> None:
    settings = Settings(llm_provider="gemini", llm_api_key="g-key", llm_model="m1")
    assert resolve_llm_profile(settings, "reviewer") == resolve_llm_profile(settings, "writer")


def test_reviewer_on_other_provider_does_not_inherit_url_model_or_key() -> None:
    settings = Settings(
        llm_provider="ollama",
        llm_model="qwen2.5:3b",
        reviewer_llm_provider="gemini",
        reviewer_llm_api_key="g-key",
    )
    reviewer = resolve_llm_profile(settings, "reviewer")
    assert reviewer.provider == "gemini"
    assert reviewer.model == "gemini-3.5-flash-lite"
    assert reviewer.api_key == "g-key"
    assert "googleapis" in reviewer.base_url


def test_writer_key_is_not_sent_to_another_provider() -> None:
    settings = Settings(llm_provider="openai", llm_api_key="sk-x", reviewer_llm_provider="gemini")
    with pytest.raises(ConfigError):
        resolve_llm_profile(settings, "reviewer")


def test_embedding_defaults_to_local_nomic() -> None:
    profile = resolve_embedding_profile(Settings())
    assert (profile.provider, profile.model) == ("ollama", "nomic-embed-text")


def test_gemini_embedding_reuses_llm_key_for_same_provider() -> None:
    settings = Settings(llm_provider="gemini", llm_api_key="g-key", embedding_provider="gemini")
    assert resolve_embedding_profile(settings).api_key == "g-key"


def test_ollama_base_url_moves_llm_and_embeddings_but_not_gemini() -> None:
    url = "http://host.docker.internal:11434/v1"
    local = Settings(llm_provider="ollama", ollama_base_url=url)
    assert resolve_llm_profile(local).base_url == url
    assert resolve_embedding_profile(local).base_url == url
    remote = Settings(llm_provider="gemini", llm_api_key="k", ollama_base_url=url)
    assert "googleapis" in resolve_llm_profile(remote).base_url


def test_eval_judge_overrides_only_the_judge() -> None:
    settings = Settings(llm_provider="gemini", llm_api_key="k", eval_judge_model="gemini-3.8-flash")
    assert resolve_llm_profile(settings, "reviewer").model == "gemini-3.5-flash-lite"
    assert resolve_llm_profile(settings, "judge").model == "gemini-3.8-flash"
    assert resolve_llm_profile(settings, "judge").api_key == "k"
    plain = Settings(llm_provider="gemini", llm_api_key="k", reviewer_llm_model="m-rev")
    assert resolve_llm_profile(plain, "judge").model == "m-rev"  # falls back to the reviewer
