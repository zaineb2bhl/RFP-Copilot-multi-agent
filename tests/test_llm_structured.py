"""Tests for structured output: JSON extraction, validation retries and format fallback."""

from __future__ import annotations

import json

import httpx
import openai
import pytest
from pydantic import BaseModel, Field

from rfp_copilot.config import LLMProfile
from rfp_copilot.llm import MockLLM, OpenAICompatClient, StructuredOutputError, extract_json
from tests.fakes import REQUEST, FakeOpenAI, completion


class Item(BaseModel):
    name: str
    score: float = Field(ge=0, le=1)


MSGS = [{"role": "system", "content": "You grade things."}, {"role": "user", "content": "go"}]


@pytest.mark.parametrize(
    "raw",
    [
        '{"name": "a", "score": 0.5}',
        '```json\n{"name": "a", "score": 0.5}\n```',
        'Sure! Here it is: {"name": "a", "score": 0.5} Hope this helps.',
    ],
)
def test_extract_json_handles_fences_and_chatter(raw: str) -> None:
    assert json.loads(extract_json(raw)) == {"name": "a", "score": 0.5}


def test_valid_first_answer() -> None:
    llm = MockLLM(['{"name": "a", "score": 0.9}'])
    assert llm.complete_structured(MSGS, Item) == Item(name="a", score=0.9)
    sent = llm.calls[0]
    assert sent[0]["role"] == "system" and "JSON schema" in sent[0]["content"]
    assert sent[0]["content"].startswith("You grade things.")
    assert llm.params[0]["response_format"]["type"] == "json_schema"


def test_retry_includes_validation_error() -> None:
    llm = MockLLM(['{"name": "a", "score": 3}', '{"name": "a", "score": 0.3}'])
    assert llm.complete_structured(MSGS, Item).score == 0.3
    retry_messages = llm.calls[1]
    assert retry_messages[-2] == {"role": "assistant", "content": '{"name": "a", "score": 3}'}
    assert "score" in retry_messages[-1]["content"]
    assert "less than or equal to 1" in retry_messages[-1]["content"]


def test_gives_up_after_max_retries() -> None:
    llm = MockLLM(["not json", '{"name": 1}', "{}"], structured_max_retries=2)
    with pytest.raises(StructuredOutputError) as info:
        llm.complete_structured(MSGS, Item)
    assert len(info.value.raw_outputs) == 3
    assert len(info.value.errors) == 3
    assert len(llm.calls) == 3


def test_prompt_mode_sends_no_response_format() -> None:
    llm = MockLLM(['{"name": "a", "score": 0}'], structured_mode="prompt")
    llm.complete_structured(MSGS, Item)
    assert llm.params[0]["response_format"] is None


def test_falls_back_when_provider_rejects_json_schema() -> None:
    rejection = openai.BadRequestError(
        "Invalid value for response_format: json_schema not supported",
        response=httpx.Response(400, request=REQUEST),
        body=None,
    )
    fake = FakeOpenAI([rejection, completion('{"name": "b", "score": 1}')])
    profile = LLMProfile(provider="gemini", base_url="http://test/v1", api_key="k", model="m")
    client = OpenAICompatClient(profile, client=fake, sleep=lambda _s: None)
    assert client.complete_structured(MSGS, Item).name == "b"
    assert fake.requests[0]["response_format"]["type"] == "json_schema"
    assert fake.requests[1]["response_format"] == {"type": "json_object"}
    assert client.structured_mode == "json_object"  # remembered for later calls
