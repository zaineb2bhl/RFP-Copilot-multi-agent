"""Test doubles for the OpenAI SDK (no network)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import openai

REQUEST = httpx.Request("POST", "http://test/v1/chat/completions")


def status_error(
    cls: type[openai.APIStatusError], status: int, headers: dict[str, str] | None = None
) -> openai.APIStatusError:
    """Build an SDK HTTP error such as `RateLimitError` with the given status and headers."""
    response = httpx.Response(status, request=REQUEST, headers=headers or {})
    return cls(f"HTTP {status}", response=response, body=None)


def completion(text: str = "ok", prompt_tokens: int = 7, completion_tokens: int = 3) -> Any:
    """Build an object shaped like an SDK `ChatCompletion`."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        model="m",
    )


class FakeOpenAI:
    """Mimics `openai.OpenAI().chat.completions.create`, replaying scripted outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> Any:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# ---------------------------------------------------------------------------
# A rule-based fake of the three pipeline LLM roles (extract, write, review)
# ---------------------------------------------------------------------------

_REQ_WORDS = re.compile(r"\b(must|shall|should)\b", re.IGNORECASE)
_PAGE = re.compile(r"^\[page (\d+)\]$")


def fake_extract(user: str) -> str:
    """Return every line containing must/shall/should as a requirement, with its page."""
    page = 1
    items = []
    for line in user.splitlines():
        if match := _PAGE.match(line.strip()):
            page = int(match.group(1))
        elif _REQ_WORDS.search(line):
            kind = "desirable" if re.search(r"\bshould\b", line, re.I) else "mandatory"
            items.append({"text": line.strip(), "type": kind, "source_page": page})
    return json.dumps({"requirements": items})


def fake_write(user: str) -> str:
    """Mention every requirement id in the prompt and cite the first passage."""
    req_ids = list(
        dict.fromkeys(re.findall(r"REQ-\d{3}", user.split("Knowledge-base passages")[0]))
    )
    cites = re.findall(r"^\[([^\]]+#\d+)\]", user, re.MULTILINE)
    cite = f" [{cites[0]}]" if cites else ""
    return "\n".join(f"- **{rid}**: we comply, based on our experience{cite}." for rid in req_ids)


class FakeReviewer:
    """Scores each section 0.4 on its first review and 0.9 afterwards (forces one revision)."""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}

    def __call__(self, user: str) -> str:
        title = re.search(r"Draft section '([^']*)'", user)
        key = title.group(1) if title else ""
        self.seen[key] = self.seen.get(key, 0) + 1
        score = 0.4 if self.seen[key] == 1 else 0.9
        return json.dumps(
            {
                "coverage": score,
                "groundedness": score,
                "uncovered_requirements": [],
                "unsupported_claims": [],
                "feedback": "Be more specific." if score < 0.75 else "Good.",
            }
        )


def make_pipeline_llm() -> Callable[[list[dict[str, str]]], str]:
    """A MockLLM handler that dispatches on the system prompt of each call."""
    from rfp_copilot.graph.nodes import REVIEWER_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT
    from rfp_copilot.ingestion.requirements import EXTRACTION_SYSTEM_PROMPT

    reviewer = FakeReviewer()

    def handler(messages: list[dict[str, str]]) -> str:
        system, user = messages[0]["content"], messages[-1]["content"]
        if system.startswith(EXTRACTION_SYSTEM_PROMPT):
            return fake_extract(user)
        if system.startswith(WRITER_SYSTEM_PROMPT):
            return fake_write(user)
        if system.startswith(REVIEWER_SYSTEM_PROMPT):
            return reviewer(user)
        raise AssertionError(f"Unexpected prompt: {system[:80]}")

    return handler
