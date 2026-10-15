"""UI tests: the Streamlit app rendered headless with a fake API, plus its pure helpers."""

from __future__ import annotations

import importlib.util
from typing import Any

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT

APP = ROOT / "app" / "streamlit_app.py"
RUN = {
    "id": "r1",
    "rfp_filename": "ballwin_rfp.pdf",
    "status": "awaiting_review",
    "stage": "human_review",
}
PAYLOAD = {
    "requirements": [
        {"id": "REQ-001", "text": "Must conform to WCAG 2.1 AA."},
        {"id": "REQ-002", "text": "Must integrate with the recreation system."},
    ],
    "groups": [
        {"id": "G01", "title": "Accessibility", "requirement_ids": ["REQ-001"]},
        {"id": "G02", "title": "Integrations", "requirement_ids": ["REQ-002"]},
    ],
    "drafts": {
        "G01": "- **REQ-001**: We conform to WCAG 2.1 Level AA [accessibility#1].",
        "G02": "- **REQ-002**: [TO CONFIRM: experience with the recreation system]",
    },
    "compliance": [
        {"requirement_id": "REQ-001", "status": "addressed", "note": None},
        {"requirement_id": "REQ-002", "status": "not_covered", "note": "recreation system"},
        {
            "requirement_id": "REQ-003",
            "status": "non_compliant",
            "note": "NON-COMPLIANT: the RFP requires an RPO of 4 hours; our offer is 24 hours.",
        },
        {"requirement_id": "REQ-004", "status": "to_confirm", "note": "Bid team check: 3 copies"},
    ],
}


def load_module():
    spec = importlib.util.spec_from_file_location("streamlit_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() only runs under `streamlit run`
    return module


class FakeAPI:
    def __init__(self, run: dict[str, Any] | None) -> None:
        self.run = run
        self.calls: list[tuple[str, str, Any]] = []

    def __call__(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        path = url.split(":8010", 1)[1]
        self.calls.append((method, path, kwargs.get("json")))
        req = httpx.Request(method, url)
        if path == "/runs":
            body = [self.run] if self.run else []
            return httpx.Response(
                200 if method == "GET" else 202, json=body if method == "GET" else RUN, request=req
            )
        if path == "/runs/r1":
            return httpx.Response(200, json=self.run, request=req)
        if path == "/runs/r1/review":
            return httpx.Response(200, json=PAYLOAD if method == "GET" else self.run, request=req)
        if path == "/runs/r1/download":
            return httpx.Response(200, content=b"PK-docx", request=req)
        raise AssertionError(path)


def _app(monkeypatch: pytest.MonkeyPatch, fake: Any, run_id: str | None = None) -> AppTest:
    monkeypatch.setenv("RFP_API_URL", "http://localhost:8010")
    monkeypatch.setattr(httpx, "request", fake)
    at = AppTest.from_string(
        f"exec(open({str(APP)!r}, encoding='utf-8').read())", default_timeout=30
    )
    if run_id:
        at.session_state["run_id"] = run_id
    return at.run()


# -- helpers ---------------------------------------------------------------------------


def test_sections_to_complete_and_edits() -> None:
    ui = load_module()
    assert [g["id"] for g in ui.sections_to_complete(PAYLOAD)] == ["G02"]
    state = {"edit-r1-G01": PAYLOAD["drafts"]["G01"], "edit-r1-G02": "- **REQ-002**: Done."}
    assert ui.collect_edits("r1", PAYLOAD, state) == {"G02": "- **REQ-002**: Done."}


def test_fetch_document_approves_once_then_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    ui = load_module()
    fake = FakeAPI(dict(RUN))
    monkeypatch.setattr(httpx, "request", fake)
    assert ui.fetch_document("r1", "awaiting_review", {"G02": "x"}) == b"PK-docx"
    assert fake.calls == [
        ("POST", "/runs/r1/review", {"action": "approve", "edits": {"G02": "x"}}),
        ("GET", "/runs/r1/download", None),
    ]
    fake.calls.clear()
    ui.fetch_document("r1", "completed", {})
    assert fake.calls == [("GET", "/runs/r1/download", None)]  # no second approval


def test_progress_fraction_is_monotonic() -> None:
    ui = load_module()
    values = [ui.progress_fraction(s) for s in [None, *ui.STAGES]]
    assert values == sorted(values) and values[0] > 0 and values[-1] <= 0.95


# -- screens ---------------------------------------------------------------------------


def test_start_screen_is_just_an_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(monkeypatch, FakeAPI(None))
    assert not at.exception
    main_buttons = [b for b in at.main.button]
    assert [b.label for b in main_buttons] == ["Generate response"]
    assert main_buttons[0].disabled  # until a PDF is chosen
    assert not at.metric and not at.expander and not at.dataframe


def test_waiting_screen_has_no_details(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(monkeypatch, FakeAPI({**RUN, "status": "running", "stage": "retrieve"}), run_id="r1")
    assert not at.exception
    assert len(at.main.button) == 0 and not at.metric
    texts = " ".join(m.value for m in at.markdown)
    assert "Generating your response" in texts and "retrieve" not in texts


def test_ready_screen_download_first_and_optional_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = _app(monkeypatch, FakeAPI(dict(RUN)), run_id="r1")
    assert not at.exception
    assert any("Your response is ready" in m.value for m in at.markdown)
    # Non-compliance is shown as a visible warning; gaps and checklist items are counted.
    assert len(at.warning) == 1 and "RPO of 4 hours" in at.warning[0].value
    assert "1 answer(s) marked [TO CONFIRM]" in at.caption[0].value
    assert "1 item(s) in the bid team checklist" in at.caption[0].value
    # Only the section needing input is offered for completion, collapsed by default.
    assert [e.label for e in at.expander] == ["Complete before downloading"]
    assert [t.key for t in at.text_area] == ["edit-r1-G02"]
    assert [b.label for b in at.main.button] == ["New RFP"]


def test_failed_run_shows_one_message(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(monkeypatch, FakeAPI({**RUN, "status": "failed", "stage": "ingest"}), run_id="r1")
    assert not at.exception
    assert len(at.error) == 1 and [b.label for b in at.main.button] == ["New RFP"]


def test_unreachable_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(method: str, url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("refused")

    at = _app(monkeypatch, down)
    assert not at.exception
    assert len(at.error) == 1 and "not available" in at.error[0].value
