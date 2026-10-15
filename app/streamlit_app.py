"""RFP Copilot UI: upload an RFP, wait, download the response.

Run the API first (`uvicorn rfp_copilot.api.app:app --port 8010`), then:
    streamlit run app/streamlit_app.py
Set RFP_API_URL if the API is not on http://localhost:8010.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("RFP_API_URL", "http://localhost:8010").rstrip("/")
ACTIVE = {"queued", "running"}
GAPS = {"not_covered", "partial"}  # answers with a [TO CONFIRM: ...] gap
STAGES = ["ingest", "extract_requirements", "retrieve", "write", "review", "human_review"]


# ---------------------------------------------------------------------------
# Helpers (no Streamlit calls: unit-testable)
# ---------------------------------------------------------------------------


def request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Call the backend and raise on HTTP errors."""
    response = httpx.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
    response.raise_for_status()
    return response


def sections_to_complete(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sections with at least one answer the knowledge base could not (fully) provide."""
    gaps = {r["requirement_id"] for r in payload.get("compliance", []) if r["status"] in GAPS}
    return [g for g in payload["groups"] if gaps & set(g["requirement_ids"])]


def count_status(payload: Mapping[str, Any], *statuses: str) -> int:
    """Number of requirements whose compliance status is one of `statuses`."""
    return sum(r["status"] in statuses for r in payload.get("compliance", []))


def collect_edits(
    run_id: str, payload: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, str]:
    """Sections the user changed in the review area (group id -> new text)."""
    edits = {}
    for group in payload["groups"]:
        text = state.get(f"edit-{run_id}-{group['id']}")
        if text is not None and text.strip() != payload["drafts"].get(group["id"], "").strip():
            edits[group["id"]] = text
    return edits


def fetch_document(run_id: str, status: str, edits: dict[str, str]) -> bytes:
    """Approve the run if needed (with the user's edits), then return the DOCX bytes."""
    if status == "awaiting_review":
        request(
            "POST",
            f"/runs/{run_id}/review",
            params={"wait": "true"},
            json={"action": "approve", "edits": edits},
        )
    return request("GET", f"/runs/{run_id}/download").content


def progress_fraction(stage: str | None) -> float:
    """Share of the pipeline done, from the last finished stage."""
    done = STAGES.index(stage) + 1 if stage in STAGES else 0
    return min(0.95, 0.05 + done / len(STAGES))


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


def api(method: str, path: str, **kwargs: Any) -> httpx.Response:
    """`request` with one clear message on failure."""
    try:
        return request(method, path, **kwargs)
    except httpx.HTTPError:
        st.error("The service is not available right now. Please try again in a moment.")
        st.stop()


def start_screen() -> None:
    st.markdown("<h1 style='text-align:center'>RFP Copilot</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#6b7280'>Upload an RFP. "
        "Get a cited draft response in Word.</p>",
        unsafe_allow_html=True,
    )
    pdf = st.file_uploader("RFP (PDF)", type=["pdf"], label_visibility="collapsed")
    if st.button(
        "Generate response", type="primary", use_container_width=True, disabled=pdf is None
    ):
        files = {"file": (pdf.name, pdf.getvalue(), "application/pdf")}
        st.session_state["run_id"] = api("POST", "/runs", files=files).json()["id"]
        st.rerun()


@st.fragment(run_every=3)
def waiting_screen(run_id: str) -> None:
    run = api("GET", f"/runs/{run_id}").json()
    if run["status"] not in ACTIVE:
        st.rerun()
    st.markdown(
        "<h3 style='text-align:center'>Generating your response…</h3>", unsafe_allow_html=True
    )
    st.progress(progress_fraction(run.get("stage")))
    st.caption("This usually takes a few minutes.")


def ready_screen(run: dict[str, Any]) -> None:
    run_id = run["id"]
    payload = (
        api("GET", f"/runs/{run_id}/review").json() if run["status"] == "awaiting_review" else None
    )
    st.markdown(
        "<h3 style='text-align:center'>✅ Your response is ready</h3>", unsafe_allow_html=True
    )

    edits = collect_edits(run_id, payload, st.session_state) if payload else {}
    st.download_button(
        "Download Word document",
        data=lambda: fetch_document(run_id, run["status"], edits),
        file_name=f"{run['rfp_filename'].rsplit('.', 1)[0]}-response.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
        use_container_width=True,
    )

    if payload:
        for row in payload.get("compliance", []):
            if row["status"] == "non_compliant":
                st.warning(f"**{row['requirement_id']}**: {row['note']}", icon="⚠️")
        gaps = count_status(payload, *GAPS)
        checks = count_status(payload, "to_confirm")
        notes = []
        if gaps:
            notes.append(f"{gaps} answer(s) marked [TO CONFIRM] (missing from your knowledge base)")
        if checks:
            notes.append(f"{checks} item(s) in the bid team checklist")
        if notes:
            st.caption(" · ".join(notes) + ".")
        pending = sections_to_complete(payload)
        if pending:
            with st.expander("Complete before downloading"):
                requirements = {r["id"]: r["text"] for r in payload["requirements"]}
                for group in pending:
                    st.markdown(f"**{group['title']}**")
                    for rid in group["requirement_ids"]:
                        st.caption(f"{rid}: {requirements.get(rid, '')}")
                    draft = payload["drafts"][group["id"]]
                    st.text_area(
                        group["title"],
                        draft,
                        key=f"edit-{run_id}-{group['id']}",
                        height=min(400, 100 + 24 * draft.count("\n")),
                        label_visibility="collapsed",
                    )


def new_rfp_button() -> None:
    st.write("")
    if st.button("New RFP", use_container_width=True):
        st.session_state.pop("run_id", None)
        st.rerun()


def history() -> None:
    """Previous RFPs, kept out of the way in the collapsed sidebar."""
    runs = api("GET", "/runs").json()
    with st.sidebar:
        st.caption("Previous RFPs")
        for run in runs:
            name = run["rfp_filename"].rsplit(".", 1)[0].replace("_", " ")
            if st.button(name[:40], key=f"open-{run['id']}", use_container_width=True):
                st.session_state["run_id"] = run["id"]
                st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="RFP Copilot",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="collapsed",
        menu_items={},
    )
    st.markdown(
        """
        <style>
          #MainMenu, footer {visibility: hidden;}
          [data-testid="stToolbar"] {display: none;}
          .block-container {padding-top: 12vh; max-width: 640px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    history()
    run_id = st.session_state.get("run_id")
    if not run_id:
        start_screen()
        return
    run = api("GET", f"/runs/{run_id}").json()
    if run["status"] in ACTIVE:
        waiting_screen(run_id)
    elif run["status"] in {"awaiting_review", "completed"}:
        ready_screen(run)
        new_rfp_button()
    else:
        st.error("This RFP could not be processed. Please try again or upload another file.")
        new_rfp_button()


if __name__ == "__main__":
    main()
