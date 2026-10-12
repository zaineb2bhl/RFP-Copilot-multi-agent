"""Streamlit UI for RFP Copilot (talks to the FastAPI backend).

Run the API first (`uvicorn rfp_copilot.api.app:app --port 8010`), then:
    streamlit run app/streamlit_app.py
Set RFP_API_URL if the API is not on http://localhost:8010.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("RFP_API_URL", "http://localhost:8010").rstrip("/")
ACTIVE_STATUSES = {"queued", "running"}
STATUS_ICONS = {
    "queued": "⏳",
    "running": "⚙️",
    "awaiting_review": "📝",
    "completed": "✅",
    "rejected": "🚫",
    "failed": "❌",
}

st.set_page_config(page_title="RFP Copilot", page_icon="📄", layout="wide")


def api(method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Call the backend; show a readable error instead of a traceback."""
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=60, **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the API at {API_URL}: {exc}")
        st.stop()
    if response.status_code >= 400:
        st.error(f"API error {response.status_code}: {response.text}")
        st.stop()
    return response


def sidebar() -> str | None:
    """Upload form and run list; returns the selected run id."""
    st.sidebar.title("📄 RFP Copilot")
    with st.sidebar.form("upload", clear_on_submit=True):
        pdf = st.file_uploader("New RFP (PDF)", type=["pdf"])
        if st.form_submit_button("Start analysis", type="primary") and pdf is not None:
            run = api(
                "POST", "/runs", files={"file": (pdf.name, pdf.getvalue(), "application/pdf")}
            ).json()
            st.session_state["run_id"] = run["id"]
            st.toast(f"Run {run['id']} started")

    runs = api("GET", "/runs").json()
    if not runs:
        st.sidebar.info("No runs yet. Upload an RFP to start.")
        return None
    labels = {
        r["id"]: f"{STATUS_ICONS.get(r['status'], '')} {r['rfp_filename']} ({r['id']})"
        for r in runs
    }
    ids = list(labels)
    current = st.session_state.get("run_id")
    index = ids.index(current) if current in ids else 0
    selected = st.sidebar.radio("Runs", ids, index=index, format_func=labels.get)
    st.session_state["run_id"] = selected
    return selected


def show_progress(run: dict[str, Any]) -> None:
    st.info(
        f"{STATUS_ICONS[run['status']]} Status: **{run['status']}**, "
        f"current step: `{run.get('stage') or 'starting'}`"
    )
    st.caption("Refreshing every 5 seconds while the pipeline runs.")


def show_requirements(run_id: str) -> None:
    requirements = api("GET", f"/runs/{run_id}/requirements").json()
    if not requirements:
        st.write("No requirements extracted yet.")
        return
    columns = ["id", "type", "source_page", "status", "where_addressed", "sources", "text"]
    rows = [{c: ", ".join(r[c]) if c == "sources" else r[c] for c in columns} for r in requirements]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def show_review(run: dict[str, Any], payload: dict[str, Any]) -> None:
    """Drafts with reviewer scores; editable when the run awaits review."""
    editable = run["status"] == "awaiting_review"
    requirements = {r["id"]: r for r in payload["requirements"]}
    history = payload.get("review_history", [])
    st.caption(
        f"Write passes: {payload['iteration']}. Average score per pass: "
        + ", ".join(f"{sum(h.values()) / len(h):.2f}" for h in history if h)
    )
    edits: dict[str, str] = {}
    for group in payload["groups"]:
        review = payload["reviews"].get(group["id"], {})
        score = (review.get("coverage", 0) + review.get("groundedness", 0)) / 2 if review else None
        header = f"{group['title']}  ·  score {score:.2f}" if score is not None else group["title"]
        with st.expander(header, expanded=editable):
            for req_id in group["requirement_ids"]:
                req = requirements.get(req_id, {})
                st.markdown(
                    f"- **{req_id}** ({req.get('type')}, p.{req.get('source_page')}): "
                    f"{req.get('text')}"
                )
            if review:
                cols = st.columns(2)
                cols[0].metric("Coverage", f"{review['coverage']:.2f}")
                cols[1].metric("Groundedness", f"{review['groundedness']:.2f}")
                if review.get("feedback"):
                    st.markdown(f"**Reviewer feedback:** {review['feedback']}")
                for claim in review.get("unsupported_claims", []):
                    st.markdown(f"- ⚠️ {claim}")
            draft = payload["drafts"].get(group["id"], "")
            if editable:
                edits[group["id"]] = st.text_area(
                    "Draft", draft, height=300, key=f"draft-{run['id']}-{group['id']}"
                )
            else:
                st.markdown(draft)
            passages = payload["passages"].get(group["id"], [])
            if passages:
                st.caption(
                    "Retrieved sources: " + ", ".join(f"[{p['chunk_id']}]" for p in passages)
                )

    if editable:
        left, right = st.columns(2)
        if left.button("✅ Approve and export DOCX", type="primary", use_container_width=True):
            changed = {
                g: t for g, t in edits.items() if t.strip() != payload["drafts"].get(g, "").strip()
            }
            api("POST", f"/runs/{run['id']}/review", json={"action": "approve", "edits": changed})
            st.rerun()
        if right.button("🚫 Reject", use_container_width=True):
            api("POST", f"/runs/{run['id']}/review", json={"action": "reject"})
            st.rerun()


@st.fragment(run_every=5)
def watch_progress(run_id: str) -> None:
    """Poll while the pipeline runs; trigger a full rerun once it stops."""
    run = api("GET", f"/runs/{run_id}").json()
    if run["status"] in ACTIVE_STATUSES:
        show_progress(run)
    else:
        st.rerun()


def show_run(run_id: str) -> None:
    run = api("GET", f"/runs/{run_id}").json()
    st.subheader(f"{run['rfp_filename']}  ·  {run['id']}")
    if run["status"] in ACTIVE_STATUSES:
        watch_progress(run_id)
        return
    if run["status"] == "failed":
        st.error(f"Run failed: {run.get('error')}")
    usage = run.get("usage") or {}
    if usage:
        cols = st.columns(4)
        cols[0].metric("LLM calls", usage.get("calls", 0))
        cols[1].metric("Cached calls", usage.get("cached_calls", 0))
        cols[2].metric("Tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
        cols[3].metric("Cost (USD)", f"{usage.get('cost_usd', 0):.4f}")
    if run["status"] == "completed":
        docx = api("GET", f"/runs/{run_id}/download")
        st.download_button(
            "⬇️ Download DOCX response",
            docx.content,
            file_name=f"{run['rfp_filename'].rsplit('.', 1)[0]}-response.docx",
            type="primary",
        )
    if run["status"] == "awaiting_review":
        st.success("Drafts are ready. Review and edit them below, then approve to export.")
    tab_review, tab_requirements = st.tabs(["Drafts & review", "Requirements"])
    with tab_review:
        show_review(run, api("GET", f"/runs/{run_id}/review").json())
    with tab_requirements:
        show_requirements(run_id)


selected_run = sidebar()
if selected_run:
    show_run(selected_run)
else:
    st.title("RFP Copilot")
    st.write(
        "Upload a public RFP (PDF) in the sidebar to extract requirements "
        "and draft a cited response."
    )
