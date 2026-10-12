"""Tests for deterministic routing, individual nodes, and the full graph with interrupt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from docx import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from rfp_copilot.config import Settings
from rfp_copilot.graph.graph import build_graph
from rfp_copilot.graph.nodes import (
    PipelineDeps,
    apply_deterministic_checks,
    make_review_node,
    make_write_node,
    route_after_human,
    route_after_review,
)
from rfp_copilot.llm import MockLLM
from rfp_copilot.models import Requirement, ReviewResult
from rfp_copilot.retrieval.retriever import HybridRetriever

REQS = [
    Requirement(id="REQ-001", text="Must hold ISO 27001.", type="mandatory", source_page=1),
    Requirement(id="REQ-002", text="Should offer training.", type="desirable", source_page=2),
]


def _review(score: float) -> dict:
    return ReviewResult(coverage=score, groundedness=score).model_dump()


# -- routing ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "iteration", "expected"),
    [
        ({"G01": 0.9, "G02": 0.8}, 1, "human_review"),  # all pass
        ({"G01": 0.9, "G02": 0.5}, 1, "write"),  # one fails, passes left
        ({"G01": 0.9, "G02": 0.5}, 2, "human_review"),  # one fails, no passes left
        ({}, 1, "human_review"),  # nothing to review
        ({"G01": 0.75}, 1, "human_review"),  # threshold is inclusive
    ],
)
def test_route_after_review(scores: dict, iteration: int, expected: str) -> None:
    state = {"reviews": {g: _review(s) for g, s in scores.items()}, "iteration": iteration}
    assert route_after_review(state, threshold=0.75, max_iterations=2) == expected


def test_route_after_human() -> None:
    assert route_after_human({"outcome": "approved"}) == "export_docx"
    assert route_after_human({"outcome": "rejected"}) == "__end__"


def test_deterministic_checks_tighten_review() -> None:
    llm_review = ReviewResult(coverage=1.0, groundedness=1.0)
    draft = "**REQ-001** We are certified [company-profile#3] and [made-up#9]."
    checked = apply_deterministic_checks(llm_review, draft, REQS, {"company-profile#3"})
    assert checked.uncovered_requirements == ["REQ-002"]
    assert checked.coverage == 0.5
    assert checked.groundedness == 0.5
    assert any("made-up#9" in c for c in checked.unsupported_claims)


def test_multi_citations_and_heading_cleanup() -> None:
    from rfp_copilot.graph.nodes import citations, clean_draft

    text = "A [x#1, y-2#3]; B [x#1]; C [z#4;w#5] and [not a citation]."
    assert citations(text) == ["x#1", "y-2#3", "z#4", "w#5"]
    draft = "## 5. Security\n\n**REQ-001** ok\n## Sub\ntext"
    assert clean_draft(draft) == "**REQ-001** ok\n## Sub\ntext"


def test_draft_without_citations_is_not_grounded() -> None:
    checked = apply_deterministic_checks(
        ReviewResult(coverage=1, groundedness=1), "REQ-001 REQ-002 all good.", REQS, set()
    )
    assert checked.groundedness == 0.0


def test_administrative_only_sections_need_no_citation() -> None:
    admin = [
        Requirement(id="REQ-009", text="Submit one PDF.", type="administrative", source_page=4)
    ]
    review = ReviewResult(coverage=1, groundedness=0.9)
    checked = apply_deterministic_checks(review, "**REQ-009**: we comply.", admin, set())
    assert checked.groundedness == 0.9


# -- nodes -----------------------------------------------------------------------------


def _state() -> dict:
    return {
        "requirements": [r.model_dump() for r in REQS],
        "groups": [{"id": "G01", "title": "Security", "requirement_ids": ["REQ-001", "REQ-002"]}],
        "passages": {
            "G01": [
                {
                    "chunk_id": "kb#1",
                    "source": "kb.md",
                    "title": "KB",
                    "text": "ISO 27001.",
                    "score": 1.0,
                }
            ]
        },
    }


def test_write_node_includes_feedback_on_revision(settings: Settings) -> None:
    writer = MockLLM(["first draft", "second draft"])
    deps = PipelineDeps(settings=settings, writer=writer, reviewer=None)
    write = make_write_node(deps)
    state = {**_state(), "iteration": 0}
    update = write(state)
    assert update["drafts"] == {"G01": "first draft"} and update["iteration"] == 1
    state |= update
    state["reviews"] = {"G01": {**_review(0.3), "feedback": "Cite the ISO certificate."}}
    update = write(state)
    assert update["drafts"]["G01"] == "second draft"
    revision_prompt = writer.calls[1][-1]["content"]
    assert "first draft" in revision_prompt and "Cite the ISO certificate." in revision_prompt


def test_review_node_applies_checks_and_survives_bad_json(settings: Settings) -> None:
    reviewer = MockLLM(lambda _m: "definitely not json")
    deps = PipelineDeps(settings=settings, writer=None, reviewer=reviewer)
    state = {**_state(), "drafts": {"G01": "REQ-001 [kb#1]"}, "pending_groups": ["G01"]}
    update = make_review_node(deps)(state)
    review = ReviewResult.model_validate(update["reviews"]["G01"])
    assert review.score == 0.0
    assert "REQ-002" in review.uncovered_requirements
    assert update["review_history"] == [{"G01": 0.0}]


# -- full graph ------------------------------------------------------------------------


def test_full_graph_loop_interrupt_and_export(
    settings: Settings, sample_pdf: Path, retriever: HybridRetriever, pipeline_llm: MockLLM
) -> None:
    deps = PipelineDeps(
        settings=settings, writer=pipeline_llm, reviewer=pipeline_llm, retriever=retriever
    )
    graph = build_graph(deps, InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    result = graph.invoke({"run_id": "t1", "rfp_path": str(sample_pdf)}, config)
    assert "__interrupt__" in result  # paused for human review
    state = graph.get_state(config)
    assert state.next == ("human_review",)
    values = state.values
    assert values["iteration"] == 2  # reviewer failed every section once -> one revision pass
    assert len(values["review_history"]) == 2
    assert all(score == 0.4 for score in values["review_history"][0].values())
    assert all(score == 0.9 for score in values["review_history"][1].values())
    assert len(values["requirements"]) >= 20

    first_group = values["groups"][0]["id"]
    edited = "Edited by a human: **REQ-001** handled [company-profile#1]."
    result = graph.invoke(
        Command(resume={"action": "approve", "edits": {first_group: edited}}), config
    )
    output = Path(result["output_path"])
    assert output.exists() and result["outcome"] == "approved"
    docx = Document(output)
    text = "\n".join(p.text for p in docx.paragraphs)
    assert "Edited by a human" in text
    assert "Compliance matrix" in text
    table = docx.tables[0]
    assert len(table.rows) == len(values["requirements"]) + 1
    assert table.rows[0].cells[0].text == "Requirement"


def test_full_graph_reject_skips_export(
    settings: Settings, sample_pdf: Path, retriever: HybridRetriever, pipeline_llm: MockLLM
) -> None:
    deps = PipelineDeps(
        settings=settings, writer=pipeline_llm, reviewer=pipeline_llm, retriever=retriever
    )
    graph = build_graph(deps, InMemorySaver())
    config = {"configurable": {"thread_id": "t2"}}
    graph.invoke({"run_id": "t2", "rfp_path": str(sample_pdf)}, config)
    result = graph.invoke(Command(resume={"action": "reject"}), config)
    assert result["outcome"] == "rejected"
    assert "output_path" not in result
    assert not (settings.runs_dir / "outputs").exists()
    json.dumps(result)  # state stays JSON-serialisable for the checkpointer
