"""Tests for deterministic compliance, structured writer answers, and the review fixes."""

from __future__ import annotations

import json

import pytest

from rfp_copilot.compliance import build_compliance_rows, derive_status, requirement_segments
from rfp_copilot.evaluation.abstentions import possible_false_abstentions
from rfp_copilot.graph.nodes import apply_deterministic_checks, buyer_name_issues
from rfp_copilot.ingestion.pdf_parser import detect_issuer
from rfp_copilot.ingestion.requirements import extract_requirements
from rfp_copilot.llm import MockLLM, StructuredOutputError
from rfp_copilot.models import (
    NonCompliance,
    ParsedDocument,
    Requirement,
    RequirementAnswer,
    ResponseGroup,
    ReviewResult,
    Section,
    SectionAnswers,
    check_section_answers,
    placeholder,
)
from rfp_copilot.textsim import near_duplicates

MANDATORY = Requirement(id="REQ-001", text="Provide 99.9% uptime.", type="mandatory", source_page=1)
ADMIN = Requirement(
    id="REQ-002", text="Submit 3 printed copies.", type="administrative", source_page=9
)


# -- 1/2/3: deterministic status, cited-only sources, administrative checklist ------------


@pytest.mark.parametrize(
    ("segment", "expected_status", "expected_sources"),
    [
        ("- **REQ-001**: We commit to 99.9% uptime [kb#1].", "addressed", ["kb#1"]),
        ("- **REQ-001**: Uptime 99.9% [kb#1]. [TO CONFIRM: penalty terms]", "partial", ["kb#1"]),
        ("- **REQ-001**: [TO CONFIRM: uptime commitment]", "not_covered", []),
        ("- **REQ-001**: to be confirmed by the bid team.", "not_covered", []),  # legacy wording
        ("- **REQ-001**: We will meet the uptime target.", "to_confirm", []),  # uncited promise
        (None, "not_covered", []),
    ],
)
def test_status_is_derived_from_the_answer(
    segment: str | None, expected_status: str, expected_sources: list[str]
) -> None:
    status, sources, _note = derive_status(MANDATORY, segment, {"kb#1", "kb#2"}, None)
    assert (status, sources) == (expected_status, expected_sources)


def test_placeholder_is_never_addressed_or_partial() -> None:
    for text in (
        "[TO CONFIRM: x]",
        "**REQ-001**: [TO CONFIRM: x] see [kb#9]",
    ):  # kb#9 not retrieved
        status, sources, _ = derive_status(MANDATORY, text, {"kb#1"}, None)
        assert status == "not_covered" and sources == []


def test_sources_are_only_cited_chunks() -> None:
    draft = "- **REQ-001**: Uptime 99.9% [kb#1].\n- **REQ-003**: Other [kb#2]."
    rows = build_compliance_rows(
        [MANDATORY],
        [ResponseGroup(id="G1", title="Hosting", requirement_ids=["REQ-001"])],
        {"G1": draft},
        {},
        {"G1": {"kb#1", "kb#2", "kb#3"}},  # kb#2/kb#3 retrieved but not cited for REQ-001
    )
    assert rows[0].sources == ["kb#1"]


def test_administrative_items_are_to_confirm_with_checklist_line() -> None:
    status, sources, note = derive_status(ADMIN, "- **REQ-002**: Complied with.", set(), None)
    assert (status, sources) == ("to_confirm", [])
    assert note == "Bid team check: Submit 3 printed copies. (RFP p.9)"


def test_segments_split_on_requirement_bullets() -> None:
    draft = "- **REQ-001**: a [kb#1].\n  continued.\n- **REQ-002**: b\nREQ-007 mentioned inline."
    segments = requirement_segments(draft)
    assert segments["REQ-001"] == "- **REQ-001**: a [kb#1].\n  continued."
    assert segments["REQ-002"].startswith("- **REQ-002**: b")
    assert segments["REQ-007"] == "REQ-007 mentioned inline."


# -- 4/5: structured writer answers ----------------------------------------------------


def _answers(*items: dict) -> SectionAnswers:
    return SectionAnswers(answers=[RequirementAnswer(**i) for i in items])


def test_check_section_answers_rejects_ranges_gaps_and_duplicates() -> None:
    expected = ["REQ-001", "REQ-002"]
    check_section_answers(
        _answers(
            {"requirement_id": "REQ-001", "answer": "x [kb#1]."},
            {"requirement_id": "REQ-002", "missing_information": "y"},
        ),
        expected,
    )
    with pytest.raises(ValueError, match="missing answers for REQ-002"):
        check_section_answers(_answers({"requirement_id": "REQ-001", "answer": "x"}), expected)
    with pytest.raises(ValueError, match="range"):
        check_section_answers(
            _answers(
                {"requirement_id": "REQ-001", "answer": "REQ-001 through REQ-002: yes"},
                {"requirement_id": "REQ-002", "answer": "ok"},
            ),
            expected,
        )
    with pytest.raises(ValueError, match="several answers for REQ-001"):
        check_section_answers(
            _answers(
                {"requirement_id": "REQ-001", "answer": "a"},
                {"requirement_id": "REQ-001", "answer": "b"},
                {"requirement_id": "REQ-002", "answer": "c"},
            ),
            expected,
        )
    with pytest.raises(ValueError, match="neither an answer nor missing_information"):
        check_section_answers(
            _answers(
                {"requirement_id": "REQ-001", "answer": " "},
                {"requirement_id": "REQ-002", "answer": "c"},
            ),
            expected,
        )


def test_requirement_id_must_be_a_single_id() -> None:
    with pytest.raises(ValueError):
        RequirementAnswer(requirement_id="REQ-030-REQ-050", answer="x")


def test_placeholders_say_what_is_missing() -> None:
    assert placeholder("RecTrac integration experience.") == (
        "[TO CONFIRM: RecTrac integration experience]"
    )
    answer = RequirementAnswer(requirement_id="REQ-004", answer="", missing_information="RPO")
    assert answer.render() == "- **REQ-004**: [TO CONFIRM: RPO]"
    partial = RequirementAnswer(
        requirement_id="REQ-004",
        answer="Daily backups [kb#1].",
        missing_information="restore tests",
    )
    assert partial.render() == "- **REQ-004**: Daily backups [kb#1]. [TO CONFIRM: restore tests]"


def test_structured_check_failure_triggers_retry_with_message() -> None:
    bad = json.dumps({"answers": [{"requirement_id": "REQ-001", "answer": "a"}]})
    good = json.dumps(
        {
            "answers": [
                {"requirement_id": "REQ-001", "answer": "a"},
                {"requirement_id": "REQ-002", "answer": "b"},
            ]
        }
    )
    llm = MockLLM([bad, good])
    result = llm.complete_structured(
        [{"role": "user", "content": "go"}],
        SectionAnswers,
        check=lambda r: check_section_answers(r, ["REQ-001", "REQ-002"]),
    )
    assert len(result.answers) == 2
    assert "missing answers for REQ-002" in llm.calls[1][-1]["content"]
    with pytest.raises(StructuredOutputError):
        MockLLM([bad, bad, bad]).complete_structured(
            [{"role": "user", "content": "go"}],
            SectionAnswers,
            check=lambda r: check_section_answers(r, ["REQ-001", "REQ-002"]),
        )


# -- 6: non-compliance --------------------------------------------------------------


def test_reviewer_non_compliance_is_kept_for_the_section_only() -> None:
    review = ReviewResult(
        coverage=1,
        groundedness=1,
        non_compliant=[
            NonCompliance(requirement_id="REQ-001", required="RPO 4 h", offered="RPO 24 h"),
            NonCompliance(requirement_id="REQ-099", required="x", offered="y"),
        ],
    )
    checked = apply_deterministic_checks(
        review, "- **REQ-001**: RPO of 24 hours [kb#1].", [MANDATORY], {"kb#1": "RPO of 24 hours"}
    )
    assert [n.requirement_id for n in checked.non_compliant] == ["REQ-001"]
    status, sources, note = derive_status(
        MANDATORY, "- **REQ-001**: RPO of 24 hours [kb#1].", {"kb#1"}, checked.non_compliant[0]
    )
    assert status == "non_compliant" and sources == ["kb#1"]
    assert note == "NON-COMPLIANT: the RFP requires RPO 4 h; our documented offer is RPO 24 h."


# -- 7: deduplication and buyer name --------------------------------------------------


def test_near_duplicates() -> None:
    body = "The vendor must provide training for administrators and content editors."
    assert near_duplicates("Training for administrators and content editors.", body, threshold=0.7)
    assert near_duplicates("The vendor must provide training for administrators", body)
    assert not near_duplicates("The vendor must host the website.", body)


def test_extraction_merges_summary_and_body_duplicates() -> None:
    document = ParsedDocument(
        source="x.pdf",
        title="X",
        page_texts=["summary", "body"],
        sections=[
            Section(id="S01", title="Summary", text="[page 1]\nsummary", page_start=1, page_end=1),
            Section(id="S02", title="Body", text="[page 2]\nbody", page_start=2, page_end=2),
        ],
    )
    summary = "The vendor must provide training for administrators and content editors"
    body = summary + " before go-live, with written documentation."
    items = [
        {"text": summary, "type": "mandatory", "source_page": 1},
        {"text": body, "type": "mandatory", "source_page": 2},
        {"text": "The vendor must host the website.", "type": "mandatory", "source_page": 2},
    ]
    llm = MockLLM(lambda _m: json.dumps({"requirements": items}))
    requirements = extract_requirements(document, llm, batch_chars=100_000)
    assert [r.id for r in requirements] == ["REQ-001", "REQ-002"]
    assert requirements[0].text == body  # the more complete wording is kept


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Request for Proposals\nCity of Ballwin Website Redesign\n", "City of Ballwin"),
        ("CITY OF HIDDEN HILLS\nREQUEST FOR PROPOSALS", "City of Hidden Hills"),
        ("Cedar County invites proposals for its website.", "County of Cedar"),
        ("Indiana Public Retirement System - RFP 23-03", None),
    ],
)
def test_detect_issuer(text: str, expected: str | None) -> None:
    assert detect_issuer([text]) == expected


def test_buyer_name_mismatch_is_flagged_but_past_clients_are_not() -> None:
    passages = {"proj#1": "The Town of Port Alder had a limited budget."}
    wrong = "- **REQ-001**: We will train the Town staff."
    assert buyer_name_issues(wrong, "City of Ballwin", passages)
    reference = "- **REQ-001**: As for the Town of Port Alder, staff were trained [proj#1]."
    assert buyer_name_issues(reference, "City of Ballwin", passages) == []
    assert buyer_name_issues("the City will host", "City of Ballwin", passages) == []
    assert buyer_name_issues(wrong, None, passages) == []


# -- 8: possible false abstentions ----------------------------------------------------


def test_possible_false_abstentions() -> None:
    other = Requirement(id="REQ-003", text="Provide SMS alerts.", type="mandatory", source_page=2)
    values = {
        "requirements": [MANDATORY.model_dump(), ADMIN.model_dump(), other.model_dump()],
        "groups": [
            {"id": "G1", "title": "T", "requirement_ids": ["REQ-001", "REQ-002", "REQ-003"]}
        ],
        "drafts": {
            "G1": "- **REQ-001**: [TO CONFIRM: uptime]\n"
            "- **REQ-002**: [TO CONFIRM: bid team check: copies]\n"
            "- **REQ-003**: [TO CONFIRM: SMS]"
        },
        "passages": {
            "G1": [
                {
                    "chunk_id": "kb#1",
                    "source": "kb.md",
                    "title": "T",
                    "text": "99.9% uptime",
                    "score": 1,
                },
                {"chunk_id": "kb#2", "source": "kb.md", "title": "T", "text": "other", "score": 1},
            ]
        },
        "evidence": {
            "REQ-001": [{"chunk_id": "kb#1", "similarity": 0.81}],
            "REQ-003": [{"chunk_id": "kb#2", "similarity": 0.62}],
        },
    }
    findings = possible_false_abstentions(values, min_similarity=0.7)
    assert [f["requirement_id"] for f in findings] == ["REQ-001"]  # admin and weak ones excluded
    assert findings[0]["best_chunk"] == "kb#1" and findings[0]["best_similarity"] == 0.81
    assert findings[0]["missing_according_to_draft"] == "uptime"
