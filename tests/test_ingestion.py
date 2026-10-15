"""Tests for PDF parsing, requirement extraction and grouping."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from rfp_copilot.ingestion.pdf_parser import parse_pdf, split_with_page_markers
from rfp_copilot.ingestion.requirements import extract_requirements, group_requirements
from rfp_copilot.llm import MockLLM
from rfp_copilot.models import ParsedDocument, Requirement
from tests.fakes import fake_extract


def test_parse_sample_rfp_sections_and_pages(sample_document: ParsedDocument) -> None:
    titles = [s.title for s in sample_document.sections]
    assert "4. Content Management System" in titles
    assert "11. Submission Instructions" in titles
    assert sample_document.title == "RFP-2026-031 City Website Redesign, Hosting and Support"
    assert sample_document.page_count >= 2
    last = sample_document.sections[-1]
    assert last.page_end == sample_document.page_count
    assert all("Page 1 of" not in s.text for s in sample_document.sections)  # footers removed


def test_sections_carry_page_markers(sample_document: ParsedDocument) -> None:
    for section in sample_document.sections:
        assert section.text.startswith(f"[page {section.page_start}]")


def test_ligatures_are_expanded(sample_document: ParsedDocument) -> None:
    assert "ﬀ" not in "".join(sample_document.page_texts)  # "ff" ligature
    assert any("City staff" in s.text for s in sample_document.sections)


def test_scanned_pdf_without_text_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(path)
    with pytest.raises(ValueError, match="no extractable text"):
        parse_pdf(path)


def test_split_with_page_markers_repeats_marker() -> None:
    text = "[page 3]\n" + "\n".join(f"line {i} " + "x" * 40 for i in range(10))
    pieces = split_with_page_markers(text, 120)
    assert len(pieces) > 1
    assert all(p.startswith("[page 3]") for p in pieces)


def test_extract_requirements_with_mock(sample_document: ParsedDocument) -> None:
    llm = MockLLM(lambda messages: fake_extract(messages[-1]["content"]))
    requirements = extract_requirements(sample_document, llm, batch_chars=1500)
    assert len(requirements) >= 20
    assert [r.id for r in requirements[:2]] == ["REQ-001", "REQ-002"]
    uptime = next(r for r in requirements if "99.9%" in r.text)
    assert uptime.type == "mandatory"
    assert uptime.section_id == next(
        s.id for s in sample_document.sections if s.title.endswith("Hosting and Security")
    )
    assert len(llm.calls) > 1  # batched


def test_extraction_verifies_page_and_dedupes(sample_document: ParsedDocument) -> None:
    deadline = "The deadline for submission is 20 January 2027 at 2:00 p.m. local time."
    item = {"text": deadline, "type": "Administrative", "source_page": 1}  # wrong page
    llm = MockLLM(lambda _m: json.dumps({"requirements": [item, item]}))
    requirements = extract_requirements(sample_document, llm, batch_chars=100_000)
    assert len(requirements) == 1
    assert requirements[0].source_page == sample_document.page_count
    assert requirements[0].type == "administrative"


def test_invalid_batch_is_skipped_not_fatal(sample_document: ParsedDocument) -> None:
    llm = MockLLM(lambda _m: "not json at all")
    assert extract_requirements(sample_document, llm, batch_chars=100_000) == []


def test_group_requirements_splits_large_sections(sample_document: ParsedDocument) -> None:
    section = sample_document.sections[1]
    reqs = [
        Requirement(
            id=f"REQ-{i:03d}",
            text=f"Requirement {i}",
            type="mandatory",
            source_page=1,
            section_id=section.id,
        )
        for i in range(1, 8)
    ]
    groups = group_requirements(reqs, sample_document, max_per_group=3)
    assert [len(g.requirement_ids) for g in groups] == [3, 3, 1]
    assert groups[0].title.endswith("(part 1)")
    assert [g.id for g in groups] == ["G01", "G02", "G03"]
