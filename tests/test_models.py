"""Tests for the shared Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rfp_copilot.models import ExtractedRequirement, ExtractedRequirements, ReviewResult


def test_requirement_type_is_normalised() -> None:
    req = ExtractedRequirement(text="The supplier shall X.", type=" Mandatory ", source_page=2)
    assert req.type == "mandatory"


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "The supplier shall X.", "type": "optional", "source_page": 1},
        {"text": "The supplier shall X.", "type": "mandatory", "source_page": 0},
        {"text": "X", "type": "mandatory", "source_page": 1},
    ],
)
def test_invalid_requirements_are_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        ExtractedRequirement.model_validate(payload)


def test_extracted_requirements_parse_from_json() -> None:
    item = '{"text": "Must be hosted in the EU.", "type": "mandatory", "source_page": 3}'
    raw = f'{{"requirements": [{item}]}}'
    assert ExtractedRequirements.model_validate_json(raw).requirements[0].source_page == 3


def test_review_score_and_bounds() -> None:
    assert ReviewResult(coverage=0.5, groundedness=1.0).score == 0.75
    with pytest.raises(ValidationError):
        ReviewResult(coverage=1.5, groundedness=0.5)
