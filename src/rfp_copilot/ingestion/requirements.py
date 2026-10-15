"""Structured requirement extraction from a parsed RFP, and grouping into response sections."""

from __future__ import annotations

import logging
import re

from rfp_copilot.ingestion.pdf_parser import split_with_page_markers
from rfp_copilot.llm import BaseLLMClient, StructuredOutputError
from rfp_copilot.models import (
    ExtractedRequirement,
    ExtractedRequirements,
    ParsedDocument,
    Requirement,
    ResponseGroup,
    Section,
)
from rfp_copilot.textsim import near_duplicates

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an experienced bid manager analysing a Request for Proposal.
Extract EVERY requirement a bidder must or should satisfy from the RFP excerpt.

Rules:
- One requirement per item. Keep the original wording; you may shorten long sentences.
- type = "mandatory" for must / shall / required / will be rejected if not;
  "desirable" for should / may / preferred / advantage / nice to have;
  "administrative" for submission format, deadlines, forms, signatures, eligibility documents.
- source_page = the number N of the closest "[page N]" marker ABOVE the requirement.
- Do not invent requirements. Ignore background information and questions to the buyer.
- If the excerpt contains no requirement, return {"requirements": []}."""

_MATCH_PREFIX_CHARS = 60
_SECTION_NUMBER_RE = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z]\.)\s+")


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _extraction_batches(document: ParsedDocument, max_chars: int) -> list[list[Section]]:
    """Group consecutive sections into batches of at most `max_chars` characters."""
    batches: list[list[Section]] = []
    current: list[Section] = []
    size = 0
    for section in document.sections:
        for piece in split_with_page_markers(section.text, max_chars):
            part = section if piece == section.text else section.model_copy(update={"text": piece})
            if current and size + len(piece) > max_chars:
                batches.append(current)
                current, size = [], 0
            current.append(part)
            size += len(piece) + len(section.title) + 4
    if current:
        batches.append(current)
    return batches


def _batch_prompt(batch: list[Section]) -> str:
    parts = [f"## {section.title}\n{section.text}" for section in batch]
    return "RFP excerpt:\n\n" + "\n\n".join(parts)


def _locate(
    item: ExtractedRequirement, batch: list[Section], document: ParsedDocument
) -> tuple[int, str | None]:
    """Return (page, section_id) for an extracted requirement, verifying the model's page.

    The page is taken from the PDF text when the requirement's opening words are found there;
    otherwise the model's page is kept if it lies within the batch, else the batch start.
    """
    needle = _normalise(item.text)[:_MATCH_PREFIX_CHARS]
    section_id = next((s.id for s in batch if needle and needle in _normalise(s.text)), None)
    pages = range(min(s.page_start for s in batch), max(s.page_end for s in batch) + 1)
    page = next(
        (p for p in pages if needle and needle in _normalise(document.page_texts[p - 1])),
        item.source_page if item.source_page in pages else pages.start,
    )
    if section_id is None:
        section_id = next((s.id for s in batch if s.page_start <= page <= s.page_end), batch[0].id)
    return page, section_id


def extract_requirements(
    document: ParsedDocument, llm: BaseLLMClient, *, batch_chars: int = 6000
) -> list[Requirement]:
    """Extract requirements from every section of `document`, with ids REQ-001, REQ-002...

    The document is processed in batches small enough for a 4k-token local model. Duplicates
    are merged, including near-duplicates such as the same requirement listed in a summary
    table and in the main body (the more complete wording is kept). A batch whose output
    never validates is skipped and logged rather than failing the whole run.
    """
    requirements: list[Requirement] = []
    seen: set[str] = set()
    for batch in _extraction_batches(document, batch_chars):
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": _batch_prompt(batch)},
        ]
        try:
            result = llm.complete_structured(
                messages, ExtractedRequirements, temperature=0.0, tag="extract_requirements"
            )
        except StructuredOutputError as exc:
            logger.error("Skipping sections %s: %s", [s.id for s in batch], exc)
            continue
        for item in result.requirements:
            key = _normalise(item.text)
            if not key or key in seen:
                continue
            seen.add(key)
            page, section_id = _locate(item, batch, document)
            candidate = Requirement(
                id=f"REQ-{len(requirements) + 1:03d}",
                text=item.text.strip(),
                type=item.type,
                source_page=page,
                section_id=section_id,
            )
            duplicate = next(
                (i for i, r in enumerate(requirements) if near_duplicates(r.text, candidate.text)),
                None,
            )
            if duplicate is None:
                requirements.append(candidate)
            elif len(candidate.text) > len(requirements[duplicate].text):
                # Keep the more complete wording (usually the main body, not a summary table)
                # with its page and section, under the id already assigned.
                requirements[duplicate] = candidate.model_copy(
                    update={"id": requirements[duplicate].id}
                )
    return requirements


def group_requirements(
    requirements: list[Requirement], document: ParsedDocument, *, max_per_group: int = 6
) -> list[ResponseGroup]:
    """Group requirements by RFP section (in document order) into response sections.

    Sections with more than `max_per_group` requirements are split into parts, so each write
    call stays small enough for a local model.
    """
    titles = {s.id: s.title for s in document.sections}
    order = {s.id: i for i, s in enumerate(document.sections)}
    by_section: dict[str, list[str]] = {}
    for req in requirements:
        by_section.setdefault(req.section_id or "S00", []).append(req.id)

    groups: list[ResponseGroup] = []
    for section_id in sorted(by_section, key=lambda sid: order.get(sid, -1)):
        ids = by_section[section_id]
        parts = [ids[i : i + max_per_group] for i in range(0, len(ids), max_per_group)]
        for part_index, part in enumerate(parts, start=1):
            # Drop the RFP's own numbering ("5. Security") - the response has its own.
            title = _SECTION_NUMBER_RE.sub("", titles.get(section_id, "General requirements"))
            if len(parts) > 1:
                title = f"{title} (part {part_index})"
            groups.append(
                ResponseGroup(id=f"G{len(groups) + 1:02d}", title=title, requirement_ids=part)
            )
    return groups
