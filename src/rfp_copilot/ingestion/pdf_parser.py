"""Parse an RFP PDF into page-aware sections with PyMuPDF.

Headings are detected from typography (font size larger than the body text, or a short
bold line), which works for most text-based RFPs without any ML model. Each section's text
contains `[page N]` markers so downstream LLM calls can cite pages.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from rfp_copilot.models import ParsedDocument, Section

PAGE_MARKER_RE = re.compile(r"^\[page (\d+)\]$", re.MULTILINE)
_PAGE_FOOTER_RE = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
_BOLD_FLAG = 16
_MAX_HEADING_CHARS = 120
# Expand ligatures ("ﬁ" -> "fi") so requirement text can be matched and searched.
_TEXT_FLAGS = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES
_DICT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES
_NUMBERED_RE = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z]\.|section|annex|appendix|part)\s", re.I)


@dataclass
class _Line:
    page: int
    text: str
    size: float
    bold: bool


def page_marker(page: int) -> str:
    """The marker inserted in section text when a new page starts."""
    return f"[page {page}]"


def _read_lines(doc: pymupdf.Document) -> tuple[list[_Line], list[str]]:
    lines: list[_Line] = []
    page_texts: list[str] = []
    for page_number, page in enumerate(doc, start=1):
        page_texts.append(page.get_text("text", flags=_TEXT_FLAGS))
        for block in page.get_text("dict", flags=_DICT_FLAGS)["blocks"]:
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans)
                if _PAGE_FOOTER_RE.match(text):
                    continue
                bold = all(s["flags"] & _BOLD_FLAG or "bold" in s["font"].lower() for s in spans)
                lines.append(_Line(page_number, text, max(s["size"] for s in spans), bold))
    return lines, page_texts


def _body_font_size(lines: list[_Line]) -> float:
    weights: Counter[float] = Counter()
    for line in lines:
        weights[round(line.size, 1)] += len(line.text)
    return weights.most_common(1)[0][0] if weights else 0.0


def _is_heading(line: _Line, body_size: float) -> bool:
    text = line.text
    if len(text) > _MAX_HEADING_CHARS or text.endswith((".", ",", ";")):
        return False
    if not any(c.isalpha() for c in text):
        return False
    return line.size >= body_size + 1.0 or (line.bold and len(text) <= 80)


def parse_pdf(path: Path | str) -> ParsedDocument:
    """Parse the PDF at `path` into sections. Raises ValueError if it has no text layer."""
    path = Path(path)
    with pymupdf.open(path) as doc:
        lines, page_texts = _read_lines(doc)
        meta_title = (doc.metadata or {}).get("title") or ""
    if not lines:
        raise ValueError(f"{path.name} has no extractable text (scanned PDF? OCR is not supported)")

    body_size = _body_font_size(lines)
    sections: list[Section] = []
    title, body, pages = "Preamble", [], []
    last_page = 0
    previous_heading: _Line | None = None

    def flush() -> None:
        if any(not PAGE_MARKER_RE.match(b) for b in body):
            sections.append(
                Section(
                    id=f"S{len(sections) + 1:02d}",
                    title=title,
                    text="\n".join(body),
                    page_start=min(pages),
                    page_end=max(pages),
                )
            )

    for line in lines:
        if _is_heading(line, body_size):
            wrapped = (
                previous_heading is not None
                and previous_heading.page == line.page
                and abs(previous_heading.size - line.size) < 0.5
                and not _NUMBERED_RE.match(line.text)
                and not any(not PAGE_MARKER_RE.match(b) for b in body)
            )
            if wrapped:
                title = f"{title} {line.text}"  # heading wrapped over two lines
            else:
                flush()
                title, body, pages = line.text, [], []
                last_page = 0
            previous_heading = line
            pages.append(line.page)
            continue
        previous_heading = None
        if line.page != last_page:
            body.append(page_marker(line.page))
            last_page = line.page
        body.append(line.text)
        pages.append(line.page)
    flush()

    doc_title = meta_title.strip() or next(
        (s.title for s in sections if s.title != "Preamble"), path.stem
    )
    return ParsedDocument(
        source=path.name,
        title=doc_title,
        page_texts=page_texts,
        sections=sections,
        issuer=detect_issuer(page_texts[:2]),
    )


_ENTITY = r"City|Town|Village|County|Township|Borough|Parish"
_TITLE_CASE_RE = re.compile(
    rf"\b({_ENTITY}) of ([A-Z][a-z][\w'.-]*(?:[ \t][A-Z][a-z][\w'.-]*){{0,2}})"
)
_UPPER_CASE_RE = re.compile(
    rf"\b({_ENTITY.upper()}) OF ([A-Z][A-Z'.-]+(?:[ \t][A-Z][A-Z'.-]+){{0,3}})"
)
_NAME_FIRST_RE = re.compile(r"\b([A-Z][a-z]+(?:[ \t][A-Z][a-z]+)?)[ \t](County|Township|Parish)\b")
_STOP_WORDS = {
    "REQUEST",
    "REQUESTS",
    "FOR",
    "PROPOSAL",
    "PROPOSALS",
    "RFP",
    "RFQ",
    "BID",
    "NOTICE",
    "INVITATION",
    "WEBSITE",
    "WEB",
    "SITE",
    "SERVICES",
    "REDESIGN",
    "AND",
    "THE",
}


def detect_issuer(page_texts: list[str]) -> str | None:
    """The public body that issued the RFP, e.g. "City of Ballwin", from its first pages.

    Looks for "<City|Town|County...> of <Name>" (title or upper case) and "<Name> County";
    the earliest match wins. Returns None when nothing is found (e.g. a state agency), and
    the writer then refers to "the buyer".
    """
    text = "\n".join(page_texts)
    found: list[tuple[int, str]] = []
    if match := _TITLE_CASE_RE.search(text):
        words = match.group(2).split()
        while len(words) > 1 and words[-1].upper() in _STOP_WORDS:
            words.pop()
        found.append((match.start(), f"{match.group(1)} of {' '.join(words)}"))
    if match := _UPPER_CASE_RE.search(text):
        words = match.group(2).split()
        while words and words[-1] in _STOP_WORDS:
            words.pop()
        if words:
            name = " ".join(w.capitalize() for w in words)
            found.append((match.start(), f"{match.group(1).capitalize()} of {name}"))
    if match := _NAME_FIRST_RE.search(text):
        found.append((match.start(), f"{match.group(2)} of {match.group(1)}"))
    return min(found)[1] if found else None


def split_with_page_markers(text: str, max_chars: int) -> list[str]:
    """Split marker-annotated text into pieces of at most ~`max_chars`.

    Splits on line boundaries and repeats the current `[page N]` marker at the start of each
    piece, so every piece stays citable on its own.
    """
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    page: str | None = None
    for line in text.splitlines():
        if PAGE_MARKER_RE.match(line):
            page = line
        if current and size + len(line) > max_chars:
            pieces.append("\n".join(current))
            current, size = ([page] if page and line != page else []), 0
        current.append(line)
        size += len(line) + 1
    if current and any(not PAGE_MARKER_RE.match(c) for c in current):
        pieces.append("\n".join(current))
    return pieces
