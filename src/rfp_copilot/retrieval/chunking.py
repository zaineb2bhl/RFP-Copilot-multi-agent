"""Load the markdown/text knowledge base and split it into citable chunks."""

from __future__ import annotations

import re
from pathlib import Path

from rfp_copilot.models import Chunk

KB_SUFFIXES = {".md", ".markdown", ".txt"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split a paragraph longer than `max_chars` on sentence, then word, boundaries."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > max_chars:  # a single huge "sentence": cut at a space
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, overlap: int) -> str:
    """The last ~`overlap` characters of `text`, starting at a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def _sections(text: str) -> list[tuple[str, str]]:
    """Return (heading path, body) pairs; the heading path is e.g. "Doc > Approach"."""
    doc_title = ""
    heading = ""
    body: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            path = f"{doc_title} > {heading}" if doc_title and heading else doc_title or heading
            sections.append((path, content))

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            body = []
            level, title = len(match.group(1)), match.group(2).strip()
            if level == 1 and not doc_title:
                doc_title, heading = title, ""
            else:
                heading = title
            continue
        body.append(line)
    flush()
    return sections


def chunk_markdown(
    text: str, source: str, *, chunk_size: int = 800, overlap: int = 120
) -> list[Chunk]:
    """Split a markdown document into chunks of at most ~`chunk_size` characters.

    Chunks never cross a heading; paragraphs are packed together, and each new chunk in a
    section starts with the last ~`overlap` characters of the previous one. Chunk ids are
    `<file stem>#<n>` and are used verbatim as citation keys.
    """
    stem = Path(source).stem
    chunks: list[Chunk] = []
    for heading_path, body in _sections(text):
        title = heading_path or stem
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        units = [u for p in paragraphs for u in _split_long(" ".join(p.split()), chunk_size)]
        current = ""
        for unit in units:
            if current and len(current) + len(unit) + 2 > chunk_size:
                chunks.append(
                    Chunk(id=f"{stem}#{len(chunks) + 1}", source=source, title=title, text=current)
                )
                current = f"{_tail(current, overlap)} {unit}".strip()
            else:
                current = f"{current}\n\n{unit}".strip()
        if current:
            chunks.append(
                Chunk(id=f"{stem}#{len(chunks) + 1}", source=source, title=title, text=current)
            )
    return chunks


def load_knowledge_base(
    kb_dir: Path | str, *, chunk_size: int = 800, overlap: int = 120
) -> list[Chunk]:
    """Chunk every markdown/text file under `kb_dir` (recursively, in sorted order)."""
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")
    chunks: list[Chunk] = []
    for path in sorted(p for p in kb_dir.rglob("*") if p.suffix.lower() in KB_SUFFIXES):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, path.name, chunk_size=chunk_size, overlap=overlap))
    return chunks
