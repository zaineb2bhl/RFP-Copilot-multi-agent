"""Resumable per-item result storage and markdown/JSON reports."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class ResultStore:
    """One JSON file per computed item, so an interrupted evaluation resumes where it stopped.

    Results are stored with the numbers measured when the item was first computed, so a re-run
    does not replace real latencies with near-zero cached ones.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path(self, *parts: str) -> Path:
        """File path for an item key such as ("rag", "chunk_400", "Q01")."""
        safe = [_UNSAFE.sub("_", p) for p in parts]
        return self.root.joinpath(*safe[:-1], f"{safe[-1]}.json")

    def get(self, *parts: str) -> dict[str, Any] | None:
        """Stored result, or None if not computed yet."""
        path = self.path(*parts)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, *parts: str, data: dict[str, Any]) -> None:
        """Store a result (atomically, so a crash never leaves a half-written file)."""
        path = self.path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Markdown table of `rows`; `columns` are (key, header) pairs."""
    header = "| " + " | ".join(h for _, h in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_cell(r.get(k)) for k, _ in columns) + " |" for r in rows]
    return "\n".join([header, rule, *body])


def write_report(
    results_dir: Path | str,
    name: str,
    *,
    title: str,
    sections: list[tuple[str, list[dict[str, Any]], list[tuple[str, str]]]],
    meta: dict[str, Any],
    notes: list[str] | None = None,
) -> tuple[Path, Path]:
    """Write `<name>.json` (all rows + metadata) and `<name>.md` (tables) in `results_dir`.

    `sections` are (heading, rows, columns) triples.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), **meta}
    json_path = results_dir / f"{name}.json"
    json_path.write_text(
        json.dumps(
            {"meta": meta, "sections": {heading: rows for heading, rows, _ in sections}},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lines = [f"# {title}", ""]
    lines += [f"- **{k}**: {v}" for k, v in meta.items()]
    for heading, rows, columns in sections:
        lines += ["", f"## {heading}", "", markdown_table(rows, columns)]
    if notes:
        lines += ["", "## Notes", "", *[f"- {n}" for n in notes]]
    md_path = results_dir / f"{name}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
