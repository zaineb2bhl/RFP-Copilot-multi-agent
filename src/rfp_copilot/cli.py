"""Command line: index the knowledge base, run the pipeline headless, make a sample RFP.

python -m rfp_copilot.cli index [--rebuild]
python -m rfp_copilot.cli run data/rfps/sample_rfp.pdf [--auto-approve]
python -m rfp_copilot.cli sample-rfp [--out data/rfps/sample_rfp.pdf]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rfp_copilot.config import get_settings


def _index(args: argparse.Namespace) -> int:
    from rfp_copilot.retrieval.retriever import build_retriever

    retriever = build_retriever(get_settings(), rebuild=args.rebuild)
    stats = {
        "embedding_model": retriever.embedder.model,
        "chunks": len(retriever.chunks),
        "indexed_vectors": retriever.store.count(),
    }
    print(json.dumps(stats, indent=2))
    return 0


def _run(args: argparse.Namespace) -> int:
    from rfp_copilot.pipeline import PipelineService, ReviewDecision

    pdf = Path(args.pdf)
    service = PipelineService(get_settings())
    run = service.create_run(pdf.name, pdf.read_bytes())
    print(f"Run {run.id}: started on {pdf.name}")
    run = service.execute(run.id)
    if run.status == "awaiting_review" and args.auto_approve:
        print("Auto-approving drafts (no human edits)")
        run = service.resume(run.id, ReviewDecision(action="approve"))
    payload = service.review_payload(run.id)
    summary = {
        "run_id": run.id,
        "status": run.status,
        "error": run.error,
        "requirements": len(payload["requirements"]),
        "sections": len(payload["groups"]),
        "write_passes": payload["iteration"],
        "review_scores_per_pass": payload["review_history"],
        "usage": run.usage,
        "output": run.output_path,
    }
    print(json.dumps(summary, indent=2))
    if run.status == "awaiting_review":
        print("Waiting for human review: approve it in the UI, or re-run with --auto-approve.")
    return 0 if run.status in {"completed", "awaiting_review"} else 1


def _sample_rfp(args: argparse.Namespace) -> int:
    from rfp_copilot.samples import write_sample_rfp

    path = write_sample_rfp(Path(args.out))
    print(f"Wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point of the `rfp-copilot` command."""
    parser = argparse.ArgumentParser(prog="rfp-copilot", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="(re)index the knowledge base into Chroma")
    p_index.add_argument("--rebuild", action="store_true", help="drop and rebuild the index")
    p_index.set_defaults(func=_index)

    p_run = sub.add_parser("run", help="run the pipeline on an RFP PDF")
    p_run.add_argument("pdf")
    p_run.add_argument("--auto-approve", action="store_true", help="skip human review")
    p_run.set_defaults(func=_run)

    p_sample = sub.add_parser("sample-rfp", help="write the synthetic sample RFP PDF")
    p_sample.add_argument("--out", default="data/rfps/sample_rfp.pdf")
    p_sample.set_defaults(func=_sample_rfp)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "chromadb", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
