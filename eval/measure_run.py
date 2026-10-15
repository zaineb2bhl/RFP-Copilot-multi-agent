"""Hallucination-related numbers for one pipeline run, scored by the evaluation judge.

    python eval/measure_run.py <run_id> [--force]

Judge = EVAL_JUDGE_MODEL from .env, else the reviewer (use a strong model, e.g. gemini-3.8-flash;
its free tier allows about 20 requests per day, one per response section). Each judged section
is stored in eval/results/items/measure/<run_id>/, so a run interrupted by an overloaded or
exhausted model resumes where it stopped without spending quota twice.
"""

from __future__ import annotations

import argparse
import json

from _common import setup, store

from rfp_copilot.evaluation.experiments import make_judge
from rfp_copilot.evaluation.judges import judge_faithfulness
from rfp_copilot.llm import LLMError, QuotaExhaustedError
from rfp_copilot.models import PLACEHOLDER_RE, RetrievedPassage, extract_citations
from rfp_copilot.pipeline import PipelineService

TO_CONFIRM = PLACEHOLDER_RE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("--force", action="store_true", help="re-judge stored sections")
    args = parser.parse_args()

    settings = setup()
    values = PipelineService(settings).get_snapshot(args.run_id).values
    judge = make_judge(settings)
    results = store()
    judged: dict[str, dict] = {}
    missing: list[str] = []
    for group_id, draft in values["drafts"].items():
        cached = None if args.force else results.get("measure", args.run_id, group_id)
        if cached is None:
            passages = [RetrievedPassage.model_validate(p) for p in values["passages"][group_id]]
            try:
                cached = judge_faithfulness(judge.llm, draft, passages)
            except QuotaExhaustedError as exc:
                print(f"Stopping: {exc}")
                missing += [g for g in values["drafts"] if g not in judged and g != group_id]
                missing.append(group_id)
                break
            except LLMError as exc:
                print(f"{group_id}: not judged ({str(exc)[:120]})")
                missing.append(group_id)
                continue
            results.put("measure", args.run_id, group_id, data=cached)
        judged[group_id] = cached

    claims = sum(v["claims"] or 0 for v in judged.values())
    supported = sum(v["supported_claims"] or 0 for v in judged.values())
    drafts = values["drafts"].values()
    figure_flags = sum(
        c.startswith("Figure")
        for review in values.get("reviews", {}).values()
        for c in review["unsupported_claims"]
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "requirements": len(values["requirements"]),
                "sections_judged": f"{len(judged)}/{len(values['drafts'])}",
                "to_be_confirmed_points": sum(len(TO_CONFIRM.findall(d)) for d in drafts),
                "claims": claims,
                "unsupported_claims": claims - supported,
                "faithfulness": round(supported / claims, 3) if claims else None,
                "figure_check_flags": figure_flags,
                "distinct_citations": sum(len(extract_citations(d)) for d in drafts),
                "judge": f"{judge.llm.provider}:{judge.llm.model}",
            },
            indent=2,
        )
    )
    if missing:
        print(f"Incomplete: {len(missing)} section(s) not judged; re-run to resume.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
