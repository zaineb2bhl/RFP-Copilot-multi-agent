"""Tests for evaluation metrics, judges, the result store, the reranker, gold data and runners."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rfp_copilot.config import Settings
from rfp_copilot.evaluation.experiments import (
    Judge,
    eval_extraction_doc,
    eval_pipeline_doc,
    eval_rag_query,
    summarize_extraction,
    summarize_rag,
)
from rfp_copilot.evaluation.gold import GoldRequirement, load_gold_files, load_rag_queries
from rfp_copilot.evaluation.judges import judge_context_precision, judge_faithfulness
from rfp_copilot.evaluation.metrics import (
    average_precision,
    match_requirements,
    micro_average,
    score_extraction,
    score_retrieval,
    text_similarity,
)
from rfp_copilot.evaluation.store import ResultStore, markdown_table, write_report
from rfp_copilot.ingestion.pdf_parser import parse_pdf
from rfp_copilot.llm import MockLLM, UsageTracker
from rfp_copilot.models import Requirement, RetrievedPassage
from rfp_copilot.retrieval.chunking import load_knowledge_base
from rfp_copilot.retrieval.rerank import LLMReranker
from rfp_copilot.retrieval.retriever import HybridRetriever
from tests.conftest import KB_DIR, ROOT
from tests.fakes import fake_extract, make_pipeline_llm

GOLD_DIR = ROOT / "data" / "eval"


def _req(i: int, text: str, type_: str = "mandatory", page: int = 1) -> Requirement:
    return Requirement(id=f"REQ-{i:03d}", text=text, type=type_, source_page=page)


def _passage(chunk_id: str, source: str = "kb.md") -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id, source=source, title="T", text=f"text {chunk_id}", score=1
    )


# -- metrics ---------------------------------------------------------------------------


def test_text_similarity() -> None:
    assert text_similarity(
        "Must hold ISO 27001", "The supplier must hold ISO 27001"
    ) == pytest.approx(2 * 3 / (3 + 4))
    assert text_similarity("", "anything") == 0.0


def test_matching_is_one_to_one_and_greedy() -> None:
    gold = ["hosting in the EU or UK", "ISO 27001 certification required"]
    pred = ["ISO 27001 certification required", "ISO 27001 certification", "hosting in EU"]
    matches = match_requirements(pred, gold, threshold=0.5)
    assert [(p, g) for p, g, _ in matches] == [(2, 0), (0, 1)]


def test_score_extraction_counts_and_accuracies() -> None:
    gold = [
        GoldRequirement(
            id="G1", text="Hosting must be in the EU or UK", type="mandatory", source_page=1
        ),
        GoldRequirement(id="G2", text="Provide CVs of key staff", type="desirable", source_page=2),
        GoldRequirement(id="G3", text="Submit one PDF", type="administrative", source_page=3),
    ]
    pred = [
        _req(1, "Hosting must be in the EU or UK", "mandatory", 1),
        _req(2, "Provide CVs of the key staff", "mandatory", 2),
        _req(3, "Unrelated invented requirement about pizza"),
    ]
    s = score_extraction(pred, gold)
    assert (s.tp, s.fp, s.fn) == (2, 1, 1)
    assert s.precision == pytest.approx(2 / 3) and s.recall == pytest.approx(2 / 3)
    assert s.type_accuracy == 0.5 and s.page_accuracy == 1.0
    empty = score_extraction([], gold)
    assert (empty.precision, empty.recall, empty.f1, empty.type_accuracy) == (0.0, 0.0, 0.0, None)


def test_micro_average_pools_counts() -> None:
    gold = [GoldRequirement(id="G1", text="Hosting in the EU", type="mandatory", source_page=1)]
    a = score_extraction([_req(1, "Hosting in the EU")], gold)
    b = score_extraction([_req(1, "pizza"), _req(2, "pasta")], gold)
    pooled = micro_average([a, b])
    assert (pooled.tp, pooled.fp, pooled.fn) == (1, 2, 1)
    assert pooled.precision == pytest.approx(1 / 3)


def test_retrieval_scores_and_average_precision() -> None:
    s = score_retrieval(["x.md", "a.md", "x.md", "b.md"], {"a.md", "b.md", "c.md"})
    assert (s.hit, s.precision, s.reciprocal_rank) == (1.0, 0.5, 0.5)
    assert s.recall == pytest.approx(2 / 3)
    assert score_retrieval([], {"a.md"}).hit == 0.0
    assert average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([False, False]) == 0.0


# -- judges ----------------------------------------------------------------------------


def test_context_precision_judge() -> None:
    judge = MockLLM(['{"relevant_ids": ["b#1", "zzz#9"]}'])
    out = judge_context_precision(judge, "q", [_passage("a#1"), _passage("b#1")])
    assert out["context_precision"] == 0.5
    assert out["context_avg_precision"] == 0.5  # first relevant at rank 2
    assert out["relevant_ids"] == ["b#1"]


def test_faithfulness_judge_and_failures() -> None:
    verdict = {
        "claims": [
            {"claim": "ISO certified", "supported": True},
            {"claim": "x", "supported": False},
        ]
    }
    out = judge_faithfulness(MockLLM([json.dumps(verdict)]), "answer", [_passage("a#1")])
    assert out == {"faithfulness": 0.5, "claims": 2, "supported_claims": 1}
    assert judge_faithfulness(MockLLM(['{"claims": []}']), "a", [])["faithfulness"] is None
    broken = judge_faithfulness(MockLLM(lambda _m: "nope"), "a", [])
    assert broken["faithfulness"] is None


# -- store & reports ---------------------------------------------------------------------


def test_result_store_roundtrip(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    assert store.get("rag", "cfg/1", "Q 01") is None
    store.put("rag", "cfg/1", "Q 01", data={"row": {"x": 1}})
    assert store.get("rag", "cfg/1", "Q 01") == {"row": {"x": 1}}
    assert store.path("rag", "cfg/1", "Q 01").name == "Q_01.json"


def test_reports(tmp_path: Path) -> None:
    table = markdown_table(
        [{"a": 0.12345, "b": None, "c": "x|y"}], [("a", "A"), ("b", "B"), ("c", "C")]
    )
    assert table.splitlines()[2] == "| 0.123 | - | x\\|y |"
    json_path, md_path = write_report(
        tmp_path, "demo", title="Demo", sections=[("S", [{"a": 1}], [("a", "A")])], meta={"k": 3}
    )
    assert json.loads(json_path.read_text())["sections"]["S"] == [{"a": 1}]
    assert "## S" in md_path.read_text() and "**k**: 3" in md_path.read_text()


# -- reranker --------------------------------------------------------------------------


def test_reranker_orders_and_repairs_ids() -> None:
    passages = [_passage("a#1"), _passage("b#1"), _passage("c#1")]
    llm = MockLLM(['{"ranked_ids": ["c#1", "invented#1", "c#1", "a#1"]}'])
    ranked = LLMReranker(llm).rerank("q", passages, k=3)
    assert [p.chunk_id for p in ranked] == ["c#1", "a#1", "b#1"]  # omitted b#1 keeps its place


def test_reranker_falls_back_on_failure() -> None:
    passages = [_passage("a#1"), _passage("b#1")]
    ranked = LLMReranker(MockLLM(lambda _m: "garbage")).rerank("q", passages, k=1)
    assert [p.chunk_id for p in ranked] == ["a#1"]


def test_retriever_uses_reranker(retriever: HybridRetriever) -> None:
    def reverse(messages: list[dict[str, str]]) -> str:
        ids = re.findall(r"^\[([^\]]+)\]", messages[-1]["content"], re.MULTILINE)
        return json.dumps({"ranked_ids": ids[::-1]})

    plain = retriever.search("ISO 27001 certification", k=12)
    retriever.reranker = LLMReranker(MockLLM(reverse))
    retriever.rerank_candidates = 12
    reranked = retriever.search("ISO 27001 certification", k=3)
    assert [p.chunk_id for p in reranked] == [p.chunk_id for p in plain[::-1][:3]]


# -- gold data ---------------------------------------------------------------------------


def test_gold_files_load_and_match_the_pdf() -> None:
    gold_files = load_gold_files(GOLD_DIR)
    assert gold_files, "data/eval must contain at least the sample gold file"
    for _path, gold in gold_files:
        pdf = gold.rfp_path(ROOT)
        if not pdf.exists():  # user-provided RFPs are not committed
            continue
        pages = [re.sub(r"[^a-z0-9]+", " ", t.lower()) for t in parse_pdf(pdf).page_texts]
        for req in gold.requirements:
            needle = re.sub(r"[^a-z0-9]+", " ", req.text.lower()).strip()[:50]
            assert needle in pages[req.source_page - 1], req.id


def test_rag_queries_reference_existing_sources() -> None:
    sources = {c.source for c in load_knowledge_base(KB_DIR)}
    queries = load_rag_queries(GOLD_DIR / "rag_queries.json")
    assert len(queries) >= 15
    assert len({q.id for q in queries}) == len(queries)
    for query in queries:
        assert set(query.relevant_sources) <= sources, query.id


# -- runners (offline, mock LLM + hash embeddings) ----------------------------------------


def test_extraction_runner_is_resumable(settings: Settings, tmp_path: Path, monkeypatch) -> None:
    import rfp_copilot.evaluation.experiments as experiments

    path, gold = next((p, g) for p, g in load_gold_files(GOLD_DIR) if p.name.startswith("sample"))
    store = ResultStore(tmp_path / "items")
    calls = []

    def fake_build_llm(_settings, _role, usage=None):
        calls.append(1)
        return MockLLM(lambda m: fake_extract(m[-1]["content"]), usage=usage)

    monkeypatch.setattr(experiments, "build_llm", fake_build_llm)
    row = eval_extraction_doc(settings, path, gold, store, "mock", root=ROOT)
    again = eval_extraction_doc(settings, path, gold, store, "mock", root=ROOT)
    assert row == again and len(calls) == 1  # second call served from the store
    assert row["gold"] == 28 and row["recall"] > 0.8  # rule-based fake finds most of them
    summary = summarize_extraction([row], "mock")
    assert summary["f1"] == pytest.approx(row["f1"])


def test_rag_runner_end_to_end(
    settings: Settings, retriever: HybridRetriever, tmp_path: Path
) -> None:
    query = load_rag_queries(GOLD_DIR / "rag_queries.json")[0]

    def judge_handler(messages: list[dict[str, str]]) -> str:
        if (
            "relevant_ids" in messages[0]["content"]
            or "retrieval quality" in messages[0]["content"]
        ):
            ids = re.findall(r"^\[([^\]]+)\]", messages[-1]["content"], re.MULTILINE)
            return json.dumps({"relevant_ids": ids[:2]})
        return json.dumps({"claims": [{"claim": "c", "supported": True}]})

    judge_usage = UsageTracker()
    judge = Judge(MockLLM(judge_handler, usage=judge_usage), judge_usage)
    row = eval_rag_query(
        settings, retriever, UsageTracker(), query, ResultStore(tmp_path), "mock", judge=judge
    )
    assert row["k"] == settings.retrieval_k
    assert 0 <= row["precision"] <= 1 and row["hit"] in (0.0, 1.0)
    assert row["context_precision"] == pytest.approx(2 / settings.retrieval_k)
    assert row["faithfulness"] == 1.0
    assert row["judge_tokens"] > 0
    summary = summarize_rag([row], "mock")
    assert summary["queries"] == 1 and summary["faithfulness"] == 1.0


def test_pipeline_runner(
    settings: Settings, retriever: HybridRetriever, tmp_path: Path, monkeypatch
) -> None:
    import rfp_copilot.pipeline as pipeline_module

    handler = make_pipeline_llm()
    monkeypatch.setattr(
        pipeline_module, "build_llm", lambda s, role, usage=None: MockLLM(handler, usage=usage)
    )
    monkeypatch.setattr(pipeline_module, "build_retriever", lambda s: retriever)
    path, gold = next((p, g) for p, g in load_gold_files(GOLD_DIR) if p.name.startswith("sample"))
    judge_usage = UsageTracker()
    judge = Judge(
        MockLLM(lambda _m: '{"claims": [{"claim": "c", "supported": false}]}', usage=judge_usage),
        judge_usage,
    )
    row = eval_pipeline_doc(
        settings, path, gold, ResultStore(tmp_path), "loop", root=ROOT, judge=judge
    )
    assert row["write_passes"] == 2  # fake reviewer fails every section once
    assert row["reviewer_score"] == pytest.approx(0.9)
    assert row["faithfulness"] == 0.0
    assert row["extraction_f1"] > 0.5
    assert row["tokens"] > 0
