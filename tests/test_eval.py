from collections import Counter

import pytest

from docrag.eval.compare import estimate_cost, render_report
from docrag.eval.dataset import (
    Evidence,
    GoldenCase,
    chunk_matches,
    load_golden,
    validate_against_corpus,
)
from docrag.eval.metrics import CaseResult, retrieval_metrics, score_case
from docrag.eval.runner import ConfigReport, EvalConfig, summarize
from docrag.generation.models import Answer, Confidence, Source
from docrag.models import Chunk

from .fakes import FakeLLM


def chunk(doc_id: str, text: str, cid: str = "c") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc_id, strategy="fixed", chunk_index=0, text=text)


def test_evidence_matching_ignores_case_whitespace_and_pdf_spacing():
    c = chunk(
        "rfc.pdf", "The resource owner password credentials grant [RFC6749]MUST NOT\nbe used."
    )
    assert chunk_matches(c, Evidence(doc_id="rfc.pdf", quote="grant [RFC6749] MUST NOT be used"))
    assert not chunk_matches(c, Evidence(doc_id="other.pdf", quote="MUST NOT be used"))


def test_validate_against_corpus_reports_bad_quotes_and_docs():
    cases = [
        GoldenCase(
            id="a",
            type="lookup",
            question="q",
            answer="x",
            evidence=[Evidence(doc_id="d.md", quote="not in the text")],
        ),
        GoldenCase(
            id="b",
            type="lookup",
            question="q",
            answer="x",
            evidence=[Evidence(doc_id="missing.md", quote="x")],
        ),
        GoldenCase(id="c", type="lookup", question="q", answer="x"),
        GoldenCase(id="d", type="no_answer", question="q", answer=""),
    ]
    problems = validate_against_corpus(cases, {"d.md": "Some real text."})
    assert len(problems) == 3
    assert any("quote not found" in p for p in problems)
    assert any("unknown doc" in p for p in problems)
    assert any("without evidence" in p for p in problems)


def test_golden_set_is_well_formed():
    cases = load_golden()
    types = Counter(c.type for c in cases)
    assert len(cases) >= 50
    assert set(types) == {"lookup", "multi_hop", "no_answer", "ambiguous"}
    assert all(len(c.evidence) >= 2 for c in cases if c.type == "multi_hop")
    assert all(not c.evidence for c in cases if c.type == "no_answer")


def test_retrieval_metrics():
    case = GoldenCase(
        id="m",
        type="multi_hop",
        question="q",
        answer="a",
        evidence=[
            Evidence(doc_id="a.md", quote="alpha fact"),
            Evidence(doc_id="b.md", quote="beta fact"),
        ],
    )
    final = [chunk("x.md", "noise"), chunk("a.md", "the alpha fact is here")]
    pool = final + [chunk("b.md", "beta fact")]
    m = retrieval_metrics(case, final, pool)
    assert m == {"recall_at_k": 0.5, "hit_at_k": 1.0, "mrr": 0.5, "pool_recall": 1.0}
    assert (
        retrieval_metrics(
            GoldenCase(id="n", type="no_answer", question="q", answer=""), final, pool
        )
        == {}
    )


def _answer(status="answered", text="Use `timeout=None` to disable it [1]."):
    src = Source(
        n=1,
        chunk_id="c",
        doc_id="t.md",
        heading="T",
        page=None,
        label="t.md › T",
        text="Disable timeouts with timeout=None.",
        relevance=0.9,
    )
    conf = Confidence(
        retrieval=0.9, citation_coverage=1, completeness=1, composite=0.96, label="high"
    )
    return Answer(
        question="q",
        status=status,
        answer=text if status == "answered" else None,
        sources=[src],
        confidence=conf,
    )


JUDGE = FakeLLM(
    {
        "_Correctness": lambda s, u: {"verdict": "partially_correct", "reason": "misses default"},
        "_Faithfulness": lambda s, u: {"claims": [{"claim_id": 1, "grounded": True}]},
        "_Verdicts": lambda s, u: {
            "verdicts": [{"claim_id": 1, "source": 1, "verdict": "supported", "reason": "ok"}]
        },
    }
)


def test_score_case_answered_runs_all_judges():
    case = GoldenCase(
        id="L",
        type="lookup",
        question="q",
        answer="5s; timeout=None",
        evidence=[Evidence(doc_id="t.md", quote="timeout=None")],
    )
    final = [chunk("t.md", "Disable timeouts with timeout=None.")]
    r = score_case(JUDGE, "judge", case, _answer(), final, final, latency_ms=12.3)
    assert r.correctness == 0.5 and r.correctness_reason.startswith("partially_correct")
    assert r.faithfulness == 1.0 and r.citation_accuracy == 1.0 and r.n_citation_pairs == 1
    assert r.recall_at_k == 1.0 and not r.abstained


@pytest.mark.parametrize("ctype,expected", [("no_answer", 1.0), ("lookup", 0.0)])
def test_abstaining_is_right_only_for_unanswerable_cases(ctype, expected):
    ev = [] if ctype == "no_answer" else [Evidence(doc_id="t.md", quote="x")]
    case = GoldenCase(id="X", type=ctype, question="q", answer="", evidence=ev)
    r = score_case(FakeLLM(), "judge", case, _answer("insufficient_context"), [], [], 1.0)
    assert r.abstained and r.correctness == expected and r.faithfulness is None


def _result(cid, ctype, correct, abstained=False, recall=None):
    return CaseResult(
        case_id=cid,
        type=ctype,
        question=f"question {cid}",
        status="answered",
        answer="a",
        abstained=abstained,
        correctness=correct,
        recall_at_k=recall,
        hit_at_k=recall,
        mrr=recall,
        latency_ms=100.0,
    )


def test_summarize_rates():
    s = summarize(
        [
            _result("1", "lookup", 1.0, recall=1.0),
            _result("2", "lookup", 0.0, abstained=True, recall=0.0),
            _result("3", "no_answer", 1.0, abstained=True),
            _result("4", "no_answer", 0.0),
        ]
    )
    assert s["correctness"] == 0.5 and s["recall_at_k"] == 0.5
    assert s["no_answer_accuracy"] == 0.5 and s["false_abstain_rate"] == 0.5
    assert s["latency_p50_ms"] == 100 and s["errors"] == 0


def test_render_report_includes_matrix_and_hybrid_diff():
    def report(mode, recall):
        results = [_result("L1", "lookup", 1.0, recall=recall), _result("N1", "no_answer", 1.0)]
        return ConfigReport(
            config=EvalConfig(strategy="recursive", mode=mode, reranker="llm"),
            summary=summarize(results),
            by_type={"lookup": summarize(results[:1])},
            results=results,
        )

    meta = {
        "timestamp": "t",
        "n_cases": 2,
        "case_types": "lookup 1",
        "generation_model": "g",
        "judge_model": "j",
        "eval_judge_model": "e",
        "usage_summary": "none",
        "cost_usd": estimate_cost({"gpt-4o": {"input": 1_000_000, "output": 0}}),
    }
    md = render_report([report("hybrid", 1.0), report("dense", 0.0)], meta)
    assert "| recursive/dense/llm |" in md and "≈ $2.50" in md
    assert "hybrid found more evidence in 1 case(s), less in 0" in md
    assert "✅ L1 (0%→100%)" in md
