import pytest

from docrag.generation.citations import extract_citations, parse_claims
from docrag.generation.confidence import (
    calibrated_relevance,
    citation_coverage,
    composite_confidence,
    retrieval_confidence,
)
from docrag.generation.generator import Generator
from docrag.generation.models import CitationCheck, Claim
from docrag.models import Chunk, RetrievedChunk
from docrag.retrieval.pipeline import RetrievalResult

from .fakes import FakeLLM

# ------------------------------------------------------------------ parsing


def test_extract_citations_handles_all_marker_styles():
    assert extract_citations("A [1] b [2][4] c [1, 3].") == [1, 2, 4, 3]
    assert extract_citations("no markers") == []


def test_parse_claims_splits_sentences_and_bullets_and_skips_code():
    answer = """Use `HTTPException` to return errors [1]. Pass `status_code=404` [1][2].

Here's how:

```python
raise HTTPException(status_code=404)  # [9] inside code is ignored
```

- The `detail` can be any JSON-serializable value [2].
- Custom headers are supported. [3]
The provided documentation does not cover rate limiting.
Therefore, it does not cover the failure handling of multiple tasks.
It does not mention pagination at all.
Short."""
    claims = parse_claims(answer, n_sources=3)
    assert [c.text for c in claims] == [
        "Use `HTTPException` to return errors",
        "Pass `status_code=404`",
        "The `detail` can be any JSON-serializable value",
        "Custom headers are supported",
    ]
    assert [c.citations for c in claims] == [[1], [1, 2], [2], [3]]  # "[3]" after the period


def test_colon_lines_are_transitions_only_when_they_read_like_one():
    answer = (
        "Here is how you can do it for an individual request:\n```python\nx()\n```\n"
        "You can also disable all timeouts on a client instance:\n```python\ny()\n```"
    )
    claims = parse_claims(answer, n_sources=2)
    assert [c.text for c in claims] == ["You can also disable all timeouts on a client instance:"]
    assert claims[0].status == "uncited"  # a fact the model forgot to cite still counts


def test_parse_claims_flags_citations_to_missing_sources_and_keeps_uncited_facts():
    claims = parse_claims("Timeouts default to 5 seconds [7]. Retries are automatic in httpx.", 3)
    assert claims[0].citations == [] and claims[0].invalid_citations == [7]
    assert claims[1].citations == [] and claims[1].status == "uncited"


# ------------------------------------------------------------------ confidence


def _rc(dense=None, rerank=None) -> RetrievedChunk:
    chunk = Chunk(chunk_id="c", doc_id="d.md", strategy="recursive", chunk_index=0, text="t")
    return RetrievedChunk(chunk=chunk, dense_score=dense, rerank_score=rerank)


def test_relevance_calibration():
    band = (0.45, 0.65)
    assert calibrated_relevance(_rc(dense=0.30), band) == 0.0
    assert calibrated_relevance(_rc(dense=0.55), band) == pytest.approx(0.5)
    assert calibrated_relevance(_rc(dense=0.80), band) == 1.0
    # cosine and reranker are averaged; either alone is used as is
    assert calibrated_relevance(_rc(dense=0.65, rerank=0.2), band) == pytest.approx(0.6)
    assert calibrated_relevance(_rc(rerank=0.2), band) == 0.2
    assert calibrated_relevance(_rc(), band) == 0.0
    # a bge-scale cosine (0.75) is "certain" under OpenAI's band but middling under bge's
    assert calibrated_relevance(_rc(dense=0.75), band) == 1.0
    assert calibrated_relevance(_rc(dense=0.75), (0.70, 0.78)) == pytest.approx(0.625)


def test_retrieval_confidence_and_coverage_math():
    assert retrieval_confidence([]) == 0.0
    assert retrieval_confidence([1.0, 0.5, 0.0, 0.9]) == pytest.approx(0.6 * 1.0 + 0.4 * 0.8)

    def claim(i, verdicts, cites=(1,)):
        return Claim(
            id=i,
            text="x",
            citations=list(cites),
            checks=[CitationCheck(claim_id=i, source=1, verdict=v) for v in verdicts],
        )

    claims = [
        claim(1, ["unsupported", "supported"]),  # any supporting source is enough
        claim(2, ["partial"]),
        claim(3, ["unsupported"]),
        claim(4, [], cites=()),  # uncited
    ]
    assert [c.status for c in claims] == ["supported", "partial", "unsupported", "uncited"]
    assert citation_coverage(claims) == pytest.approx(1.5 / 4)


def test_composite_confidence_labels():
    w = (0.4, 0.4, 0.2)
    assert composite_confidence(1, 1, 1, w).label == "high"
    assert composite_confidence(0.6, 0.5, 0.5, w).label == "medium"
    low = composite_confidence(0.2, 0.0, 0.0, w)
    assert low.label == "low" and low.composite == pytest.approx(0.08)


# ------------------------------------------------------------------ generator


def _result(*texts_and_scores) -> RetrievalResult:
    results = []
    for i, (text, score) in enumerate(texts_and_scores, start=1):
        chunk = Chunk(
            chunk_id=f"c{i}",
            doc_id=f"doc{i}.md",
            strategy="recursive",
            chunk_index=0,
            text=text,
            heading=f"H{i}",
        )
        results.append(RetrievedChunk(chunk=chunk, rerank_score=score, final_rank=i))
    return RetrievalResult(
        query="q",
        mode="hybrid",
        strategy="recursive",
        reranker="llm",
        candidates=results,
        results=results,
        timings_ms={},
    )


GOOD = _result(
    ("Raise HTTPException with status_code=404 to return a not found error.", 0.9),
    ("The detail field of HTTPException can be any JSON value.", 0.8),
)


def _llm(answer: str, verdicts: list[dict], parts=("yes",)) -> FakeLLM:
    return FakeLLM(
        {
            "_Verdicts": lambda s, u: {"verdicts": verdicts},
            "_Parts": lambda s, u: {
                "parts": [{"part": f"p{i}", "addressed": a} for i, a in enumerate(parts)]
            },
            "_Gaps": lambda s, u: {"found": ["Something related [1]."], "missing": ["The X"]},
        },
        text=answer,
    )


def test_answer_with_verified_citations(settings):
    answer = "Raise `HTTPException(status_code=404)` [1]. Its `detail` can be any JSON value [2]."
    llm = _llm(
        answer,
        [
            {"claim_id": 1, "source": 1, "verdict": "supported", "reason": "stated"},
            {"claim_id": 2, "source": 2, "verdict": "supported", "reason": "stated"},
        ],
    )
    ans = Generator(retriever=None, llm=llm, settings=settings).answer_from("How do I 404?", GOOD)

    assert ans.status == "answered" and ans.answer == answer
    assert [c.status for c in ans.claims] == ["supported", "supported"]
    assert ans.unsupported_citations == []
    assert ans.confidence.citation_coverage == 1.0 and ans.confidence.label == "high"
    assert [s.n for s in ans.sources] == [1, 2] and ans.sources[0].label == "doc1.md › H1"
    # the generation prompt numbers the context blocks the same way
    gen_prompt = next(c for c in llm.calls if c[0] == "text")[2]
    assert "[1] Source: doc1.md › H1" in gen_prompt and "[2] Source: doc2.md › H2" in gen_prompt


def test_hallucinated_citation_is_flagged(settings):
    answer = "Raise `HTTPException` [1]. The default timeout is 30 seconds [2]. It retries [5]."
    llm = _llm(
        answer,
        [
            {"claim_id": 1, "source": 1, "verdict": "supported", "reason": "stated"},
            {"claim_id": 2, "source": 2, "verdict": "unsupported", "reason": "no timeout in [2]"},
        ],
    )
    ans = Generator(None, llm, settings).answer_from("q", GOOD)

    assert [c.status for c in ans.claims] == ["supported", "unsupported", "uncited"]
    flagged = {(u.claim_id, u.source) for u in ans.unsupported_citations}
    assert flagged == {(2, 2), (3, 5)}  # judged unsupported + cites a source that doesn't exist
    assert ans.confidence.citation_coverage == pytest.approx(1 / 3, abs=1e-3)


def test_missing_judge_verdict_counts_as_unsupported(settings):
    other = {"claim_id": 99, "source": 1, "verdict": "supported", "reason": "wrong pair"}
    llm = _llm("Raise `HTTPException` with a code [1].", verdicts=[other])
    ans = Generator(None, llm, settings).answer_from("q", GOOD)
    assert ans.claims[0].checks[0].verdict == "unsupported"
    assert "no verdict" in ans.claims[0].checks[0].reason


def test_low_retrieval_confidence_abstains_without_generating(settings):
    weak = _result(("Unrelated text about cookies.", 0.1), ("More unrelated text.", 0.05))
    llm = _llm("should never be generated", [])
    ans = Generator(None, llm, settings).answer_from("How do I configure Kubernetes?", weak)

    assert ans.status == "insufficient_context" and ans.answer is None
    assert not any(kind == "text" for kind, *_ in llm.calls)  # generation was skipped
    assert ans.found == ["Something related [1]."] and ans.missing == ["The X"]
    assert ans.suggested_documents == ["doc1.md", "doc2.md"]
    assert ans.confidence.label == "low"


def test_model_reported_insufficient_context_abstains(settings):
    llm = _llm("INSUFFICIENT_CONTEXT: The context only covers HTTP errors.", [])
    ans = Generator(None, llm, settings).answer_from("What is the meaning of life?", GOOD)
    assert ans.status == "insufficient_context"
    assert ans.missing == ["The X"]


def test_low_citation_coverage_caps_label_even_with_strong_retrieval():
    c = composite_confidence(1.0, 0.33, 1.0, (0.4, 0.4, 0.2))
    assert c.composite > 0.7 and c.label == "low"
