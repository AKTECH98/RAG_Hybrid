"""Per-case eval metrics.

retrieval     evidence recall@k, hit@k and MRR over the final top-k, plus recall over the
              whole candidate pool (shows what the reranker had to work with)
correctness   LLM judge: answer vs golden answer -> 1 / 0.5 / 0
faithfulness  LLM judge: share of answer claims grounded in the retrieved context,
              cited or not (catches hallucination independent of citation formatting)
citations     LLM judge: share of (claim, cited source) pairs where the source supports
              the claim (partial = 0.5)
abstention    abstained on no-answer cases (good) vs on answerable cases (false abstain)

The eval judge is a separate, stronger model than the system's own verifier, so the
system is not grading itself.
"""

from typing import Literal

from pydantic import BaseModel

from docrag.eval.dataset import GoldenCase, chunk_matches
from docrag.generation.citations import parse_claims, verify_claims
from docrag.generation.models import Answer, Source
from docrag.generation.prompt import format_context
from docrag.llm import LLM
from docrag.models import Chunk


class CaseResult(BaseModel):
    case_id: str
    type: str
    question: str
    status: str
    answer: str | None
    abstained: bool
    # retrieval (None for no-answer cases)
    recall_at_k: float | None = None
    hit_at_k: float | None = None
    mrr: float | None = None
    pool_recall: float | None = None
    # generation
    correctness: float | None = None
    correctness_reason: str = ""
    faithfulness: float | None = None
    citation_accuracy: float | None = None
    n_claims: int = 0
    n_citation_pairs: int = 0
    unsupported_pairs: list[str] = []
    confidence: float | None = None
    confidence_label: str | None = None
    latency_ms: float = 0.0
    sources: list[str] = []
    error: str | None = None


# ------------------------------------------------------------------ retrieval


def retrieval_metrics(case: GoldenCase, final: list[Chunk], pool: list[Chunk]) -> dict[str, float]:
    """Each evidence quote is one relevant item; a chunk containing it counts as a hit."""
    ev = case.evidence
    if not ev:
        return {}
    found = [any(chunk_matches(c, e) for c in final) for e in ev]
    first = next(
        (rank for rank, c in enumerate(final, start=1) if any(chunk_matches(c, e) for e in ev)),
        None,
    )
    pool_found = [any(chunk_matches(c, e) for c in pool) for e in ev]
    return {
        "recall_at_k": sum(found) / len(ev),
        "hit_at_k": 1.0 if any(found) else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "pool_recall": sum(pool_found) / len(ev),
    }


# ------------------------------------------------------------------ judges


class _Correctness(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    reason: str


CORRECTNESS_SYSTEM = """You grade answers from a documentation Q&A system against a \
reference answer written by an expert.
- "correct": contains the key facts of the reference and nothing that contradicts it. \
Extra correct detail, different wording or a different but valid code example is fine.
- "partially_correct": gets some key facts but misses an important one, or includes a \
minor error.
- "incorrect": misses the main point, contradicts the reference, or is mostly fabricated.
For questions marked UNANSWERABLE, the documentation does not contain the answer: a \
response that says the information isn't available is "correct"; any response that \
answers anyway is "incorrect".
For AMBIGUOUS questions, accept any reasonable interpretation the reference lists."""

CORRECTNESS_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def judge_correctness(llm: LLM, model: str, case: GoldenCase, answer: str) -> tuple[float, str]:
    if case.type == "no_answer":
        reference = "UNANSWERABLE: the documentation does not cover this. " + case.notes
    elif case.type == "ambiguous":
        reference = "AMBIGUOUS question. " + case.answer
    else:
        reference = case.answer
    r = llm.structured(
        model,
        CORRECTNESS_SYSTEM,
        f"Question: {case.question}\n\nReference answer: {reference}\n\nSystem answer:\n{answer}",
        _Correctness,
    )
    return CORRECTNESS_SCORE[r.verdict], f"{r.verdict}: {r.reason}"


class _Grounded(BaseModel):
    claim_id: int
    grounded: bool


class _Faithfulness(BaseModel):
    claims: list[_Grounded]


FAITHFULNESS_SYSTEM = """You check whether statements are grounded in a set of context \
passages. A statement is grounded if the passages (taken together) state it or it follows \
directly from them. Ignore whether the statement has citations. Be strict about specific \
values, defaults, names and recommendations: if the passages don't contain them, the \
statement is not grounded. Return one entry per statement."""


def judge_faithfulness(llm: LLM, model: str, answer: str, sources: list[Source]) -> float | None:
    claims = parse_claims(answer, len(sources))
    if not claims:
        return None
    listed = "\n".join(f"{c.id}. {c.text}" for c in claims)
    r = llm.structured(
        model,
        FAITHFULNESS_SYSTEM,
        f"Context passages:\n\n{format_context(sources)}\n\nStatements:\n{listed}",
        _Faithfulness,
    )
    verdicts = {g.claim_id: g.grounded for g in r.claims}
    return sum(verdicts.get(c.id, False) for c in claims) / len(claims)


def judge_citations(
    llm: LLM, model: str, question: str, answer: str, sources: list[Source]
) -> tuple[float | None, int, list[str]]:
    """Independent re-check of every (claim, cited source) pair with the eval judge."""
    claims = parse_claims(answer, len(sources))
    checks = verify_claims(llm, model, question, claims, sources)
    invalid = sum(len(c.invalid_citations) for c in claims)  # [n] with no such source
    total = len(checks) + invalid
    if total == 0:
        return None, 0, []
    score = sum({"supported": 1.0, "partial": 0.5}.get(c.verdict, 0.0) for c in checks)
    bad = [
        f"claim {c.claim_id} -> [{c.source}]: {c.reason}"
        for c in checks
        if c.verdict == "unsupported"
    ]
    return score / total, total, bad


# ------------------------------------------------------------------ one case


def score_case(
    llm: LLM,
    judge_model: str,
    case: GoldenCase,
    answer: Answer,
    final: list[Chunk],
    pool: list[Chunk],
    latency_ms: float,
) -> CaseResult:
    abstained = answer.status == "insufficient_context"
    res = CaseResult(
        case_id=case.id,
        type=case.type,
        question=case.question,
        status=answer.status,
        answer=answer.answer,
        abstained=abstained,
        confidence=answer.confidence.composite if answer.confidence else None,
        confidence_label=answer.confidence.label if answer.confidence else None,
        latency_ms=round(latency_ms, 1),
        sources=[s.label for s in answer.sources],
        **retrieval_metrics(case, final, pool),
    )
    if abstained:
        res.correctness = 1.0 if case.type == "no_answer" else 0.0
        res.correctness_reason = "abstained"
        return res

    text = answer.answer or ""
    res.correctness, res.correctness_reason = judge_correctness(llm, judge_model, case, text)
    res.faithfulness = judge_faithfulness(llm, judge_model, text, answer.sources)
    res.citation_accuracy, res.n_citation_pairs, res.unsupported_pairs = judge_citations(
        llm, judge_model, case.question, text, answer.sources
    )
    res.n_claims = len(answer.claims)
    return res
