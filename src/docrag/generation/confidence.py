"""Answer confidence: retrieval relevance, verified citation coverage, completeness.

    composite = w_r * retrieval + w_c * citation_coverage + w_q * completeness

Each part is 0..1 and returned on its own, so a low score says *why* it is low.
"""

from typing import Literal

from pydantic import BaseModel

from docrag.generation.models import Claim, Confidence
from docrag.llm import LLM
from docrag.models import RetrievedChunk

# Raw cosine similarities from text-embedding-3-small sit in a narrow band. Measured on
# this corpus: off-topic questions top out at 0.13-0.23, near-domain questions the docs
# don't answer (Django upload limits, FastAPI + MongoDB) at 0.46-0.49, and answerable
# questions at 0.59-0.73. Map 0.45..0.65 onto 0..1 when no reranker ran.
COSINE_FLOOR = 0.45
COSINE_CEIL = 0.65


def calibrated_relevance(rc: RetrievedChunk) -> float:
    if rc.rerank_score is not None:
        return rc.rerank_score  # rerankers already emit 0..1 relevance
    cos = rc.dense_score or 0.0
    return min(max((cos - COSINE_FLOOR) / (COSINE_CEIL - COSINE_FLOOR), 0.0), 1.0)


def retrieval_confidence(relevances: list[float]) -> float:
    """Mostly the best chunk, tempered by how the next couple of chunks look."""
    if not relevances:
        return 0.0
    top = sorted(relevances, reverse=True)
    return 0.6 * top[0] + 0.4 * (sum(top[:3]) / len(top[:3]))


CLAIM_SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0, "uncited": 0.0}


def citation_coverage(claims: list[Claim]) -> float:
    if not claims:
        return 0.0
    return sum(CLAIM_SCORE[c.status] for c in claims) / len(claims)


class _Part(BaseModel):
    part: str
    addressed: Literal["yes", "partial", "no"]


class _Parts(BaseModel):
    parts: list[_Part]


COMPLETENESS_SYSTEM = """You check whether an answer addresses everything a question asks.
Split the question into its distinct parts (often just one; at most five). For each part, \
say whether the answer addresses it: "yes", "partial", or "no". An answer that explicitly \
says the documentation does not cover a part counts as "no" for that part. Judge coverage \
only, not correctness."""

PART_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def completeness(llm: LLM, model: str, question: str, answer: str) -> tuple[float, list[dict]]:
    result = llm.structured(
        model, COMPLETENESS_SYSTEM, f"Question: {question}\n\nAnswer:\n{answer}", _Parts
    )
    if not result.parts:
        return 0.0, []
    score = sum(PART_SCORE[p.addressed] for p in result.parts) / len(result.parts)
    return score, [p.model_dump() for p in result.parts]


def composite_confidence(
    retrieval: float,
    coverage: float,
    complete: float,
    weights: tuple[float, float, float],
    parts: list[dict] | None = None,
) -> Confidence:
    total = sum(weights) or 1.0
    score = (weights[0] * retrieval + weights[1] * coverage + weights[2] * complete) / total
    label = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    # Great retrieval can't vouch for an answer whose claims mostly failed verification
    if coverage < 0.5:
        label = "low"
    return Confidence(
        retrieval=round(retrieval, 3),
        citation_coverage=round(coverage, 3),
        completeness=round(complete, 3),
        composite=round(score, 3),
        label=label,
        question_parts=parts or [],
    )
