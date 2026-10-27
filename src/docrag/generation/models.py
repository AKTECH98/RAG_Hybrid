"""Response types for the generation layer (also the /v1/ask API schema)."""

from typing import Literal

from pydantic import BaseModel, Field, computed_field

from docrag.models import RetrievedChunk

Verdict = Literal["supported", "partial", "unsupported"]


class Source(BaseModel):
    """A numbered context block the answer can cite as [n]."""

    n: int
    chunk_id: str
    doc_id: str
    heading: str
    page: int | None
    label: str
    text: str
    relevance: float  # calibrated 0..1
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None

    @classmethod
    def from_retrieved(cls, n: int, rc: RetrievedChunk, relevance: float) -> "Source":
        c = rc.chunk
        return cls(
            n=n,
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            heading=c.heading,
            page=c.page,
            label=c.citation_label(),
            text=c.text,
            relevance=round(relevance, 3),
            dense_rank=rc.dense_rank,
            sparse_rank=rc.sparse_rank,
            fused_rank=rc.fused_rank,
            rerank_score=rc.rerank_score,
        )


class CitationCheck(BaseModel):
    claim_id: int
    source: int
    verdict: Verdict
    reason: str = ""


class Claim(BaseModel):
    id: int
    text: str
    citations: list[int]  # [n] markers attached to this claim
    invalid_citations: list[int] = Field(default_factory=list)  # [n] with no such source
    checks: list[CitationCheck] = Field(default_factory=list)

    @computed_field  # serialized, so API clients get it without re-deriving
    @property
    def status(self) -> Literal["supported", "partial", "unsupported", "uncited"]:
        if not self.citations:
            return "uncited"
        verdicts = {c.verdict for c in self.checks}
        if "supported" in verdicts:
            return "supported"
        if "partial" in verdicts:
            return "partial"
        return "unsupported"


class Confidence(BaseModel):
    retrieval: float  # how relevant were the top chunks
    citation_coverage: float  # share of claims backed by a verified citation
    completeness: float  # share of the question's parts that were addressed
    composite: float
    label: Literal["high", "medium", "low"]
    question_parts: list[dict] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    status: Literal["answered", "insufficient_context"]
    answer: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    unsupported_citations: list[CitationCheck] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    confidence: Confidence | None = None
    # populated when status == "insufficient_context"
    found: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    suggested_documents: list[str] = Field(default_factory=list)
    retrieval: dict = Field(default_factory=dict)  # mode/strategy/reranker/timings
    timings_ms: dict[str, float] = Field(default_factory=dict)
