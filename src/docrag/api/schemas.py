"""Request/response models for the HTTP API (the answer itself is generation.models.Answer)."""

from pydantic import BaseModel, Field

from docrag.config import ChunkStrategy, RerankerName, RetrievalMode
from docrag.generation.models import Answer
from docrag.models import RetrievedChunk


class RetrievalOptions(BaseModel):
    mode: RetrievalMode | None = Field(None, description="Default: DOCRAG_RETRIEVAL_MODE")
    strategy: ChunkStrategy | None = Field(None, description="Default: DOCRAG_DEFAULT_STRATEGY")
    reranker: RerankerName | None = Field(None, description="Default: DOCRAG_RERANKER")
    top_n: int | None = Field(None, ge=1, le=20, description="Chunks given to the generator")


class AskRequest(RetrievalOptions):
    question: str = Field(min_length=3, max_length=2000)
    verify: bool = Field(True, description="Verify every citation with the judge model")
    debug: bool = Field(False, description="Also return the whole reranker candidate pool")


class RetrieveRequest(RetrievalOptions):
    query: str = Field(min_length=1, max_length=2000)


class Candidate(BaseModel):
    """One chunk with its rank and score at every retrieval stage."""

    chunk_id: str
    doc_id: str
    label: str
    text: str
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None
    fused_rank: int | None
    fused_score: float | None
    rerank_score: float | None
    final_rank: int | None

    @classmethod
    def from_retrieved(cls, rc: RetrievedChunk, final_rank: int | None) -> "Candidate":
        return cls(
            chunk_id=rc.chunk.chunk_id,
            doc_id=rc.chunk.doc_id,
            label=rc.chunk.citation_label(),
            text=rc.chunk.text,
            dense_rank=rc.dense_rank,
            dense_score=rc.dense_score,
            sparse_rank=rc.sparse_rank,
            sparse_score=rc.sparse_score,
            fused_rank=rc.fused_rank,
            fused_score=rc.fused_score,
            rerank_score=rc.rerank_score,
            final_rank=final_rank,
        )


class AskResponse(Answer):
    candidates: list[Candidate] | None = None  # only with debug=true


class RetrieveResponse(BaseModel):
    query: str
    mode: RetrievalMode
    strategy: ChunkStrategy
    reranker: RerankerName
    candidates: list[Candidate]  # the pool sent to the reranker, in fused order
    timings_ms: dict[str, float]


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    format: str
    sections: int
    chars: int
    chunks: dict[str, int]  # per chunking strategy


class StrategyStats(BaseModel):
    chunks: int
    duplicates: int


class IngestResponse(BaseModel):
    doc_id: str
    title: str
    format: str
    sections: int
    strategies: dict[str, StrategyStats]
    elapsed_ms: float


class Health(BaseModel):
    status: str
    llm_provider: str
    generation_model: str
    embedder: str
    reranker: str
    chunks: dict[str, int]  # per chunking strategy
    in_sync: bool
