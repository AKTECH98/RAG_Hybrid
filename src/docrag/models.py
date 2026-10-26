"""Core data types shared across ingestion, indexing, retrieval and generation."""

import hashlib

from pydantic import BaseModel, Field


class Section(BaseModel):
    heading_path: list[str] = Field(default_factory=list)
    text: str
    page: int | None = None

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)


class Document(BaseModel):
    doc_id: str  # stable id, e.g. "fastapi/tutorial/body.md"
    source_path: str  # path of the raw copy under data/raw
    format: str  # md | txt | html | pdf
    title: str
    sections: list[Section]

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    strategy: str
    chunk_index: int
    text: str
    heading: str = ""
    page: int | None = None
    # The naive fixed-size baseline embeds raw windows; structure-aware strategies prepend
    # the heading path so a chunk like "Use Explicit Typing" knows it's about JWTs.
    embed_with_heading: bool = True

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def embed_text(self) -> str:
        """Text sent to the embedder (and BM25)."""
        if self.embed_with_heading and self.heading:
            return f"{self.heading}\n\n{self.text}"
        return self.text

    @staticmethod
    def make_id(doc_id: str, strategy: str, index: int, text: str) -> str:
        return hashlib.sha1(f"{doc_id}|{strategy}|{index}|{text}".encode()).hexdigest()

    def citation_label(self) -> str:
        """Human-readable source, e.g. "fastapi/body.md › Request Body > Results (p. 4)"."""
        label = f"{self.doc_id} › {self.heading}" if self.heading else self.doc_id
        return f"{label} (p. {self.page})" if self.page else label

    def metadata(self) -> dict:
        """Flat metadata for Chroma (only scalar values allowed; None is dropped)."""
        meta = {
            "doc_id": self.doc_id,
            "strategy": self.strategy,
            "chunk_index": self.chunk_index,
            "heading": self.heading,
            "char_count": self.char_count,
        }
        if self.page is not None:
            meta["page"] = self.page
        return meta


class RetrievedChunk(BaseModel):
    """A chunk plus its rank/score at every retrieval stage (ranks are 1-based)."""

    chunk: Chunk
    dense_rank: int | None = None
    dense_score: float | None = None  # cosine similarity
    sparse_rank: int | None = None
    sparse_score: float | None = None  # raw BM25
    fused_rank: int | None = None
    fused_score: float | None = None  # weighted RRF
    rerank_score: float | None = None  # 0..1 relevance from the reranker
    final_rank: int | None = None

    @property
    def relevance(self) -> float:
        """Best available 0..1 relevance estimate: reranker score, else cosine similarity."""
        if self.rerank_score is not None:
            return self.rerank_score
        return max(self.dense_score or 0.0, 0.0)
