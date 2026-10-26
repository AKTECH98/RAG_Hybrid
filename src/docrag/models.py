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
