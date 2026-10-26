"""Chunker interface and shared helpers."""

from abc import ABC, abstractmethod

from docrag.models import Chunk, Document, Section


class Chunker(ABC):
    name: str

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]: ...

    def make_chunks(
        self, doc: Document, pieces: list[tuple[str, str, int | None]], with_heading: bool = True
    ) -> list[Chunk]:
        """Turn (text, heading, page) pieces into Chunks with deterministic ids."""
        chunks = []
        for text, heading, page in pieces:
            text = text.strip("\n")
            if not text.strip():
                continue
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(doc.doc_id, self.name, idx, text),
                    doc_id=doc.doc_id,
                    strategy=self.name,
                    chunk_index=idx,
                    text=text,
                    heading=heading,
                    page=page,
                    embed_with_heading=with_heading,
                )
            )
        return chunks


def merge_page_pieces(sections: list[Section]) -> list[Section]:
    """PDF sections are split per page; rejoin consecutive pieces of the same heading."""
    merged: list[Section] = []
    for s in sections:
        if merged and merged[-1].heading_path == s.heading_path:
            merged[-1] = Section(
                heading_path=s.heading_path,
                text=f"{merged[-1].text}\n{s.text}",
                page=merged[-1].page,
            )
        else:
            merged.append(s.model_copy())
    return merged
