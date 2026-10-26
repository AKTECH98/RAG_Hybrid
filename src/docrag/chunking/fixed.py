"""Baseline: fixed-size token windows with overlap, blind to document structure."""

import bisect

import tiktoken

from docrag.chunking.base import Chunker
from docrag.models import Chunk, Document


class FixedChunker(Chunker):
    name = "fixed"

    def __init__(self, chunk_tokens: int = 512, overlap_tokens: int = 64):
        if overlap_tokens >= chunk_tokens:
            raise ValueError("overlap must be smaller than chunk size")
        self.size = chunk_tokens
        self.overlap = overlap_tokens
        self.enc = tiktoken.get_encoding("cl100k_base")

    def chunk(self, doc: Document) -> list[Chunk]:
        # Concatenate the whole document, remembering where each section starts (in tokens)
        # so every window can still report the heading/page it starts in.
        tokens: list[int] = []
        starts: list[int] = []
        sep = self.enc.encode("\n\n")
        for section in doc.sections:
            starts.append(len(tokens))
            tokens.extend(self.enc.encode(section.text, disallowed_special=()))
            tokens.extend(sep)

        pieces = []
        step = self.size - self.overlap
        for start in range(0, max(len(tokens) - self.overlap, 1), step):
            window = tokens[start : start + self.size]
            section = doc.sections[max(bisect.bisect_right(starts, start) - 1, 0)]
            pieces.append((self.enc.decode(window), section.heading, section.page))
        return self.make_chunks(doc, pieces, with_heading=False)
