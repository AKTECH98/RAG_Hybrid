"""Semantic: split where consecutive sentences stop talking about the same thing.

Each sentence (code blocks stay whole) is embedded; at every gap between sentences the
mean embedding of the few sentences before it is compared with the few after it. A chunk
boundary goes wherever that cosine distance is in the top (100 - percentile)% for the
document, subject to min/max size guards.
"""

import re

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docrag.chunking.base import Chunker
from docrag.chunking.recursive import SEPARATORS
from docrag.index.embedder import Embedder
from docrag.models import Chunk, Document

SENTENCE_END_RE = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9`*\"(\[])")
FENCE_BLOCK_RE = re.compile(r"(```.*?```)", re.S)


def split_units(text: str) -> list[str]:
    """Split text into sentence-like units, keeping fenced code blocks intact."""
    units: list[str] = []
    for part in FENCE_BLOCK_RE.split(text):
        if part.startswith("```"):
            units.append(part)
            continue
        for para in re.split(r"\n\s*\n", part):
            para = para.strip("\n")
            if not para.strip():
                continue
            # list items / table rows are natural units; prose gets sentence-split
            lines = para.splitlines()
            if len(lines) > 1 and all(re.match(r"\s*([-*•]|\d+\.|\|)", ln) for ln in lines[1:]):
                units.extend(lines)
            else:
                units.extend(u for u in SENTENCE_END_RE.split(para) if u.strip())
    return units


class SemanticChunker(Chunker):
    name = "semantic"

    def __init__(
        self,
        embedder: Embedder,
        breakpoint_percentile: float = 95.0,
        min_chars: int = 200,
        max_chars: int = 2500,
        window: int = 1,
    ):
        self.embedder = embedder
        self.percentile = breakpoint_percentile
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.window = window
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars, chunk_overlap=0, separators=SEPARATORS
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        # (unit text, section) pairs across the whole document; oversized units (long code
        # blocks, run-on paragraphs) are pre-split so no chunk can exceed max_chars
        units = [
            (piece, s)
            for s in doc.sections
            for u in split_units(s.text)
            for piece in (self.splitter.split_text(u) if len(u) > self.max_chars else [u])
        ]
        if not units:
            return []
        if len(units) == 1:
            return self.make_chunks(doc, [(units[0][0], units[0][1].heading, units[0][1].page)])

        # distances[i] scores the gap between unit i and i+1: compare the mean embedding of
        # the `window`+1 units before the gap with the units after it (no overlap, so the
        # peak lands exactly on the boundary)
        vecs = self.embedder.embed([u for u, _ in units])
        w = self.window
        distances = np.empty(len(units) - 1, dtype=np.float32)
        for i in range(len(units) - 1):
            left = vecs[max(0, i - w) : i + 1].mean(axis=0)
            right = vecs[i + 1 : i + 2 + w].mean(axis=0)
            denom = np.linalg.norm(left) * np.linalg.norm(right) or 1.0
            distances[i] = 1.0 - float(left @ right) / denom
        threshold = np.percentile(distances, self.percentile)

        pieces = []
        current: list[str] = []
        first = units[0][1]
        size = 0
        for i, (text, section) in enumerate(units):
            if not current:
                first = section
            current.append(text)
            size += len(text) + 1
            at_end = i == len(units) - 1
            # gaps next to a boundary also score high; only break at the local peak
            topic_shift = (
                not at_end
                and distances[i] >= threshold
                and distances[i] >= distances[max(i - 1, 0)]
                and distances[i] >= distances[min(i + 1, len(distances) - 1)]
            )
            too_big = not at_end and size + len(units[i + 1][0]) > self.max_chars
            if at_end or (size >= self.min_chars and (topic_shift or too_big)):
                text = "\n".join(current)
                if at_end and pieces and size < self.min_chars:
                    prev_text, heading, page = pieces[-1]  # fold a tiny tail into its neighbour
                    pieces[-1] = (f"{prev_text}\n{text}", heading, page)
                else:
                    pieces.append((text, first.heading, first.page))
                current, size = [], 0
        return self.make_chunks(doc, pieces)
