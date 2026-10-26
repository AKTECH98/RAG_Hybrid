"""Per-strategy chunk store: data/chunks/<strategy>.jsonl.

This file is the single source of truth for what is indexed. Chroma holds the vectors and
BM25 is rebuilt from here, so both indexes always describe the same set of chunk ids.
"""

import os
from pathlib import Path

from docrag.config import Settings, get_settings
from docrag.models import Chunk


class ChunkStore:
    def __init__(self, strategy: str, settings: Settings | None = None):
        s = settings or get_settings()
        self.strategy = strategy
        self.path: Path = s.chunks_dir / f"{strategy}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks: dict[str, Chunk] | None = None
        self.version = 0  # bumped on every change so BM25 knows when to rebuild

    @property
    def chunks(self) -> dict[str, Chunk]:
        if self._chunks is None:
            self._chunks = {}
            if self.path.exists():
                with self.path.open() as f:
                    for line in f:
                        if line.strip():
                            c = Chunk.model_validate_json(line)
                            self._chunks[c.chunk_id] = c
        return self._chunks

    def all(self) -> list[Chunk]:
        return list(self.chunks.values())

    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def ids_for_doc(self, doc_id: str) -> list[str]:
        return [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]

    def doc_ids(self) -> set[str]:
        return {c.doc_id for c in self.chunks.values()}

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.chunks[c.chunk_id] = c
        self.version += 1

    def delete_doc(self, doc_id: str) -> list[str]:
        ids = self.ids_for_doc(doc_id)
        for cid in ids:
            del self.chunks[cid]
        self.version += 1
        return ids

    def save(self) -> None:
        """Atomic rewrite (write temp file, then rename)."""
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for c in sorted(self.chunks.values(), key=lambda c: (c.doc_id, c.chunk_index)):
                f.write(c.model_dump_json() + "\n")
        os.replace(tmp, self.path)
