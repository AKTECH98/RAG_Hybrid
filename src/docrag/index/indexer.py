"""Indexing orchestration: Document -> chunks -> dedup -> embeddings -> Chroma + BM25.

Per strategy, the order is: drop the doc's old chunks from both indexes, embed, dedup,
then write Chroma first and the chunk store second. `check_sync` / `repair` compare the
two and fix any drift (e.g. a crash between the two writes).
"""

import json
from dataclasses import dataclass, field

import numpy as np

from docrag.chunking import get_chunker
from docrag.config import CHUNK_STRATEGIES, ChunkStrategy, Settings, get_settings
from docrag.index.bm25_store import BM25Store
from docrag.index.chunk_store import ChunkStore
from docrag.index.dedup import Duplicate, split_duplicates
from docrag.index.embedder import Embedder
from docrag.index.vector_store import VectorStore, get_chroma_client
from docrag.models import Chunk, Document


@dataclass
class IndexStats:
    strategy: str
    documents: int = 0
    chunks: int = 0
    duplicates: int = 0
    duplicate_examples: list[Duplicate] = field(default_factory=list)


@dataclass
class SyncReport:
    strategy: str
    chunk_store: int
    vector_store: int
    bm25: int
    missing_in_vectors: list[str]
    extra_in_vectors: list[str]

    @property
    def in_sync(self) -> bool:
        return (
            not self.missing_in_vectors
            and not self.extra_in_vectors
            and (self.chunk_store == self.bm25)
        )


class StrategyIndex:
    """Chunk store + vector store + BM25 for one chunking strategy."""

    def __init__(self, strategy: ChunkStrategy, settings: Settings, client=None):
        self.strategy = strategy
        self.chunks = ChunkStore(strategy, settings)
        self.vectors = VectorStore(strategy, client or get_chroma_client(settings))
        self.bm25 = BM25Store(self.chunks)


class Indexer:
    def __init__(self, embedder: Embedder, settings: Settings | None = None, client=None):
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.client = client or get_chroma_client(self.settings)
        self._indexes: dict[str, StrategyIndex] = {}
        self.dedup_log = self.settings.data_dir / "dedup_log.jsonl"

    def index(self, strategy: ChunkStrategy) -> StrategyIndex:
        if strategy not in self._indexes:
            self._indexes[strategy] = StrategyIndex(strategy, self.settings, self.client)
        return self._indexes[strategy]

    # ------------------------------------------------------------------ writes

    def index_documents(
        self, docs: list[Document], strategies: tuple[ChunkStrategy, ...] = CHUNK_STRATEGIES
    ) -> list[IndexStats]:
        return [self._index_strategy(docs, s) for s in strategies]

    def _index_strategy(self, docs: list[Document], strategy: ChunkStrategy) -> IndexStats:
        idx = self.index(strategy)
        chunker = get_chunker(strategy, self.embedder, self.settings)
        stats = IndexStats(strategy=strategy, documents=len(docs))

        # Re-indexing replaces a document: remove its old chunks from both indexes first
        for doc in docs:
            idx.vectors.delete_ids(idx.chunks.delete_doc(doc.doc_id))

        per_doc = [(doc, chunker.chunk(doc)) for doc in docs]
        all_chunks = [c for _, cs in per_doc for c in cs]
        all_vecs = self.embedder.embed([c.embed_text for c in all_chunks])  # one batched pass

        offset = 0
        for _, chunks in per_doc:
            vecs = all_vecs[offset : offset + len(chunks)]
            offset += len(chunks)
            keep, dups = split_duplicates(
                chunks, vecs, idx.vectors, self.settings.dedup_threshold, idx.chunks.chunks
            )
            kept = [chunks[i] for i in keep]
            if kept:
                idx.vectors.upsert(kept, vecs[keep])
                idx.chunks.add(kept)
            stats.chunks += len(kept)
            stats.duplicates += len(dups)
            stats.duplicate_examples.extend(dups)
            self._log_duplicates(dups)

        idx.chunks.save()
        return stats

    def remove_document(self, doc_id: str) -> int:
        removed = 0
        for strategy in CHUNK_STRATEGIES:
            idx = self.index(strategy)
            ids = idx.chunks.delete_doc(doc_id)
            idx.vectors.delete_ids(ids)
            idx.chunks.save()
            removed += len(ids)
        return removed

    def reset(self, strategy: ChunkStrategy) -> None:
        idx = self.index(strategy)
        self.client.delete_collection(f"chunks_{strategy}")
        idx.chunks.path.unlink(missing_ok=True)
        self._indexes.pop(strategy)

    def _log_duplicates(self, dups: list[Duplicate]) -> None:
        if not dups:
            return
        self.dedup_log.parent.mkdir(parents=True, exist_ok=True)
        with self.dedup_log.open("a") as f:
            for d in dups:
                f.write(json.dumps(d.to_dict()) + "\n")

    # ------------------------------------------------------------------ consistency

    def check_sync(self, strategy: ChunkStrategy) -> SyncReport:
        idx = self.index(strategy)
        store_ids = set(idx.chunks.chunks)
        vector_ids = idx.vectors.all_ids()
        return SyncReport(
            strategy=strategy,
            chunk_store=len(store_ids),
            vector_store=len(vector_ids),
            bm25=len(idx.bm25.ids),
            missing_in_vectors=sorted(store_ids - vector_ids),
            extra_in_vectors=sorted(vector_ids - store_ids),
        )

    def repair(self, strategy: ChunkStrategy) -> SyncReport:
        """Make Chroma match the chunk store (the source of truth)."""
        idx = self.index(strategy)
        report = self.check_sync(strategy)
        idx.vectors.delete_ids(report.extra_in_vectors)
        missing: list[Chunk] = [idx.chunks.chunks[cid] for cid in report.missing_in_vectors]
        if missing:
            vecs = self.embedder.embed([c.embed_text for c in missing])
            idx.vectors.upsert(missing, np.asarray(vecs))
        return self.check_sync(strategy)
