"""ChromaDB wrapper: one cosine collection per chunking strategy (chunks_fixed, ...).

Chroma's embedded (Rust) client can deadlock when several Python threads call into it at
once (seen with 8 eval workers: every thread parked inside chromadb_rust_bindings). All
calls therefore go through one process-wide lock. Queries take milliseconds, so the cost
is negligible next to the LLM calls around them.
"""

import threading
import time

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from docrag.config import Settings, get_settings
from docrag.models import Chunk

_clients: dict[str, chromadb.ClientAPI] = {}
_lock = threading.RLock()


def get_chroma_client(settings: Settings | None = None) -> chromadb.ClientAPI:
    s = settings or get_settings()
    chroma_settings = ChromaSettings(anonymized_telemetry=False)
    with _lock:
        if s.chroma_mode == "http":
            key = f"http://{s.chroma_host}:{s.chroma_port}"
            if key not in _clients:
                _clients[key] = _connect_http(s, chroma_settings)
        else:
            key = str(s.chroma_dir)
            if key not in _clients:
                _clients[key] = chromadb.PersistentClient(path=key, settings=chroma_settings)
        return _clients[key]


def _connect_http(s: Settings, chroma_settings: ChromaSettings, wait_s: float = 60.0):
    """Connect to a Chroma server, waiting for it to come up (docker-compose start order)."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            client = chromadb.HttpClient(
                host=s.chroma_host, port=s.chroma_port, settings=chroma_settings
            )
            client.heartbeat()
            return client
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def collection_name(strategy: str, settings: Settings) -> str:
    """Embedded mode keeps one Chroma directory per embedder; a shared server needs the
    embedder in the collection name instead (vectors from different models can't mix)."""
    if settings.chroma_mode == "http":
        return f"{settings.index_slug}.chunks_{strategy}"
    return f"chunks_{strategy}"


class VectorStore:
    def __init__(self, strategy: str, client: chromadb.ClientAPI, name: str | None = None):
        self.strategy = strategy
        self.name = name or f"chunks_{strategy}"
        with _lock:
            self.collection = client.get_or_create_collection(
                self.name,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,  # we always pass our own vectors
            )

    def count(self) -> int:
        with _lock:
            return self.collection.count()

    def all_ids(self) -> set[str]:
        with _lock:
            return set(self.collection.get(include=[])["ids"])

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with _lock:
            for i in range(0, len(chunks), 1000):
                batch = chunks[i : i + 1000]
                self.collection.upsert(
                    ids=[c.chunk_id for c in batch],
                    embeddings=vectors[i : i + 1000],
                    documents=[c.text for c in batch],
                    metadatas=[c.metadata() for c in batch],
                )

    def delete_ids(self, ids: list[str]) -> None:
        if ids:
            with _lock:
                self.collection.delete(ids=ids)

    def query(
        self, vector: np.ndarray, k: int, where: dict | None = None
    ) -> list[tuple[str, float]]:
        """Top-k (chunk_id, cosine similarity), best first."""
        with _lock:
            n = self.collection.count()
            if n == 0:
                return []
            res = self.collection.query(
                query_embeddings=vector.reshape(1, -1),
                n_results=min(k, n),
                where=where,
                include=["distances"],
            )
        return [(cid, 1.0 - d) for cid, d in zip(res["ids"][0], res["distances"][0], strict=True)]

    def nearest(
        self, vectors: np.ndarray, exclude_doc_id: str | None = None, n: int = 3
    ) -> list[list[tuple[str, float]]]:
        """Up to n nearest stored neighbours (id, similarity) per vector, best first."""
        if len(vectors) == 0:
            return []
        with _lock:
            count = self.collection.count()
            if count == 0:
                return [[] for _ in vectors]
            where = {"doc_id": {"$ne": exclude_doc_id}} if exclude_doc_id else None
            res = self.collection.query(
                query_embeddings=vectors,
                n_results=min(n, count),
                where=where,
                include=["distances"],
            )
        return [
            [(cid, 1.0 - d) for cid, d in zip(ids, dists, strict=True)]
            for ids, dists in zip(res["ids"], res["distances"], strict=True)
        ]
