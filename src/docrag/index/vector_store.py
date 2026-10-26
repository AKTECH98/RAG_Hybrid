"""ChromaDB wrapper: one cosine collection per chunking strategy (chunks_fixed, ...)."""

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from docrag.config import Settings, get_settings
from docrag.models import Chunk

_clients: dict[str, chromadb.ClientAPI] = {}


def get_chroma_client(settings: Settings | None = None) -> chromadb.ClientAPI:
    s = settings or get_settings()
    chroma_settings = ChromaSettings(anonymized_telemetry=False)
    if s.chroma_mode == "http":
        key = f"http://{s.chroma_host}:{s.chroma_port}"
        if key not in _clients:
            _clients[key] = chromadb.HttpClient(
                host=s.chroma_host, port=s.chroma_port, settings=chroma_settings
            )
    else:
        key = str(s.chroma_dir)
        if key not in _clients:
            _clients[key] = chromadb.PersistentClient(path=key, settings=chroma_settings)
    return _clients[key]


class VectorStore:
    def __init__(self, strategy: str, client: chromadb.ClientAPI):
        self.strategy = strategy
        self.collection = client.get_or_create_collection(
            f"chunks_{strategy}",
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,  # we always pass our own vectors
        )

    def count(self) -> int:
        return self.collection.count()

    def all_ids(self) -> set[str]:
        return set(self.collection.get(include=[])["ids"])

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
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
            self.collection.delete(ids=ids)

    def query(
        self, vector: np.ndarray, k: int, where: dict | None = None
    ) -> list[tuple[str, float]]:
        """Top-k (chunk_id, cosine similarity), best first."""
        n = self.count()
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
        self, vectors: np.ndarray, exclude_doc_id: str | None = None
    ) -> list[tuple[str, float] | None]:
        """Nearest stored neighbour (id, similarity) for each vector, or None if empty."""
        if self.count() == 0 or len(vectors) == 0:
            return [None] * len(vectors)
        where = {"doc_id": {"$ne": exclude_doc_id}} if exclude_doc_id else None
        res = self.collection.query(
            query_embeddings=vectors, n_results=1, where=where, include=["distances"]
        )
        return [
            (ids[0], 1.0 - dists[0]) if ids else None
            for ids, dists in zip(res["ids"], res["distances"], strict=True)
        ]
