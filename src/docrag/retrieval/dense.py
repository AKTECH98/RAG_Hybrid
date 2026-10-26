"""Dense retrieval: embed the query, nearest neighbours by cosine similarity in Chroma."""

from docrag.index.embedder import Embedder
from docrag.index.vector_store import VectorStore


def dense_search(
    vectors: VectorStore, embedder: Embedder, query: str, k: int
) -> list[tuple[str, float]]:
    """Top-k (chunk_id, cosine similarity), best first."""
    return vectors.query(embedder.embed_query(query), k)
