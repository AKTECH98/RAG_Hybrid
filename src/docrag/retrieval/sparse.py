"""Sparse retrieval: BM25 over the same chunks, for exact identifiers, config keys, codes."""

from docrag.index.bm25_store import BM25Store


def sparse_search(bm25: BM25Store, query: str, k: int) -> list[tuple[str, float]]:
    """Top-k (chunk_id, BM25 score) with a positive score, best first."""
    return bm25.search(query, k)
