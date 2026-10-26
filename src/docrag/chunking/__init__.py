from docrag.chunking.base import Chunker
from docrag.chunking.fixed import FixedChunker
from docrag.chunking.recursive import RecursiveChunker
from docrag.chunking.semantic import SemanticChunker
from docrag.config import ChunkStrategy, Settings, get_settings
from docrag.index.embedder import Embedder


def get_chunker(
    strategy: ChunkStrategy, embedder: Embedder | None = None, settings: Settings | None = None
) -> Chunker:
    s = settings or get_settings()
    if strategy == "fixed":
        return FixedChunker(s.fixed_chunk_tokens, s.fixed_overlap_tokens)
    if strategy == "recursive":
        return RecursiveChunker(s.recursive_chunk_chars, s.recursive_overlap_chars)
    if strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking needs an embedder")
        return SemanticChunker(
            embedder, s.semantic_breakpoint_percentile, s.semantic_min_chars, s.semantic_max_chars
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")


__all__ = ["Chunker", "FixedChunker", "RecursiveChunker", "SemanticChunker", "get_chunker"]
