"""Central configuration. Every tunable lives here and can be overridden via env / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ChunkStrategy = Literal["fixed", "recursive", "semantic"]
RetrievalMode = Literal["hybrid", "dense", "sparse"]
RerankerName = Literal["cross_encoder", "llm", "none"]

CHUNK_STRATEGIES: tuple[ChunkStrategy, ...] = ("fixed", "recursive", "semantic")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_prefix="DOCRAG_", extra="ignore"
    )

    # --- OpenAI ---
    # Read without the DOCRAG_ prefix so the standard variable name works.
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o"
    judge_model: str = "gpt-4o-mini"

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    corpus_dir: Path = PROJECT_ROOT / "corpus"

    # --- Vector store ---
    chroma_mode: Literal["persistent", "http"] = "persistent"
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # --- Chunking ---
    default_strategy: ChunkStrategy = "recursive"
    fixed_chunk_tokens: int = 512
    fixed_overlap_tokens: int = 64
    recursive_chunk_chars: int = 1500
    recursive_overlap_chars: int = 200
    semantic_breakpoint_percentile: float = 95.0
    semantic_min_chars: int = 200
    semantic_max_chars: int = 2500

    # --- Dedup ---
    dedup_threshold: float = 0.95

    # --- Retrieval ---
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = 10
    rrf_k: int = 60
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rerank_candidates: int = 20
    rerank_top_n: int = 5
    reranker: RerankerName = "llm"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Generation / confidence ---
    abstain_threshold: float = 0.35
    # Composite confidence weights: (retrieval, citation coverage, completeness)
    confidence_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
