"""Central configuration. Every tunable lives here and can be overridden via env / .env."""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ChunkStrategy = Literal["fixed", "recursive", "semantic"]
RetrievalMode = Literal["hybrid", "dense", "sparse"]
RerankerName = Literal["cross_encoder", "llm", "none"]
LLMProvider = Literal["openai", "ollama"]
EmbeddingProvider = Literal["openai", "local", "ollama"]

CHUNK_STRATEGIES: tuple[ChunkStrategy, ...] = ("fixed", "recursive", "semantic")
EMBEDDING_PROVIDERS: tuple[EmbeddingProvider, ...] = ("openai", "local", "ollama")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "local": "BAAI/bge-base-en-v1.5",  # sentence-transformers, runs on this machine
    "ollama": "nomic-embed-text",
}

# Cosine band per embedding model, mapped onto 0..1 relevance: (median top cosine of
# unanswerable golden questions, median of answerable ones). Each model has its own scale:
# bge-base puts nearly everything at 0.70-0.80, so OpenAI's band would clip it all to 1.0.
# Re-derive with: scripts/run_eval.py --retrieval-only --modes dense --rerankers none
COSINE_BANDS: dict[str, tuple[float, float]] = {
    "text-embedding-3-small": (0.49, 0.64),
    "BAAI/bge-base-en-v1.5": (0.70, 0.78),
    "nomic-embed-text": (0.69, 0.78),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_prefix="DOCRAG_", extra="ignore"
    )

    # --- LLM ---
    llm_provider: LLMProvider = "openai"
    generation_model: str = "gpt-4o"
    judge_model: str = "gpt-4o-mini"  # system-side reranking / citation verification
    eval_judge_model: str = "gpt-4o"  # offline eval judge, deliberately a different model
    # Read without the DOCRAG_ prefix so the standard variable name works.
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    # Ollama is called through its native /api endpoints: the OpenAI-compatible /v1 layer
    # can't set the context window (server default 2048 silently truncates RAG prompts)
    # or reliably turn thinking off.
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192
    ollama_think: bool = False
    ollama_timeout_s: float = 900.0
    # Concurrent chat requests from this process. An 8 GB box runs out of memory with two
    # 8k-context requests in flight ("model runner has unexpectedly stopped").
    ollama_max_parallel: int = 1

    # --- Embeddings ---
    embedding_provider: EmbeddingProvider = "openai"
    embedding_model: str | None = None  # None = DEFAULT_EMBEDDING_MODELS[provider]

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
    dedup_threshold: float = 0.95  # cosine; a duplicate must also pass the word overlap check
    dedup_min_jaccard: float = 0.7

    # --- Retrieval ---
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = 20  # per-retriever search depth (never below rerank_candidates)
    rrf_k: int = 60
    # 0.5/0.5 from the retrieval sweep: with 0.7/0.3 and the fused list cut to 20, keyword-only
    # hits can never enter the pool (hybrid == dense-only); 0.5 gave the best pool recall.
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rerank_candidates: int = 20
    rerank_top_n: int = 5
    reranker: RerankerName = "llm"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- API ---
    api_key: str | None = None  # if set, ingest/delete require the X-API-Key header
    max_upload_mb: int = 20

    # --- Generation / confidence ---
    abstain_threshold: float = 0.35
    cosine_band: tuple[float, float] | None = None  # None = COSINE_BANDS[embedding model]
    # Composite confidence weights: (retrieval, citation coverage, completeness)
    confidence_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)

    @property
    def embedding_model_name(self) -> str:
        return self.embedding_model or DEFAULT_EMBEDDING_MODELS[self.embedding_provider]

    @property
    def index_slug(self) -> str:
        """e.g. "local-baai-bge-base-en-v1.5". Vectors from different embedding models
        can't share an index (different spaces, often different dimensions), and dedup and
        semantic chunking depend on the embedder too, so each gets its own index dir."""
        name = re.sub(r"[^a-z0-9.]+", "-", self.embedding_model_name.lower()).strip("-")
        return f"{self.embedding_provider}-{name}"

    @property
    def cosine_band_resolved(self) -> tuple[float, float]:
        return self.cosine_band or COSINE_BANDS.get(self.embedding_model_name, (0.45, 0.65))

    def with_embedder(self, provider: EmbeddingProvider, model: str | None = None) -> "Settings":
        return self.model_copy(update={"embedding_provider": provider, "embedding_model": model})

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index" / self.index_slug

    @property
    def chunks_dir(self) -> Path:
        return self.index_dir / "chunks"

    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
