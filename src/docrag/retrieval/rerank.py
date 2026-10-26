"""Second-pass rerankers: score each (question, chunk) pair directly.

First-stage retrievers compare a query embedding with chunk embeddings computed
independently. A reranker reads the question and the chunk *together*, which is much
more precise, but too slow to run over the whole corpus, so it only sees the top
candidates. Every reranker returns a 0..1 relevance per chunk; Phase 3 uses those
scores as the retrieval-confidence signal.
"""

import math
import threading
from functools import lru_cache
from typing import Protocol

from pydantic import BaseModel, Field

from docrag.config import RerankerName, Settings, get_settings
from docrag.llm import LLM
from docrag.models import Chunk

MAX_PASSAGE_CHARS = 1500


class Reranker(Protocol):
    name: str

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        """0..1 relevance for each chunk, in input order."""
        ...


class CrossEncoderReranker:
    """Local MS MARCO cross-encoder (MiniLM, ~22M params). Free, ~100ms for 20 pairs."""

    name = "cross_encoder"

    def __init__(self, model_name: str):
        self.model = _load_cross_encoder(model_name)
        self._lock = threading.Lock()  # one torch model shared by concurrent requests

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        with self._lock:
            logits = self.model.predict([(query, c.embed_text) for c in chunks])
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


@lru_cache
def _load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:  # installed via the default "rerank" dependency group
        raise RuntimeError(
            "cross_encoder reranker needs sentence-transformers: `uv sync --group rerank`"
        ) from e
    return CrossEncoder(model_name)


class _PassageScore(BaseModel):
    id: int
    score: int = Field(ge=0, le=10)


class _Scores(BaseModel):
    scores: list[_PassageScore]


LLM_RERANK_SYSTEM = """You are a search relevance judge for technical documentation.
Given a question and numbered passages, rate how useful each passage is for answering \
the question:
10 = directly answers it, 7 = contains a key part of the answer,
4 = related topic but does not help answer, 0 = unrelated.
Judge only what the passage itself says. Score every passage id exactly once."""


class LLMReranker:
    """LLM-as-judge: one structured-output call scores all candidates at once."""

    name = "llm"

    def __init__(self, llm: LLM, model: str):
        self.llm = llm
        self.model = model

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        passages = "\n\n".join(
            f"[{i}] ({c.citation_label()})\n{c.text[:MAX_PASSAGE_CHARS]}"
            for i, c in enumerate(chunks, start=1)
        )
        result = self.llm.structured(
            self.model,
            LLM_RERANK_SYSTEM,
            f"Question: {query}\n\nPassages:\n\n{passages}",
            _Scores,
        )
        by_id = {s.id: s.score for s in result.scores}
        return [by_id.get(i, 0) / 10 for i in range(1, len(chunks) + 1)]


def get_reranker(
    name: RerankerName, llm: LLM | None = None, settings: Settings | None = None
) -> Reranker | None:
    s = settings or get_settings()
    if name == "none":
        return None
    if name == "cross_encoder":
        return CrossEncoderReranker(s.cross_encoder_model)
    if name == "llm":
        if llm is None:
            raise ValueError("llm reranker needs an LLM client")
        return LLMReranker(llm, s.judge_model)
    raise ValueError(f"Unknown reranker: {name}")
