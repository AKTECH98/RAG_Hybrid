"""Hybrid retrieval pipeline: dense + sparse -> weighted RRF -> rerank -> top-n.

Every RetrievedChunk keeps its rank and score from each stage, so the dashboard and the
eval can show exactly where a chunk came from and how each stage moved it.
"""

import sys
import threading
import time

from pydantic import BaseModel

from docrag.config import ChunkStrategy, RerankerName, RetrievalMode, Settings, get_settings
from docrag.index.embedder import Embedder
from docrag.index.indexer import Indexer
from docrag.llm import LLM
from docrag.models import RetrievedChunk
from docrag.retrieval.dense import dense_search
from docrag.retrieval.fusion import weighted_rrf
from docrag.retrieval.rerank import Reranker, get_reranker
from docrag.retrieval.sparse import sparse_search


class RetrievalResult(BaseModel):
    query: str
    mode: RetrievalMode
    strategy: ChunkStrategy
    reranker: RerankerName
    candidates: list[RetrievedChunk]  # fused pool sent to the reranker, in fused order
    results: list[RetrievedChunk]  # final top-n, best first
    timings_ms: dict[str, float]


class Retriever:
    def __init__(
        self,
        indexer: Indexer,
        embedder: Embedder,
        llm: LLM | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.indexer = indexer
        self.embedder = embedder
        self.llm = llm
        self._rerankers: dict[str, Reranker | None] = {}
        self._reranker_lock = threading.Lock()

    def _reranker(self, name: RerankerName) -> Reranker | None:
        # Locked: concurrent first calls would each load the cross-encoder, and parallel
        # torch model loads crash the process.
        with self._reranker_lock:
            if name not in self._rerankers:
                self._rerankers[name] = get_reranker(name, self.llm, self.settings)
            return self._rerankers[name]

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode | None = None,
        strategy: ChunkStrategy | None = None,
        reranker: RerankerName | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
        dense_weight: float | None = None,
    ) -> RetrievalResult:
        s = self.settings
        mode = mode or s.retrieval_mode
        strategy = strategy or s.default_strategy
        reranker = reranker or s.reranker
        top_n = top_n or s.rerank_top_n
        pool = max(s.rerank_candidates, top_n)
        # Every retriever searches at least as deep as the reranker pool, in every mode, so
        # hybrid never sees less of the dense ranking than dense-only does. (Eval finding:
        # with 10 per retriever, recursive-chunk evidence at dense ranks 11-20 was lost.)
        fetch = max(top_k or s.top_k, pool)
        idx = self.indexer.index(strategy)
        timings: dict[str, float] = {}

        t = time.perf_counter()
        dense = dense_search(idx.vectors, self.embedder, query, fetch) if mode != "sparse" else []
        timings["dense"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        sparse = sparse_search(idx.bm25, query, fetch) if mode != "dense" else []
        timings["sparse"] = (time.perf_counter() - t) * 1000

        # Note: the fused list is cut to `pool`, so with weights w_d > w_s a sparse-only hit
        # can only enter the pool if w_s/(k + r_s) beats the weakest dense hit kept.
        w_d = s.dense_weight if dense_weight is None else dense_weight
        w_s = s.sparse_weight if dense_weight is None else 1.0 - dense_weight
        weights = {"dense": w_d, "sparse": w_s} if mode == "hybrid" else {mode: 1.0}
        fused = weighted_rrf(
            {"dense": [cid for cid, _ in dense], "sparse": [cid for cid, _ in sparse]},
            weights,
            s.rrf_k,
        )[:pool]

        dense_info = {cid: (r, sc) for r, (cid, sc) in enumerate(dense, start=1)}
        sparse_info = {cid: (r, sc) for r, (cid, sc) in enumerate(sparse, start=1)}
        candidates: list[RetrievedChunk] = []
        for cid, score in fused:
            chunk = idx.chunks.get(cid)
            if chunk is None:  # vector index ahead of the chunk store; `repair` fixes it
                continue
            d, sp = dense_info.get(cid), sparse_info.get(cid)
            candidates.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_rank=d[0] if d else None,
                    dense_score=d[1] if d else None,
                    sparse_rank=sp[0] if sp else None,
                    sparse_score=sp[1] if sp else None,
                    fused_rank=len(candidates) + 1,
                    fused_score=score,
                )
            )

        t = time.perf_counter()
        ranked = candidates
        rr = self._reranker(reranker)
        if rr is not None and candidates:
            for c, score in zip(
                candidates, rr.score(query, [c.chunk for c in candidates]), strict=True
            ):
                c.rerank_score = score
            # stable sort: equal rerank scores keep their fused order
            ranked = sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))
        timings["rerank"] = (time.perf_counter() - t) * 1000

        results = [c.model_copy() for c in ranked[:top_n]]
        for rank, c in enumerate(results, start=1):
            c.final_rank = rank
        return RetrievalResult(
            query=query,
            mode=mode,
            strategy=strategy,
            reranker=reranker,
            candidates=candidates,
            results=results,
            timings_ms={k: round(v, 1) for k, v in timings.items()},
        )


def build_retriever(settings: Settings | None = None) -> Retriever:
    """Wire up the real components (OpenAI embedder/LLM, persistent indexes)."""
    from docrag.index.embedder import get_embedder
    from docrag.llm import get_llm

    s = settings or get_settings()
    embedder = get_embedder(s)
    return Retriever(Indexer(embedder, s), embedder, get_llm(s), s)


if __name__ == "__main__":
    from docrag.cli import main

    main(["retrieve", *sys.argv[1:]])
