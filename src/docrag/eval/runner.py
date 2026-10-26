"""Run the golden set through one or more pipeline configurations and score every case."""

import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from docrag.config import ChunkStrategy, EmbeddingProvider, RerankerName, RetrievalMode
from docrag.eval.dataset import GoldenCase
from docrag.eval.metrics import CaseResult, retrieval_metrics, score_case
from docrag.generation.confidence import calibrated_relevance, retrieval_confidence
from docrag.generation.generator import Generator
from docrag.llm import LLM
from docrag.retrieval.pipeline import RetrievalResult


class EvalConfig(BaseModel):
    strategy: ChunkStrategy
    mode: RetrievalMode
    reranker: RerankerName
    dense_weight: float | None = None  # hybrid only; None = configured default
    embedder: EmbeddingProvider | None = None  # None = configured default

    @property
    def name(self) -> str:
        weight = f"@{self.dense_weight:g}" if self.dense_weight is not None else ""
        prefix = f"{self.embedder}:" if self.embedder else ""
        return f"{prefix}{self.strategy}/{self.mode}{weight}/{self.reranker}"


class ConfigReport(BaseModel):
    config: EvalConfig
    summary: dict[str, float | int | None]
    by_type: dict[str, dict[str, float | int | None]]
    results: list[CaseResult]


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def separation_auc(answerable: list[float], unanswerable: list[float]) -> float | None:
    """P(top score of a random answerable case > that of a random unanswerable one).

    1.0 means a single threshold on the top retrieval score perfectly separates questions
    the corpus answers from ones it doesn't, i.e. how well the "I don't know" gate can work.
    """
    if not answerable or not unanswerable:
        return None
    wins = sum(1.0 if a > u else 0.5 if a == u else 0.0 for a in answerable for u in unanswerable)
    return round(wins / (len(answerable) * len(unanswerable)), 3)


def summarize(results: list[CaseResult]) -> dict[str, float | int | None]:
    answerable = [r for r in results if r.type != "no_answer"]
    unanswerable = [r for r in results if r.type == "no_answer"]
    latencies = [r.latency_ms for r in results if not r.error]
    return {
        "n": len(results),
        "correctness": _mean([r.correctness for r in results]),
        "faithfulness": _mean([r.faithfulness for r in results]),
        "citation_accuracy": _mean([r.citation_accuracy for r in results]),
        "recall_at_k": _mean([r.recall_at_k for r in answerable]),
        "hit_at_k": _mean([r.hit_at_k for r in answerable]),
        "mrr": _mean([r.mrr for r in answerable]),
        "pool_recall": _mean([r.pool_recall for r in answerable]),
        "separation_auc": separation_auc(
            [r.top_relevance for r in answerable if r.top_relevance is not None],
            [r.top_relevance for r in unanswerable if r.top_relevance is not None],
        ),
        # declined correctly: abstained, or answered that the docs don't cover it
        "no_answer_accuracy": _mean([r.correctness for r in unanswerable]),
        "false_abstain_rate": _mean(
            [float(r.abstained) for r in answerable if r.status != "retrieval_only"]
        ),
        "latency_p50_ms": round(statistics.median(latencies)) if latencies else None,
        "errors": sum(1 for r in results if r.error),
    }


def _top_scores(res: RetrievalResult, band: tuple[float, float]) -> dict[str, float | None]:
    top = res.results[0] if res.results else None
    relevances = [calibrated_relevance(c, band) for c in res.results]
    return {
        "retrieval_confidence": round(retrieval_confidence(relevances), 4),
        "top_relevance": round(calibrated_relevance(top, band), 4) if top else None,
        "top_cosine": round(max((c.dense_score or 0.0) for c in res.candidates), 4)
        if any(c.dense_score is not None for c in res.candidates)
        else None,
    }


def run_config(
    generator: Generator,
    cases: list[GoldenCase],
    cfg: EvalConfig,
    judge: LLM,
    judge_model: str,
    workers: int = 8,
    progress: Callable[[str], None] | None = None,
    retrieval_only: bool = False,
) -> ConfigReport:
    retriever = generator.retriever
    band = generator.settings.cosine_band_resolved
    kwargs = {
        "strategy": cfg.strategy,
        "mode": cfg.mode,
        "reranker": cfg.reranker,
        "dense_weight": cfg.dense_weight,
    }
    # build BM25, load the chunk store and the reranker model once, before the threads start
    # (keeps one-off load time out of the first cases' latency)
    retriever.retrieve("warm up", **kwargs)

    def one(case: GoldenCase) -> CaseResult:
        t = time.perf_counter()
        try:
            res = retriever.retrieve(case.question, **kwargs)
            final = [c.chunk for c in res.results]
            pool = [c.chunk for c in res.candidates]
            if retrieval_only:
                out = CaseResult(
                    case_id=case.id,
                    type=case.type,
                    question=case.question,
                    status="retrieval_only",
                    answer=None,
                    abstained=False,
                    latency_ms=round((time.perf_counter() - t) * 1000, 1),
                    sources=[c.citation_label() for c in final],
                    **retrieval_metrics(case, final, pool),
                    **_top_scores(res, band),
                )
            else:
                answer = generator.answer_from(case.question, res)
                latency = (time.perf_counter() - t) * 1000
                out = score_case(judge, judge_model, case, answer, final, pool, latency)
                out = out.model_copy(update=_top_scores(res, band))
        except Exception as e:  # keep going; the report lists errors
            out = CaseResult(
                case_id=case.id,
                type=case.type,
                question=case.question,
                status="error",
                answer=None,
                abstained=False,
                error=f"{type(e).__name__}: {e}",
            )
        if progress:
            progress(f"{cfg.name} {case.id} {out.correctness_reason[:60] or out.error or ''}")
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, cases))

    types = sorted({r.type for r in results})
    return ConfigReport(
        config=cfg,
        summary=summarize(results),
        by_type={t: summarize([r for r in results if r.type == t]) for t in types},
        results=results,
    )
