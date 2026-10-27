"""Question -> retrieval -> grounded answer -> citation verification -> confidence."""

import time
from concurrent.futures import ThreadPoolExecutor

from docrag.config import ChunkStrategy, RerankerName, RetrievalMode, Settings, get_settings
from docrag.generation.abstain import explain_gaps, suggest_documents
from docrag.generation.citations import attach_checks, parse_claims, verify_claims
from docrag.generation.confidence import (
    calibrated_relevance,
    citation_coverage,
    completeness,
    composite_confidence,
    retrieval_confidence,
)
from docrag.generation.models import Answer, CitationCheck, Claim, Source
from docrag.generation.prompt import INSUFFICIENT_MARKER, SYSTEM_PROMPT, build_user_prompt
from docrag.llm import LLM
from docrag.retrieval.pipeline import RetrievalResult, Retriever


class Generator:
    def __init__(self, retriever: Retriever, llm: LLM, settings: Settings | None = None):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings or get_settings()

    def ask(
        self,
        question: str,
        *,
        mode: RetrievalMode | None = None,
        strategy: ChunkStrategy | None = None,
        reranker: RerankerName | None = None,
        verify: bool = True,
    ) -> Answer:
        t = time.perf_counter()
        res = self.retriever.retrieve(question, mode=mode, strategy=strategy, reranker=reranker)
        answer = self.answer_from(question, res, verify=verify)
        answer.timings_ms = {"retrieval": round((time.perf_counter() - t) * 1000, 1)} | (
            answer.timings_ms
        )
        return answer

    def sources_for(self, res: RetrievalResult) -> list[Source]:
        """The numbered context blocks [1]..[n] for a retrieval result."""
        band = self.settings.cosine_band_resolved
        return [
            Source.from_retrieved(n, rc, calibrated_relevance(rc, band))
            for n, rc in enumerate(res.results, start=1)
        ]

    def answer_from(self, question: str, res: RetrievalResult, verify: bool = True) -> Answer:
        s = self.settings
        sources = self.sources_for(res)
        retrieval = retrieval_confidence([src.relevance for src in sources])
        meta = {
            "mode": res.mode,
            "strategy": res.strategy,
            "reranker": res.reranker,
            "candidates": len(res.candidates),
            "timings_ms": res.timings_ms,
        }

        if retrieval < s.abstain_threshold:
            return self._abstain(question, sources, retrieval, meta)

        t = time.perf_counter()
        text = self.llm.generate(
            s.generation_model, SYSTEM_PROMPT, build_user_prompt(question, sources)
        ).strip()
        timings = {"generation": round((time.perf_counter() - t) * 1000, 1)}
        if text.startswith(INSUFFICIENT_MARKER):
            note = text[len(INSUFFICIENT_MARKER) :].lstrip(": ").strip()
            return self._abstain(question, sources, retrieval, meta, model_note=note)

        answer = self.check_answer(question, text, sources, retrieval, verify=verify)
        answer.retrieval = meta
        answer.timings_ms = timings | answer.timings_ms
        return answer

    def check_answer(
        self,
        question: str,
        text: str,
        sources: list[Source],
        retrieval: float,
        verify: bool = True,
    ) -> Answer:
        """Parse, verify and score an answer. Public so any answer text can be audited,
        e.g. a deliberately hallucinated one in the demo."""
        s = self.settings
        claims = parse_claims(text, len(sources))
        checks: list[CitationCheck] = []
        complete, parts = 0.0, []
        t = time.perf_counter()
        if verify:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_checks = pool.submit(
                    verify_claims, self.llm, s.judge_model, question, claims, sources
                )
                f_complete = pool.submit(completeness, self.llm, s.judge_model, question, text)
                checks = f_checks.result()
                complete, parts = f_complete.result()
            attach_checks(claims, checks)
        else:  # unverified: trust the markers (coverage is then an upper bound)
            for c in claims:
                c.checks = [
                    CitationCheck(
                        claim_id=c.id, source=n, verdict="supported", reason="not verified"
                    )
                    for n in c.citations
                ]
            complete = 1.0
        timings = {"verification": round((time.perf_counter() - t) * 1000, 1)}

        return Answer(
            question=question,
            status="answered",
            answer=text,
            claims=claims,
            unsupported_citations=_unsupported(claims, checks),
            sources=sources,
            confidence=composite_confidence(
                retrieval, citation_coverage(claims), complete, s.confidence_weights, parts
            ),
            timings_ms=timings,
        )

    def _abstain(
        self,
        question: str,
        sources: list[Source],
        retrieval: float,
        meta: dict,
        model_note: str | None = None,
    ) -> Answer:
        t = time.perf_counter()
        gaps = explain_gaps(self.llm, self.settings.judge_model, question, sources)
        found = gaps.found or ([model_note] if model_note else [])
        return Answer(
            question=question,
            status="insufficient_context",
            found=found,
            missing=gaps.missing,
            suggested_documents=suggest_documents(sources),
            sources=sources,
            confidence=composite_confidence(retrieval, 0.0, 0.0, self.settings.confidence_weights),
            retrieval=meta,
            timings_ms={"gap_analysis": round((time.perf_counter() - t) * 1000, 1)},
        )


def _unsupported(claims: list[Claim], checks: list[CitationCheck]) -> list[CitationCheck]:
    flagged = [c for c in checks if c.verdict == "unsupported"]
    for claim in claims:
        flagged.extend(
            CitationCheck(
                claim_id=claim.id,
                source=n,
                verdict="unsupported",
                reason=f"cites [{n}], but there is no source [{n}]",
            )
            for n in claim.invalid_citations
        )
    return flagged


def build_generator(settings: Settings | None = None) -> Generator:
    from docrag.llm import get_llm
    from docrag.retrieval.pipeline import build_retriever

    s = settings or get_settings()
    return Generator(build_retriever(s), get_llm(s), s)
