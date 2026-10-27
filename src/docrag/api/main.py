"""FastAPI service. Interactive docs at /docs.

uv run uvicorn docrag.api.main:app --reload
"""

import logging
import re
import time
from collections import Counter
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from docrag.api.deps import Services, build_services, get_services, require_api_key
from docrag.api.schemas import (
    AskRequest,
    AskResponse,
    Candidate,
    DocumentInfo,
    Health,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    StrategyStats,
    VerifyRequest,
)
from docrag.config import CHUNK_STRATEGIES, ChunkStrategy
from docrag.generation.confidence import retrieval_confidence
from docrag.ingest.loaders import supported
from docrag.models import Document

log = logging.getLogger("docrag.api")

Svc = Annotated[Services, Depends(get_services)]
ALL_STRATEGIES = ",".join(CHUNK_STRATEGIES)


def create_app(services_factory: Callable[[], Services] = build_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = services_factory()
        app.state.services = svc
        try:  # load the reranker and touch the indexes now, not on the first request
            svc.retriever.retrieve("warm up")
        except Exception as e:  # e.g. the embedding server is down; requests will say so
            log.warning("Warm-up retrieval failed: %s", e)
        yield

    app = FastAPI(
        title="docrag",
        version="0.1.0",
        description="Question answering over technical docs: hybrid retrieval, reranking, "
        "cited answers with verified citations, and confidence scores.",
        lifespan=lifespan,
    )

    @app.exception_handler(RuntimeError)
    async def upstream_error(request: Request, exc: RuntimeError) -> JSONResponse:
        # The LLM and embedding clients raise RuntimeError when the model server fails
        log.exception("Request failed")
        return JSONResponse(status_code=502, content={"detail": f"Model call failed: {exc}"})

    @app.get("/healthz", response_model=Health, tags=["meta"])
    def healthz(svc: Svc) -> Health:
        s = svc.settings
        with svc.index_lock:
            reports = [svc.indexer.check_sync(st) for st in CHUNK_STRATEGIES]
        chunks = {r.strategy: r.chunk_store for r in reports}
        return Health(
            status="ok" if chunks[s.default_strategy] else "empty_index",
            llm_provider=s.llm_provider,
            generation_model=s.generation_model,
            embedder=f"{s.embedding_provider}/{s.embedding_model_name}",
            reranker=s.reranker,
            chunks=chunks,
            in_sync=all(r.in_sync for r in reports),
        )

    @app.post("/v1/ask", response_model=AskResponse, tags=["query"])
    def ask(req: AskRequest, svc: Svc) -> AskResponse:
        """Answer a question from the indexed docs, with verified [n] citations and a
        confidence breakdown. Returns status `insufficient_context` (plus what was and
        wasn't found) instead of guessing when the docs don't cover the question."""
        t = time.perf_counter()
        with svc.index_lock:
            res = svc.retriever.retrieve(
                req.question,
                mode=req.mode,
                strategy=req.strategy,
                reranker=req.reranker,
                top_n=req.top_n,
            )
        retrieval_ms = round((time.perf_counter() - t) * 1000, 1)
        answer = svc.generator.answer_from(req.question, res, verify=req.verify)
        answer.timings_ms = {"retrieval": retrieval_ms} | answer.timings_ms
        out = AskResponse(**answer.model_dump())
        if req.debug:
            final = {c.chunk.chunk_id: c.final_rank for c in res.results}
            out.candidates = [
                Candidate.from_retrieved(c, final.get(c.chunk.chunk_id)) for c in res.candidates
            ]
        return out

    @app.post("/v1/verify", response_model=AskResponse, tags=["query"])
    def verify(req: VerifyRequest, svc: Svc) -> AskResponse:
        """Audit a given answer instead of generating one: retrieve sources for the question
        (numbered [1]..[n] exactly as /v1/ask would), then check every claim's citations."""
        with svc.index_lock:
            res = svc.retriever.retrieve(
                req.question,
                mode=req.mode,
                strategy=req.strategy,
                reranker=req.reranker,
                top_n=req.top_n,
            )
        sources = svc.generator.sources_for(res)
        answer = svc.generator.check_answer(
            req.question, req.answer, sources, retrieval_confidence([s.relevance for s in sources])
        )
        answer.retrieval = {"mode": res.mode, "strategy": res.strategy, "reranker": res.reranker}
        return AskResponse(**answer.model_dump())

    @app.post("/v1/retrieve", response_model=RetrieveResponse, tags=["query"])
    def retrieve(req: RetrieveRequest, svc: Svc) -> RetrieveResponse:
        """Retrieval only (no generation): every candidate with its dense, sparse, fused and
        rerank rank/score. Cheap enough to compare retrieval modes side by side."""
        with svc.index_lock:
            res = svc.retriever.retrieve(
                req.query,
                mode=req.mode,
                strategy=req.strategy,
                reranker=req.reranker,
                top_n=req.top_n,
            )
        final = {c.chunk.chunk_id: c.final_rank for c in res.results}
        return RetrieveResponse(
            query=res.query,
            mode=res.mode,
            strategy=res.strategy,
            reranker=res.reranker,
            candidates=[
                Candidate.from_retrieved(c, final.get(c.chunk.chunk_id)) for c in res.candidates
            ],
            timings_ms=res.timings_ms,
        )

    @app.get("/v1/documents", response_model=list[DocumentInfo], tags=["documents"])
    def list_documents(svc: Svc) -> list[DocumentInfo]:
        with svc.index_lock:
            counts = _chunk_counts(svc)
        return [_doc_info(d, counts) for d in svc.store.list()]

    @app.get("/v1/documents/{doc_id:path}", response_model=DocumentInfo, tags=["documents"])
    def get_document(doc_id: str, svc: Svc) -> DocumentInfo:
        doc = svc.store.get(doc_id)
        if doc is None:
            raise HTTPException(404, f"No document {doc_id!r}")
        with svc.index_lock:
            return _doc_info(doc, _chunk_counts(svc))

    @app.delete(
        "/v1/documents/{doc_id:path}",
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
    )
    def delete_document(doc_id: str, svc: Svc) -> dict:
        with svc.index_lock:
            removed = svc.indexer.remove_document(doc_id)
            found = svc.store.delete(doc_id)
        if not (found or removed):
            raise HTTPException(404, f"No document {doc_id!r}")
        return {"doc_id": doc_id, "chunks_removed": removed}

    @app.post(
        "/v1/ingest",
        response_model=IngestResponse,
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
    )
    def ingest(
        file: Annotated[UploadFile, File(description="A .md, .txt, .html or .pdf file")],
        svc: Svc,
        doc_id: Annotated[str | None, Form(description="Default: uploads/<file name>")] = None,
        strategies: Annotated[str, Form(description="Comma-separated")] = ALL_STRATEGIES,
    ) -> IngestResponse:
        """Upload a document and index it with every requested chunking strategy.
        Re-uploading the same doc_id replaces the old version in all indexes."""
        t = time.perf_counter()
        doc_id = doc_id or f"uploads/{_safe_name(file.filename or '')}"
        if not supported(PurePosixPath(doc_id)):
            raise HTTPException(415, "Supported formats: .md, .markdown, .txt, .html, .htm, .pdf")
        chosen = _parse_strategies(strategies)
        limit = svc.settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"File is larger than {svc.settings.max_upload_mb} MB")
        if not data.strip():
            raise HTTPException(400, "File is empty")

        with svc.index_lock:
            try:
                doc = svc.store.ingest_bytes(data, doc_id)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            except Exception as e:  # malformed PDF/HTML etc.
                svc.store.delete(doc_id)
                raise HTTPException(422, f"Could not parse {doc_id}: {e}") from e
            if not doc.text.strip():
                svc.store.delete(doc_id)
                raise HTTPException(422, f"No text could be extracted from {doc_id}")
            stats = svc.indexer.index_documents([doc], chosen)
        return IngestResponse(
            doc_id=doc.doc_id,
            title=doc.title,
            format=doc.format,
            sections=len(doc.sections),
            strategies={
                st.strategy: StrategyStats(chunks=st.chunks, duplicates=st.duplicates)
                for st in stats
            },
            elapsed_ms=round((time.perf_counter() - t) * 1000, 1),
        )

    return app


def _safe_name(filename: str) -> str:
    name = PurePosixPath(filename.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    if not name:
        raise HTTPException(400, "Missing file name")
    return name


def _parse_strategies(value: str) -> tuple[ChunkStrategy, ...]:
    chosen = tuple(dict.fromkeys(s.strip() for s in value.split(",") if s.strip()))
    unknown = [s for s in chosen if s not in CHUNK_STRATEGIES]
    if unknown or not chosen:
        raise HTTPException(400, f"strategies must be among {', '.join(CHUNK_STRATEGIES)}")
    return chosen  # type: ignore[return-value]


def _chunk_counts(svc: Services) -> dict[str, Counter]:
    return {
        st: Counter(c.doc_id for c in svc.indexer.index(st).chunks.all()) for st in CHUNK_STRATEGIES
    }


def _doc_info(doc: Document, counts: dict[str, Counter]) -> DocumentInfo:
    return DocumentInfo(
        doc_id=doc.doc_id,
        title=doc.title,
        format=doc.format,
        sections=len(doc.sections),
        chars=len(doc.text),
        chunks={st: c[doc.doc_id] for st, c in counts.items()},
    )


app = create_app()
