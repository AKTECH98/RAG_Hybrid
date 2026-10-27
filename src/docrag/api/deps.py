"""Long-lived components shared by every request, built once at startup."""

import threading
from dataclasses import dataclass, field

from fastapi import Header, HTTPException, Request

from docrag.config import Settings, get_settings
from docrag.generation.generator import Generator
from docrag.index.indexer import Indexer
from docrag.ingest.store import DocumentStore
from docrag.retrieval.pipeline import Retriever


@dataclass
class Services:
    settings: Settings
    store: DocumentStore
    indexer: Indexer
    retriever: Retriever
    generator: Generator
    # Ingest mutates the chunk store that BM25 and retrieval read. Retrieval takes
    # milliseconds, so it simply waits while a document is being indexed; generation and
    # verification (the slow part) run outside the lock.
    index_lock: threading.RLock = field(default_factory=threading.RLock)


def build_services(settings: Settings | None = None) -> Services:
    from docrag.index.embedder import get_embedder
    from docrag.llm import get_llm

    s = settings or get_settings()
    embedder, llm = get_embedder(s), get_llm(s)
    indexer = Indexer(embedder, s)
    retriever = Retriever(indexer, embedder, llm, s)
    return Services(s, DocumentStore(s), indexer, retriever, Generator(retriever, llm, s))


def get_services(request: Request) -> Services:
    return request.app.state.services


def require_api_key(request: Request, x_api_key: str | None = Header(None)) -> None:
    expected = get_services(request).settings.api_key
    if expected and x_api_key != expected:
        raise HTTPException(401, "Missing or invalid X-API-Key")
