import re

import pytest
from fastapi.testclient import TestClient

from docrag.api.deps import Services
from docrag.api.main import create_app
from docrag.generation.generator import Generator
from docrag.ingest.store import DocumentStore
from docrag.retrieval.pipeline import Retriever

from .fakes import FakeLLM

DOC = b"""# Errors

## Raising HTTPException

Raise `HTTPException` with a `status_code` to return an HTTP error such as 404 to the client.
The `detail` can be any value that can be converted to JSON.

## Custom headers

Pass `headers` to `HTTPException` to add custom headers to the error response.
"""


def _verdicts(system: str, user: str) -> dict:
    """Mark every listed (claim, source) pair as supported."""
    verdicts = [
        {"claim_id": int(cid), "source": int(n), "verdict": "supported", "reason": "stated"}
        for cid, ns in re.findall(
            r"Claim (\d+):.*?-> check against sources \[([\d, ]+)\]", user, re.S
        )
        for n in ns.split(",")
    ]
    return {"verdicts": verdicts}


def _llm(text="Raise `HTTPException` with a `status_code` [1].") -> FakeLLM:
    return FakeLLM(
        {
            "_Verdicts": _verdicts,
            "_Parts": lambda s, u: {"parts": [{"part": "q", "addressed": "yes"}]},
            "_Gaps": lambda s, u: {"found": [], "missing": ["Anything about the question"]},
        },
        text=text,
    )


def _client(settings, indexer, embedder, llm=None, **overrides) -> TestClient:
    s = settings.model_copy(update={"reranker": "none", "abstain_threshold": 0.0} | overrides)
    llm = llm or _llm()
    indexer.settings = s
    retriever = Retriever(indexer, embedder, llm, s)
    services = Services(s, DocumentStore(s), indexer, retriever, Generator(retriever, llm, s))
    return TestClient(create_app(lambda: services))


def _upload(client, name="errors.md", data=DOC, **form):
    return client.post("/v1/ingest", files={"file": (name, data)}, data=form)


@pytest.fixture
def client(settings, indexer, embedder):
    with _client(settings, indexer, embedder) as c:
        yield c


def test_ingest_then_list_and_get(client):
    r = _upload(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["doc_id"] == "uploads/errors.md"
    assert set(body["strategies"]) == {"fixed", "recursive", "semantic"}
    assert all(s["chunks"] > 0 for s in body["strategies"].values())

    docs = client.get("/v1/documents").json()
    assert [d["doc_id"] for d in docs] == ["uploads/errors.md"]
    assert docs[0]["chunks"]["recursive"] == body["strategies"]["recursive"]["chunks"]
    assert client.get("/v1/documents/uploads/errors.md").json()["title"] == "Errors"
    assert client.get("/v1/documents/nope.md").status_code == 404


def test_ingest_one_strategy_and_reingest_replaces(client):
    first = _upload(client, strategies="recursive").json()
    assert list(first["strategies"]) == ["recursive"]
    again = _upload(client, strategies="recursive").json()
    assert again["strategies"]["recursive"]["chunks"] == first["strategies"]["recursive"]["chunks"]
    counts = client.get("/healthz").json()["chunks"]
    assert counts == {
        "fixed": 0,
        "recursive": first["strategies"]["recursive"]["chunks"],
        "semantic": 0,
    }


@pytest.mark.parametrize(
    ("name", "data", "form", "status"),
    [
        ("notes.docx", DOC, {}, 415),
        ("empty.md", b"   ", {}, 400),
        ("errors.md", DOC, {"strategies": "fixed,nope"}, 400),
        ("errors.md", DOC, {"doc_id": "../escape.md"}, 400),
    ],
)
def test_ingest_rejects_bad_input(client, name, data, form, status):
    assert _upload(client, name, data, **form).status_code == status


def test_ask_returns_verified_answer_and_debug_candidates(client):
    _upload(client)
    r = client.post("/v1/ask", json={"question": "How do I return a 404?", "debug": True})
    assert r.status_code == 200, r.text
    ans = r.json()
    assert ans["status"] == "answered"
    assert ans["claims"][0]["checks"][0]["verdict"] == "supported"
    assert ans["confidence"]["label"] in {"high", "medium", "low"}
    assert ans["retrieval"]["strategy"] == "recursive"
    assert "retrieval" in ans["timings_ms"] and "generation" in ans["timings_ms"]
    finals = [c for c in ans["candidates"] if c["final_rank"] is not None]
    assert len(finals) == len(ans["sources"])

    plain = client.post("/v1/ask", json={"question": "How do I return a 404?"}).json()
    assert plain["candidates"] is None


def test_ask_abstains_when_retrieval_is_weak(settings, indexer, embedder):
    with _client(settings, indexer, embedder, abstain_threshold=1.01) as c:
        _upload(c)
        ans = c.post("/v1/ask", json={"question": "How do I return a 404?"}).json()
    assert ans["status"] == "insufficient_context" and ans["answer"] is None
    assert ans["missing"] == ["Anything about the question"]


def test_retrieve_shows_every_stage(client):
    _upload(client)
    r = client.post("/v1/retrieve", json={"query": "HTTPException custom headers", "top_n": 2})
    body = r.json()
    assert body["mode"] == "hybrid"
    first = body["candidates"][0]
    assert first["fused_rank"] == 1 and first["dense_rank"] is not None
    n_final = sum(c["final_rank"] is not None for c in body["candidates"])
    assert n_final == min(2, len(body["candidates"]))

    dense = client.post("/v1/retrieve", json={"query": "headers", "mode": "dense"}).json()
    assert all(c["sparse_rank"] is None for c in dense["candidates"])


def test_validation_errors(client):
    assert client.post("/v1/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/v1/ask", json={"question": "what?", "mode": "bogus"}).status_code == 422


def test_model_failure_returns_502(settings, indexer, embedder):
    def boom(system, user):
        raise RuntimeError("Ollama /api/chat failed: HTTP 500")

    with _client(settings, indexer, embedder, llm=_llm(text=boom)) as c:
        _upload(c)
        r = c.post("/v1/ask", json={"question": "How do I return a 404?"})
    assert r.status_code == 502 and "HTTP 500" in r.json()["detail"]


def test_delete_requires_api_key_when_configured(settings, indexer, embedder):
    with _client(settings, indexer, embedder, api_key="secret") as c:
        assert _upload(c).status_code == 401
        ok = c.post(
            "/v1/ingest",
            files={"file": ("errors.md", DOC)},
            headers={"X-API-Key": "secret"},
        )
        assert ok.status_code == 200
        assert c.delete("/v1/documents/uploads/errors.md").status_code == 401
        r = c.delete("/v1/documents/uploads/errors.md", headers={"X-API-Key": "secret"})
        assert r.status_code == 200 and r.json()["chunks_removed"] > 0
        assert c.get("/v1/documents").json() == []
        missing = c.delete("/v1/documents/uploads/errors.md", headers={"X-API-Key": "secret"})
        assert missing.status_code == 404


def test_healthz(client):
    h = client.get("/healthz").json()
    assert h["status"] == "empty_index" and h["in_sync"]
    _upload(client)
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["chunks"]["recursive"] > 0


def test_verify_flags_a_hallucinated_citation(settings, indexer, embedder):
    def verdicts(system, user):
        return {
            "verdicts": [
                {"claim_id": 1, "source": 1, "verdict": "supported", "reason": "stated"},
                {"claim_id": 2, "source": 1, "verdict": "unsupported", "reason": "no 30s in [1]"},
            ]
        }

    llm = _llm()
    llm.structured_handlers["_Verdicts"] = verdicts
    with _client(settings, indexer, embedder, llm=llm) as c:
        _upload(c)
        r = c.post(
            "/v1/verify",
            json={
                "question": "How do I return a 404?",
                "answer": "Raise `HTTPException` with a `status_code` [1]. "
                "The default timeout is 30 seconds [1].",
            },
        )
    ans = r.json()
    assert r.status_code == 200, r.text
    assert [c["status"] for c in ans["claims"]] == ["supported", "unsupported"]
    assert [(u["claim_id"], u["source"]) for u in ans["unsupported_citations"]] == [(2, 1)]
    assert ans["confidence"]["label"] in {"medium", "low"}
    assert not any(llm_call[0] == "text" for llm_call in llm.calls)  # nothing generated
