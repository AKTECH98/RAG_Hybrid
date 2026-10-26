import re

import pytest

from docrag.retrieval.fusion import weighted_rrf
from docrag.retrieval.pipeline import Retriever
from docrag.retrieval.rerank import LLMReranker

from .conftest import make_doc
from .fakes import FakeLLM

# ------------------------------------------------------------------ fusion


def test_rrf_math_matches_formula():
    fused = dict(
        weighted_rrf({"dense": ["a", "b"], "sparse": ["b", "c"]}, {"dense": 1, "sparse": 1})
    )
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement_and_respects_weights():
    ranked = {"dense": ["a", "b", "c"], "sparse": ["c", "x", "y"]}
    # c is found by both retrievers, so it beats a (top of one list only) -- with k=60
    # agreement dominates even at 0.9/0.1, which is why RRF is robust to bad weights
    assert weighted_rrf(ranked, {"dense": 0.5, "sparse": 0.5})[0][0] == "c"
    assert weighted_rrf(ranked, {"dense": 0.9, "sparse": 0.1})[0][0] == "c"
    # between disjoint top hits, the weights decide
    disjoint = {"dense": ["a"], "sparse": ["b"]}
    assert weighted_rrf(disjoint, {"dense": 0.7, "sparse": 0.3})[0][0] == "a"
    assert weighted_rrf(disjoint, {"dense": 0.3, "sparse": 0.7})[0][0] == "b"


def test_rrf_zero_weight_drops_a_list_and_ties_are_stable():
    assert [i for i, _ in weighted_rrf({"dense": ["a"], "sparse": ["b"]}, {"dense": 1})] == ["a"]
    tied = weighted_rrf({"dense": ["a", "b"], "sparse": ["b", "a"]}, {"dense": 1, "sparse": 1})
    assert [i for i, _ in tied] == ["a", "b"]


# ------------------------------------------------------------------ reranker


def test_llm_reranker_normalizes_scores_and_defaults_missing_ids(settings):
    llm = FakeLLM(
        {"_Scores": lambda s, u: {"scores": [{"id": 2, "score": 9}, {"id": 1, "score": 3}]}}
    )
    docs = [make_doc(f"{i}.md", [(["H"], f"text {i}")]) for i in range(3)]
    from docrag.chunking import RecursiveChunker

    chunks = [RecursiveChunker().chunk(d)[0] for d in docs]
    scores = LLMReranker(llm, "judge").score("question?", chunks)
    assert scores == [0.3, 0.9, 0.0]  # id 3 was never scored -> 0
    assert "[3] (2.md › H)" in llm.calls[0][2]


def test_cross_encoder_prefers_relevant_passage():
    pytest.importorskip("sentence_transformers")
    from docrag.models import Chunk
    from docrag.retrieval.rerank import CrossEncoderReranker

    def chunk(text):
        return Chunk(chunk_id=text, doc_id="d", strategy="s", chunk_index=0, text=text)

    rr = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    good, bad = rr.score(
        "How do I return a 404 not found error?",
        [
            chunk("Raise HTTPException(status_code=404) to return a Not Found error."),
            chunk("Pydantic models declare fields with type annotations."),
        ],
    )
    assert 0 <= bad < good <= 1 and good > 0.5


# ------------------------------------------------------------------ pipeline

CORPUS = [
    (
        "errors.md",
        "Handling Errors",
        "Raise HTTPException with status_code 404 when an item is missing. " * 4,
    ),
    ("timeouts.md", "Timeouts", "Configure the client timeout so slow requests fail fast. " * 4),
    ("models.md", "Body Models", "Declare a request body with a Pydantic BaseModel subclass. " * 4),
    (
        "config.md",
        "Settings",
        "Set the env var DOCRAG_RERANK_TOP_N to change how many results are kept. " * 4,
    ),
]


@pytest.fixture
def retriever(indexer, embedder, settings):
    docs = [make_doc(d, [([h], t)]) for d, h, t in CORPUS]
    indexer.index_documents(docs, ("recursive",))

    def keyword_judge(system, user):
        # score passages by overlap with the question's longest word
        q = re.search(r"Question: (.*)", user).group(1)
        key = max(re.findall(r"\w+", q), key=len).lower()
        passages = re.split(r"\n\n(?=\[\d+\])", user.split("Passages:\n\n", 1)[1])
        return {
            "scores": [
                {
                    "id": int(re.match(r"\[(\d+)\]", p).group(1)),
                    "score": 10 if key in p.lower() else 1,
                }
                for p in passages
            ]
        }

    return Retriever(indexer, embedder, FakeLLM({"_Scores": keyword_judge}), settings)


def test_hybrid_pipeline_tracks_every_stage(retriever):
    res = retriever.retrieve(
        "configure client timeout", strategy="recursive", reranker="none", top_n=2
    )
    assert res.mode == "hybrid" and len(res.results) == 2
    top = res.results[0]
    assert top.chunk.doc_id == "timeouts.md"
    assert top.final_rank == 1 and top.fused_rank == 1
    assert top.dense_rank is not None and top.sparse_rank is not None
    assert top.rerank_score is None and top.relevance == pytest.approx(top.dense_score)
    assert set(res.timings_ms) == {"dense", "sparse", "rerank"}


def test_sparse_mode_matches_exact_identifier(retriever):
    res = retriever.retrieve(
        "DOCRAG_RERANK_TOP_N", strategy="recursive", mode="sparse", reranker="none"
    )
    assert res.results[0].chunk.doc_id == "config.md"
    assert all(c.dense_rank is None for c in res.candidates)


def test_dense_mode_has_no_sparse_ranks_and_gets_full_pool(retriever, settings):
    res = retriever.retrieve("pydantic body", strategy="recursive", mode="dense", reranker="none")
    assert all(c.sparse_rank is None for c in res.candidates)
    assert len(res.candidates) == min(settings.rerank_candidates, 4)


def test_reranker_reorders_fused_candidates(retriever):
    res = retriever.retrieve(
        "how many results are kept for DOCRAG_RERANK_TOP_N", strategy="recursive", reranker="llm"
    )
    assert res.results[0].chunk.doc_id == "config.md"
    assert res.results[0].rerank_score == 1.0
    assert [c.final_rank for c in res.results] == list(range(1, len(res.results) + 1))
    assert len(res.candidates) == 4  # tiny corpus: every chunk is a candidate


def test_reranker_is_built_once_under_concurrency(retriever, monkeypatch):
    """Regression: parallel first calls each loaded the cross-encoder and crashed torch."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import docrag.retrieval.pipeline as pipeline

    builds = []
    lock = threading.Lock()

    def slow_build(name, llm, settings):
        with lock:
            builds.append(name)
        time.sleep(0.05)
        return None

    monkeypatch.setattr(pipeline, "get_reranker", slow_build)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: retriever._reranker("cross_encoder"), range(16)))
    assert builds == ["cross_encoder"]


def test_dense_weight_override_and_equal_fetch_depth(retriever, settings):
    q = "configure client timeout"
    sparse_only = retriever.retrieve(q, strategy="recursive", reranker="none", dense_weight=0.0)
    assert [c.chunk.chunk_id for c in sparse_only.candidates][: len(sparse_only.candidates)] == [
        c.chunk.chunk_id
        for c in sorted(
            [c for c in sparse_only.candidates if c.sparse_rank], key=lambda c: c.sparse_rank
        )
    ]
    # hybrid searches as deep as dense-only: every chunk dense-only sees, hybrid sees too
    hybrid = retriever.retrieve(q, strategy="recursive", reranker="none")
    dense = retriever.retrieve(q, strategy="recursive", mode="dense", reranker="none")
    assert {c.chunk.chunk_id for c in dense.candidates if c.dense_rank} <= {
        c.chunk.chunk_id for c in hybrid.candidates if c.dense_rank
    }
