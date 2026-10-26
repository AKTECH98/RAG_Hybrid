import json

import httpx
import numpy as np
import pytest

from docrag.eval.compare import render_report
from docrag.eval.dataset import Evidence, GoldenCase
from docrag.eval.runner import EvalConfig, run_config, separation_auc
from docrag.generation.generator import Generator
from docrag.index.embedder import CachedEmbedder, OllamaEmbedder
from docrag.llm import OllamaLLM
from docrag.retrieval.pipeline import Retriever
from docrag.retrieval.rerank import _Scores

from .conftest import make_doc
from .fakes import FakeLLM

# ------------------------------------------------------------------ settings


def test_each_embedder_gets_its_own_index_dir(settings):
    assert settings.embedding_model_name == "text-embedding-3-small"
    local = settings.with_embedder("local")
    ollama = settings.with_embedder("ollama", "nomic-embed-text:v1.5")
    assert local.embedding_model_name == "BAAI/bge-base-en-v1.5"
    assert local.index_slug == "local-baai-bge-base-en-v1.5"
    assert ollama.index_slug == "ollama-nomic-embed-text-v1.5"
    assert len({settings.chroma_dir, local.chroma_dir, ollama.chroma_dir}) == 3
    assert local.raw_dir == settings.raw_dir  # documents are shared, indexes are not


# ------------------------------------------------------------------ embedders


class CountingEmbedder(CachedEmbedder):
    query_prefix = "Q: "
    document_prefix = "D: "

    def __init__(self, settings):
        super().__init__(settings)
        self.seen: list[str] = []

    def _embed_batch(self, texts):
        self.seen.extend(texts)
        return np.array([[len(t), 1.0, 0.0] for t in texts], dtype=np.float32)


def test_cached_embedder_prefixes_caches_and_normalizes(settings):
    emb = CountingEmbedder(settings)
    docs = emb.embed(["alpha", "beta"])
    q = emb.embed_query("alpha")
    assert emb.seen == ["D: alpha", "D: beta", "Q: alpha"]  # query != document text
    emb.embed(["alpha"])
    assert len(emb.seen) == 3  # cache hit, no new call
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0) and np.isclose(np.linalg.norm(q), 1.0)


def test_ollama_embedder_uses_nomic_prefixes(settings):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]] * len(body["input"])})

    emb = OllamaEmbedder(settings.with_embedder("ollama"))
    emb.client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
    emb.embed(["a doc"])
    emb.embed_query("a question")
    assert calls[0] == {
        "model": "nomic-embed-text",
        "input": ["search_document: a doc"],
        "truncate": True,
    }
    assert calls[1]["input"] == ["search_query: a question"]


# ------------------------------------------------------------------ Ollama LLM


def _ollama(settings, responses: list[dict]) -> tuple[OllamaLLM, list[dict]]:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=responses[len(sent) - 1])

    llm = OllamaLLM(settings, use_cache=False)
    llm.client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
    return llm, sent


def _msg(content: str, prompt_tokens: int = 50) -> dict:
    return {"message": {"content": content}, "prompt_eval_count": prompt_tokens, "eval_count": 7}


def test_ollama_llm_sends_context_think_and_schema(settings):
    llm, sent = _ollama(settings, [_msg('{"scores": [{"id": 1, "score": 9}]}')])
    result = llm.structured("qwen", "system", "user", _Scores)
    assert result.scores[0].score == 9
    body = sent[0]
    assert body["think"] is False and body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_ctx": settings.ollama_num_ctx}
    assert body["format"] == _Scores.model_json_schema()
    assert llm.usage == {"qwen": {"input": 50, "output": 7, "calls": 1}}


def test_ollama_llm_retries_once_on_invalid_structured_output(settings):
    llm, sent = _ollama(
        settings,
        [
            _msg('{"scores": [{"id": 1, "score": 11}]}'),  # violates le=10
            _msg('{"scores": [{"id": 1, "score": 10}]}'),
        ],
    )
    assert llm.structured("qwen", "s", "u", _Scores).scores[0].score == 10
    assert "previous output was invalid" in sent[1]["messages"][1]["content"]


def test_ollama_llm_refuses_silently_truncated_prompts(settings):
    llm, _ = _ollama(settings, [_msg("answer", prompt_tokens=settings.ollama_num_ctx)])
    with pytest.raises(RuntimeError, match="truncated"):
        llm.generate("qwen", "s", "u")


# ------------------------------------------------------------------ retrieval-only eval


def test_separation_auc():
    assert separation_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert separation_auc([0.1], [0.9]) == 0.0
    assert separation_auc([0.5], [0.5]) == 0.5
    assert separation_auc([], [0.1]) is None


def test_retrieval_only_run_skips_generation_and_reports(indexer, embedder, settings):
    docs = [
        make_doc("t.md", [(["Timeouts"], "Set the client timeout to fail fast. " * 5)]),
        make_doc("c.md", [(["Cookies"], "Cookies are sent with every request. " * 5)]),
    ]
    indexer.index_documents(docs, ("recursive",))
    llm = FakeLLM()  # any LLM call would fail: no handlers, no text
    gen = Generator(Retriever(indexer, embedder, llm, settings), llm, settings)
    cases = [
        GoldenCase(
            id="L1",
            type="lookup",
            question="client timeout fail fast",
            answer="",
            evidence=[Evidence(doc_id="t.md", quote="client timeout")],
        ),
        GoldenCase(id="N1", type="no_answer", question="kubernetes autoscaling", answer=""),
    ]
    cfg = EvalConfig(strategy="recursive", mode="hybrid", reranker="none", embedder="local")
    report = run_config(gen, cases, cfg, llm, "judge", workers=2, retrieval_only=True)

    assert llm.calls == []
    assert report.summary["recall_at_k"] == 1.0 and report.summary["errors"] == 0
    assert report.summary["correctness"] is None
    assert report.summary["separation_auc"] == 1.0
    assert cfg.name == "local:recursive/hybrid/none"

    md = render_report(
        [report],
        {
            "timestamp": "t",
            "n_cases": 2,
            "case_types": "x",
            "generation_model": "g",
            "judge_model": "j",
            "eval_judge_model": "n/a",
            "usage_summary": "none",
            "cost_usd": 0.0,
        },
    )
    assert "Abstain AUC" in md and "Faithful" not in md  # empty columns are hidden
    assert "Score separation" in md and "Evidence missed" in md
