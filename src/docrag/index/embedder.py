"""Embedders with an on-disk cache: OpenAI, local sentence-transformers, or Ollama.

Every text is keyed by sha1(model|text) in a SQLite cache, so re-chunking, re-indexing and
eval runs only pay for text that was never embedded before.

Retrieval-tuned models embed queries and documents differently: bge expects an
instruction in front of queries, nomic-embed-text expects "search_query: " /
"search_document: " prefixes. `embed` is for documents, `embed_query` for queries.
"""

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Protocol

import numpy as np
import tiktoken

from docrag.config import Settings, get_settings

MAX_INPUT_TOKENS = 8000  # text-embedding-3 limit is 8191


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Documents -> (n, dim) float32 array of L2-normalized vectors."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """One query -> (dim,) L2-normalized vector."""
        ...


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / np.where(norms == 0, 1, norms)).astype(np.float32)


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 900):  # SQLite variable limit
            batch = keys[i : i + 900]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})"
            with self.lock:
                rows = self.conn.execute(q, batch).fetchall()
            for key, blob in rows:
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        with self.lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO emb VALUES (?, ?)",
                [(k, v.astype(np.float32).tobytes()) for k, v in items.items()],
            )
            self.conn.commit()


class CachedEmbedder:
    """Base class: prefixes, caching and batching. Subclasses implement `_embed_batch`."""

    model: str
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 64

    def __init__(self, settings: Settings):
        self.model = settings.embedding_model_name
        self.cache = EmbeddingCache(settings.cache_dir / "embeddings.sqlite")

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}|{text}".encode()).hexdigest()

    def _embed_cached(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        keys = [self._key(t) for t in texts]
        cached = self.cache.get_many(list(set(keys)))
        missing = list({k: t for k, t in zip(keys, texts, strict=True) if k not in cached}.items())
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i : i + self.batch_size]
            vecs = self._embed_batch([t or " " for _, t in batch])
            new = {k: v for (k, _), v in zip(batch, vecs, strict=True)}
            self.cache.put_many(new)
            cached.update(new)
        return _normalize(np.stack([cached[k] for k in keys]))

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._embed_cached([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_cached([self.query_prefix + text])[0]


class OpenAIEmbedder(CachedEmbedder):
    batch_size = 256

    def __init__(self, settings: Settings):
        from openai import OpenAI

        super().__init__(settings)
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (add it to .env)")
        self.client = OpenAI(api_key=settings.openai_api_key, max_retries=5)
        self.enc = tiktoken.get_encoding("cl100k_base")

    def _truncate(self, text: str) -> str:
        tokens = self.enc.encode(text, disallowed_special=())
        return (
            self.enc.decode(tokens[:MAX_INPUT_TOKENS]) if len(tokens) > MAX_INPUT_TOKENS else text
        )

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=self.model, input=[self._truncate(t) for t in texts]
        )
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


# Query instructions recommended by each model's authors (documents get none for bge).
LOCAL_QUERY_PREFIXES = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class LocalEmbedder(CachedEmbedder):
    """sentence-transformers on this machine (Apple GPU / CUDA / CPU). Inputs beyond the
    model's max sequence length (512 tokens for bge) are truncated by the model."""

    batch_size = 64

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.query_prefix = LOCAL_QUERY_PREFIXES.get(self.model, "")
        self._st = None
        self._lock = threading.Lock()

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        with self._lock:  # one torch model shared across threads; load it once
            if self._st is None:
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(self.model)
            return np.asarray(
                self._st.encode(texts, batch_size=32, normalize_embeddings=True),
                dtype=np.float32,
            )


OLLAMA_PREFIXES = {  # (query, document)
    "nomic-embed-text": ("search_query: ", "search_document: "),
}


class OllamaEmbedder(CachedEmbedder):
    """Ollama's native /api/embed endpoint (batched)."""

    batch_size = 32

    def __init__(self, settings: Settings):
        import httpx

        super().__init__(settings)
        base = self.model.split(":")[0]
        self.query_prefix, self.document_prefix = OLLAMA_PREFIXES.get(base, ("", ""))
        self.client = httpx.Client(
            base_url=settings.ollama_base_url, timeout=settings.ollama_timeout_s
        )

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        from docrag.llm import ollama_post

        data = ollama_post(
            self.client, "/api/embed", {"model": self.model, "input": texts, "truncate": True}
        )
        return np.array(data["embeddings"], dtype=np.float32)


class HashEmbedder:
    """Deterministic offline embedder for tests: bag of hashed word features.

    Texts sharing words get similar vectors, which is enough to exercise dedup, semantic
    chunking and dense retrieval without network calls.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _normalize(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


_embedders: dict[str, Embedder] = {}
_embedders_lock = threading.Lock()


def get_embedder(settings: Settings | None = None) -> Embedder:
    """One shared embedder per (provider, model)."""
    s = settings or get_settings()
    with _embedders_lock:
        if s.index_slug not in _embedders:
            cls = {"openai": OpenAIEmbedder, "local": LocalEmbedder, "ollama": OllamaEmbedder}
            _embedders[s.index_slug] = cls[s.embedding_provider](s)
        return _embedders[s.index_slug]
