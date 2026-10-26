"""Embedding with an on-disk cache.

Every text is keyed by sha1(model|text) in a SQLite cache, so re-chunking, re-indexing and
eval runs only pay for text that was never embedded before.
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np
import tiktoken

from docrag.config import Settings, get_settings

MAX_INPUT_TOKENS = 8000  # text-embedding-3 limit is 8191
BATCH_SIZE = 256


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / np.where(norms == 0, 1, norms)).astype(np.float32)


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 900):  # SQLite variable limit
            batch = keys[i : i + 900]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})"
            for key, blob in self.conn.execute(q, batch):
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?)",
            [(k, v.astype(np.float32).tobytes()) for k, v in items.items()],
        )
        self.conn.commit()


class OpenAIEmbedder:
    def __init__(self, settings: Settings | None = None):
        from openai import OpenAI

        s = settings or get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (add it to .env)")
        self.client = OpenAI(api_key=s.openai_api_key, max_retries=5)
        self.model = s.embedding_model
        self.dim = 1536
        self.cache = EmbeddingCache(s.cache_dir / "embeddings.sqlite")
        self.enc = tiktoken.get_encoding("cl100k_base")

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}|{text}".encode()).hexdigest()

    def _truncate(self, text: str) -> str:
        tokens = self.enc.encode(text, disallowed_special=())
        return (
            self.enc.decode(tokens[:MAX_INPUT_TOKENS]) if len(tokens) > MAX_INPUT_TOKENS else text
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        keys = [self._key(t) for t in texts]
        cached = self.cache.get_many(list(set(keys)))
        missing = list({k: t for k, t in zip(keys, texts, strict=True) if k not in cached}.items())
        for i in range(0, len(missing), BATCH_SIZE):
            batch = missing[i : i + BATCH_SIZE]
            resp = self.client.embeddings.create(
                model=self.model, input=[self._truncate(t) or " " for _, t in batch]
            )
            new = {
                k: np.asarray(d.embedding, dtype=np.float32)
                for (k, _), d in zip(batch, resp.data, strict=True)
            }
            self.cache.put_many(new)
            cached.update(new)
        return _normalize(np.stack([cached[k] for k in keys]))


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


def get_embedder(settings: Settings | None = None) -> Embedder:
    return OpenAIEmbedder(settings)
