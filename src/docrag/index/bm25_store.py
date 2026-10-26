"""BM25 keyword index built from the chunk store.

The tokenizer is tuned for technical docs: identifiers are kept whole (`response_model`,
`HTTPException`, `request.url`, `422`) *and* split into their parts, so both an exact
identifier query and a plain-English query ("response model") can match.
"""

import re

from rank_bm25 import BM25Okapi

from docrag.index.chunk_store import ChunkStore

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|\d+(?:\.\d+)*")
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in into is it its
    may of on or so such that the their then there these this to was we what when where which
    who will with you your""".split()
)


def _parts(token: str) -> list[str]:
    parts = []
    for piece in re.split(r"[._]", token):
        parts.extend(CAMEL_RE.findall(piece))
    return [p.lower() for p in parts]


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in TOKEN_RE.findall(text):
        low = tok.lower()
        if low in STOPWORDS:
            continue
        tokens.append(low)
        parts = _parts(tok)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p not in STOPWORDS and len(p) > 1)
    return tokens


class BM25Store:
    """Rebuilt lazily whenever the chunk store's version changes."""

    def __init__(self, chunk_store: ChunkStore):
        self.chunk_store = chunk_store
        self._version = -1
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def _ensure(self) -> None:
        if self._version == self.chunk_store.version and self._bm25 is not None:
            return
        chunks = self.chunk_store.all()
        self._ids = [c.chunk_id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks]) if chunks else None
        self._version = self.chunk_store.version

    @property
    def ids(self) -> set[str]:
        self._ensure()
        return set(self._ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk_id, bm25 score) with score > 0, best first."""
        self._ensure()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1][:k]
        return [(self._ids[i], float(scores[i])) for i in order if scores[i] > 0]
