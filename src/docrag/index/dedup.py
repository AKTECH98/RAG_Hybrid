"""Near-duplicate detection: skip chunks that are both semantically AND lexically
near-identical to something already indexed.

Same content often appears in several places (FastAPI repeats whole example files under
each subheading). Without dedup, retrieval wastes context-window slots on copies.

Embedding similarity alone is not enough: it measures "same topic", not "same text", and
its scale depends on the model. With bge-base at cosine > 0.95, Starlette's
`BackgroundTasks` section was dropped as a copy of `BackgroundTask`, and RFC 6749 §4.4.3
("a refresh token SHOULD NOT be included") as a copy of §4.1.4, which says the opposite.
On those false positives word overlap (Jaccard) had a median of 0.64, while true
duplicates were >= 0.69 (median 1.0). So a duplicate must pass both checks.
"""

import re
from dataclasses import asdict, dataclass

import numpy as np

from docrag.index.vector_store import VectorStore
from docrag.models import Chunk

WORD_RE = re.compile(r"\w+")


@dataclass
class Duplicate:
    strategy: str
    dropped_id: str
    dropped_doc: str
    kept_id: str
    kept_doc: str
    similarity: float
    jaccard: float

    def to_dict(self) -> dict:
        return asdict(self)


def word_jaccard(a: str, b: str) -> float:
    wa, wb = set(WORD_RE.findall(a.lower())), set(WORD_RE.findall(b.lower()))
    return len(wa & wb) / len(wa | wb) if wa or wb else 1.0


def split_duplicates(
    chunks: list[Chunk],
    vectors: np.ndarray,
    store: VectorStore,
    threshold: float,
    known: dict[str, Chunk],
    min_jaccard: float = 0.7,
) -> tuple[list[int], list[Duplicate]]:
    """Return (indices of chunks to keep, duplicates found).

    Each chunk is compared against its nearest stored neighbours (other documents) and
    against chunks of this batch that were already kept. `known` maps stored chunk ids to
    chunks so their text can be compared.
    """
    if not chunks:
        return [], []
    neighbours = store.nearest(vectors, exclude_doc_id=chunks[0].doc_id)
    keep: list[int] = []
    dups: list[Duplicate] = []
    for i, chunk in enumerate(chunks):
        candidates = [(known[cid], sim) for cid, sim in neighbours[i] if cid in known]
        if keep:
            sims = vectors[keep] @ vectors[i]
            candidates += [(chunks[keep[j]], float(sims[j])) for j in np.argsort(-sims)[:3]]
        match = next(
            (
                (other, sim, jac)
                for other, sim in sorted(candidates, key=lambda c: -c[1])
                if sim > threshold and (jac := word_jaccard(chunk.text, other.text)) >= min_jaccard
            ),
            None,
        )
        if match is None:
            keep.append(i)
            continue
        other, sim, jac = match
        dups.append(
            Duplicate(
                strategy=chunk.strategy,
                dropped_id=chunk.chunk_id,
                dropped_doc=chunk.doc_id,
                kept_id=other.chunk_id,
                kept_doc=other.doc_id,
                similarity=round(sim, 4),
                jaccard=round(jac, 3),
            )
        )
    return keep, dups
