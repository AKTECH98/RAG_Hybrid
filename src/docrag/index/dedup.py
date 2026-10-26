"""Near-duplicate detection: skip chunks with cosine > threshold to anything already indexed.

Same content often appears in several docs (shared code samples, boilerplate notes).
Without dedup, retrieval wastes context-window slots on copies of one passage.
"""

from dataclasses import asdict, dataclass

import numpy as np

from docrag.index.vector_store import VectorStore
from docrag.models import Chunk


@dataclass
class Duplicate:
    strategy: str
    dropped_id: str
    dropped_doc: str
    kept_id: str
    kept_doc: str
    similarity: float

    def to_dict(self) -> dict:
        return asdict(self)


def split_duplicates(
    chunks: list[Chunk],
    vectors: np.ndarray,
    store: VectorStore,
    threshold: float,
    known: dict[str, Chunk],
) -> tuple[list[int], list[Duplicate]]:
    """Return (indices of chunks to keep, duplicates found).

    Each chunk is compared against the vector store (other documents) and against chunks
    of this batch that were already kept. `known` maps chunk ids to chunks for logging.
    """
    if not chunks:
        return [], []
    neighbours = store.nearest(vectors, exclude_doc_id=chunks[0].doc_id)
    keep: list[int] = []
    dups: list[Duplicate] = []
    for i, chunk in enumerate(chunks):
        best_id, best_sim = None, -1.0
        if neighbours[i] is not None:
            best_id, best_sim = neighbours[i]
        if keep:
            sims = vectors[keep] @ vectors[i]
            j = int(np.argmax(sims))
            if sims[j] > best_sim:
                best_id, best_sim = chunks[keep[j]].chunk_id, float(sims[j])
        if best_id is not None and best_sim > threshold:
            kept = known.get(best_id) or next(c for c in chunks if c.chunk_id == best_id)
            dups.append(
                Duplicate(
                    strategy=chunk.strategy,
                    dropped_id=chunk.chunk_id,
                    dropped_doc=chunk.doc_id,
                    kept_id=best_id,
                    kept_doc=kept.doc_id,
                    similarity=round(best_sim, 4),
                )
            )
        else:
            keep.append(i)
    return keep, dups
