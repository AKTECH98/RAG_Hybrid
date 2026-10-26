"""Weighted Reciprocal Rank Fusion.

    score(d) = sum over lists L of  w_L / (k + rank_L(d))      (rank is 1-based)

RRF only looks at rank positions, so it can merge cosine similarities and BM25 scores
without calibrating them against each other. k (default 60) damps the gap between the
first few ranks; the weights let one retriever count more (e.g. 0.7 dense / 0.3 sparse).
"""


def weighted_rrf(
    ranked: dict[str, list[str]], weights: dict[str, float], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one list of (id, score), best first.

    Ties keep the order of first appearance (lists in the order given), so results are
    deterministic.
    """
    scores: dict[str, float] = {}
    for name, ids in ranked.items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
