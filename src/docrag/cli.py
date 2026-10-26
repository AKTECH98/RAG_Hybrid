"""Command line entry point: `uv run docrag <command> ...`."""

import argparse

from docrag.config import CHUNK_STRATEGIES


def _fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def cmd_retrieve(args: argparse.Namespace) -> None:
    from docrag.retrieval.pipeline import build_retriever

    retriever = build_retriever()
    res = retriever.retrieve(
        args.query, mode=args.mode, strategy=args.strategy, reranker=args.reranker, top_n=args.n
    )
    print(
        f"\nQ: {res.query}\nmode={res.mode} strategy={res.strategy} reranker={res.reranker} "
        f"candidates={len(res.candidates)} timings_ms={res.timings_ms}\n"
    )
    rows = res.candidates if args.all else res.results
    print(f"{'#':>2} {'rerank':>6} | {'fused':>9} | {'dense':>10} | {'sparse':>10} | source")
    for i, c in enumerate(rows, start=1):
        print(
            f"{c.final_rank or i:>2} {_fmt(c.rerank_score, '.2f'):>6} | "
            f"{_fmt(c.fused_rank, '>2')} {_fmt(c.fused_score, '.4f'):>6} | "
            f"{_fmt(c.dense_rank, '>2')} {_fmt(c.dense_score, '.3f'):>7} | "
            f"{_fmt(c.sparse_rank, '>2')} {_fmt(c.sparse_score, '.1f'):>7} | "
            f"{c.chunk.citation_label()[:90]}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("retrieve", help="run the retrieval pipeline and show every stage")
    p.add_argument("query")
    p.add_argument("--mode", choices=["hybrid", "dense", "sparse"])
    p.add_argument("--strategy", choices=CHUNK_STRATEGIES)
    p.add_argument("--reranker", choices=["llm", "cross_encoder", "none"])
    p.add_argument("-n", type=int, default=None, help="final results to keep")
    p.add_argument("--all", action="store_true", help="show the whole candidate pool")
    p.set_defaults(func=cmd_retrieve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
