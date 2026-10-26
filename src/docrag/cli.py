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


STATUS_MARK = {"supported": "✓", "partial": "~", "unsupported": "✗", "uncited": "?"}


def cmd_ask(args: argparse.Namespace) -> None:
    from docrag.generation.generator import build_generator

    ans = build_generator().ask(
        args.question,
        mode=args.mode,
        strategy=args.strategy,
        reranker=args.reranker,
        verify=not args.no_verify,
    )
    if args.json:
        print(ans.model_dump_json(indent=2))
        return

    print(f"\nQ: {ans.question}\n")
    if ans.status == "insufficient_context":
        print("Not enough information in the indexed documentation to answer.")
        for label, items in (("Found", ans.found), ("Missing", ans.missing)):
            print(f"\n{label}:")
            for item in items or ["(nothing)"]:
                print(f"  - {item}")
        print(f"\nWorth checking manually: {', '.join(ans.suggested_documents) or '-'}")
    else:
        print(ans.answer)
        print("\nClaims:")
        for c in ans.claims:
            verdicts = " ".join(f"[{k.source}]{STATUS_MARK[k.verdict]}" for k in c.checks)
            print(f"  {STATUS_MARK[c.status]} {c.text[:90]}  {verdicts}")
        for u in ans.unsupported_citations:
            print(f"  ! claim {u.claim_id} cites [{u.source}] unsupported: {u.reason}")
    print("\nSources:")
    for s in ans.sources:
        print(f"  [{s.n}] {s.relevance:.2f}  {s.label[:100]}")
    c = ans.confidence
    print(
        f"\nConfidence: {c.composite:.2f} ({c.label})  retrieval={c.retrieval:.2f} "
        f"citations={c.citation_coverage:.2f} completeness={c.completeness:.2f}"
    )
    print(f"Timings (ms): {ans.timings_ms}  retrieval stages: {ans.retrieval.get('timings_ms')}")


def _add_retrieval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["hybrid", "dense", "sparse"])
    p.add_argument("--strategy", choices=CHUNK_STRATEGIES)
    p.add_argument("--reranker", choices=["llm", "cross_encoder", "none"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("retrieve", help="run the retrieval pipeline and show every stage")
    p.add_argument("query")
    _add_retrieval_args(p)
    p.add_argument("-n", type=int, default=None, help="final results to keep")
    p.add_argument("--all", action="store_true", help="show the whole candidate pool")
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("ask", help="answer a question with verified citations")
    p.add_argument("question")
    _add_retrieval_args(p)
    p.add_argument("--no-verify", action="store_true", help="skip citation verification")
    p.add_argument("--json", action="store_true", help="print the full Answer as JSON")
    p.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
