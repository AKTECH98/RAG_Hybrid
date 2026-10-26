"""Run the golden eval suite across pipeline configurations.

Usage:
  uv run python scripts/run_eval.py                                  # 3 strategies x hybrid,dense
  uv run python scripts/run_eval.py --strategies recursive --modes hybrid --limit 10
  uv run python scripts/run_eval.py --rerankers llm,cross_encoder,none --strategies recursive
  uv run python scripts/run_eval.py --retrieval-only --embedders local,ollama --rerankers none

--retrieval-only skips generation and LLM judging: it scores recall/MRR and how well the top
retrieval score separates answerable from unanswerable questions. With a local reranker it
needs no LLM at all, which makes embedder/strategy/weight sweeps take minutes.

All LLM calls are cached, so re-running after an unrelated change only pays for the cases
whose prompts changed. Writes reports/eval_<timestamp>.{json,md} and reports/latest.md.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from itertools import product

from docrag.config import CHUNK_STRATEGIES, EMBEDDING_PROVIDERS, PROJECT_ROOT, get_settings
from docrag.eval.compare import estimate_cost, render_report
from docrag.eval.dataset import load_golden
from docrag.eval.runner import EvalConfig, run_config
from docrag.generation.generator import Generator
from docrag.index.chunk_store import ChunkStore
from docrag.index.embedder import get_embedder
from docrag.index.indexer import Indexer
from docrag.llm import get_llm
from docrag.retrieval.pipeline import Retriever


def _csv(value: str, allowed: tuple[str, ...]) -> list[str]:
    items = list(allowed) if value == "all" else [v.strip() for v in value.split(",")]
    bad = [i for i in items if i not in allowed]
    if bad:
        raise SystemExit(f"invalid value(s) {bad}; choose from {allowed}")
    return items


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--strategies", default="all")
    p.add_argument("--modes", default="hybrid,dense")
    p.add_argument("--rerankers", default="llm")
    p.add_argument(
        "--dense-weights",
        help="hybrid only: comma-separated dense weights to sweep, e.g. 0.7,0.5,0.3 "
        "(sparse weight = 1 - dense). Default: the configured weights.",
    )
    p.add_argument(
        "--embedders",
        help="comma-separated embedding providers, each needs its own index "
        "(scripts/seed.py --embedder X). Default: the configured provider.",
    )
    p.add_argument("--retrieval-only", action="store_true", help="skip generation and judging")
    p.add_argument("--limit", type=int, help="only the first N cases")
    p.add_argument("--cases", help="comma-separated case ids")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--judge-model", help="override the eval judge model")
    args = p.parse_args()

    base = get_settings()
    judge_model = args.judge_model or base.eval_judge_model
    cases = load_golden()
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.id in wanted]
    if args.limit:
        cases = cases[: args.limit]

    embedders = _csv(args.embedders, EMBEDDING_PROVIDERS) if args.embedders else [None]
    weights = [float(w) for w in args.dense_weights.split(",")] if args.dense_weights else [None]
    configs = [
        EvalConfig(strategy=s, mode=m, reranker=r, dense_weight=w, embedder=e)
        for e, s, m, r in product(
            embedders,
            _csv(args.strategies, CHUNK_STRATEGIES),
            _csv(args.modes, ("hybrid", "dense", "sparse")),
            _csv(args.rerankers, ("llm", "cross_encoder", "none")),
        )
        # weights only matter when two retrievers are fused
        for w in (weights if m == "hybrid" else [None])
    ]
    mode = "retrieval only" if args.retrieval_only else f"eval judge {judge_model}"
    print(f"{len(cases)} cases x {len(configs)} configs; {mode}; LLM provider {base.llm_provider}")

    llm = get_llm(base)
    generators: dict[str | None, Generator] = {}

    def generator_for(embedder: str | None) -> Generator:
        if embedder not in generators:
            s = base.with_embedder(embedder) if embedder else base
            if not ChunkStore("recursive", s).path.exists():
                raise SystemExit(
                    f"No index for {s.index_slug}; run: scripts/seed.py --embedder {embedder}"
                )
            emb = get_embedder(s)
            generators[embedder] = Generator(Retriever(Indexer(emb, s), emb, llm, s), llm, s)
        return generators[embedder]

    reports = []
    for cfg in configs:
        done = 0

        def progress(msg: str) -> None:
            nonlocal done
            done += 1
            print(f"\r  [{done:>3}/{len(cases)}] {msg[:90]:<90}", end="", file=sys.stderr)

        report = run_config(
            generator_for(cfg.embedder),
            cases,
            cfg,
            llm,
            judge_model,
            args.workers,
            progress,
            retrieval_only=args.retrieval_only,
        )
        print(file=sys.stderr)
        s = report.summary
        print(
            f"{cfg.name:<36} correct={s['correctness']} faithful={s['faithfulness']} "
            f"cite={s['citation_accuracy']} recall@5={s['recall_at_k']} mrr={s['mrr']} "
            f"auc={s['separation_auc']} errors={s['errors']}"
        )
        reports.append(report)

    usage = dict(getattr(llm, "usage", {}))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {
        "timestamp": ts,
        "n_cases": len(cases),
        "case_types": ", ".join(f"{k} {v}" for k, v in Counter(c.type for c in cases).items()),
        "llm_provider": base.llm_provider,
        "generation_model": base.generation_model,
        "judge_model": base.judge_model,
        "eval_judge_model": "n/a (retrieval only)" if args.retrieval_only else judge_model,
        "usage": usage,
        "usage_summary": ", ".join(
            f"{m}: {u['input']:,} in / {u['output']:,} out" for m, u in usage.items()
        )
        or "none (all cached)",
        "cost_usd": estimate_cost(usage) if base.llm_provider == "openai" else 0.0,
    }
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"eval_{ts}.json").write_text(
        json.dumps({"meta": meta, "reports": [r.model_dump() for r in reports]}, indent=1)
    )
    md = render_report(reports, meta)
    (out_dir / f"eval_{ts}.md").write_text(md)
    (out_dir / "latest.md").write_text(md)
    print(f"\nReport: reports/eval_{ts}.md  (≈ ${meta['cost_usd']:.2f} in API calls this run)")


if __name__ == "__main__":
    main()
