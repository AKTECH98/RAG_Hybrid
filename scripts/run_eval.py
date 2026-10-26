"""Run the golden eval suite across pipeline configurations.

Usage:
  uv run python scripts/run_eval.py                                  # 3 strategies x hybrid,dense
  uv run python scripts/run_eval.py --strategies recursive --modes hybrid --limit 10
  uv run python scripts/run_eval.py --rerankers llm,cross_encoder,none --strategies recursive

All LLM calls are cached, so re-running after an unrelated change only pays for the cases
whose prompts changed. Writes reports/eval_<timestamp>.{json,md} and reports/latest.md.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from itertools import product

from docrag.config import CHUNK_STRATEGIES, PROJECT_ROOT, get_settings
from docrag.eval.compare import estimate_cost, render_report
from docrag.eval.dataset import load_golden
from docrag.eval.runner import EvalConfig, run_config
from docrag.generation.generator import build_generator


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
    p.add_argument("--limit", type=int, help="only the first N cases")
    p.add_argument("--cases", help="comma-separated case ids")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--judge-model", help="override the eval judge model")
    args = p.parse_args()

    settings = get_settings()
    judge_model = args.judge_model or settings.eval_judge_model
    cases = load_golden()
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.id in wanted]
    if args.limit:
        cases = cases[: args.limit]

    weights = [float(w) for w in args.dense_weights.split(",")] if args.dense_weights else [None]
    configs = [
        EvalConfig(strategy=s, mode=m, reranker=r, dense_weight=w)
        for s, m, r in product(
            _csv(args.strategies, CHUNK_STRATEGIES),
            _csv(args.modes, ("hybrid", "dense", "sparse")),
            _csv(args.rerankers, ("llm", "cross_encoder", "none")),
        )
        # weights only matter when two retrievers are fused
        for w in (weights if m == "hybrid" else [None])
    ]
    print(f"{len(cases)} cases x {len(configs)} configs; eval judge {judge_model}")

    generator = build_generator(settings)
    llm = generator.llm
    reports = []
    for cfg in configs:
        done = 0

        def progress(msg: str) -> None:
            nonlocal done
            done += 1
            print(f"\r  [{done:>3}/{len(cases)}] {msg[:90]:<90}", end="", file=sys.stderr)

        report = run_config(generator, cases, cfg, llm, judge_model, args.workers, progress)
        print(file=sys.stderr)
        s = report.summary
        print(
            f"{cfg.name:<28} correct={s['correctness']} faithful={s['faithfulness']} "
            f"cite={s['citation_accuracy']} recall@5={s['recall_at_k']} mrr={s['mrr']} "
            f"errors={s['errors']}"
        )
        reports.append(report)

    usage = dict(getattr(llm, "usage", {}))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {
        "timestamp": ts,
        "n_cases": len(cases),
        "case_types": ", ".join(f"{k} {v}" for k, v in Counter(c.type for c in cases).items()),
        "generation_model": settings.generation_model,
        "judge_model": settings.judge_model,
        "eval_judge_model": judge_model,
        "usage": usage,
        "usage_summary": ", ".join(
            f"{m}: {u['input']:,} in / {u['output']:,} out" for m, u in usage.items()
        )
        or "none (all cached)",
        "cost_usd": estimate_cost(usage),
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
