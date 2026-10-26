"""Render eval results as a Markdown report: config matrix, per-type breakdown, hybrid vs
dense case diffs, and the failures that need a human look."""

from docrag.eval.runner import ConfigReport

# USD per 1M tokens (input, output). List prices assumed at time of writing; update if
# they change. Only used for the cost line in the report.
PRICES = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}

MATRIX_COLUMNS = [
    ("correctness", "Correct"),
    ("faithfulness", "Faithful"),
    ("citation_accuracy", "Cite acc"),
    ("recall_at_k", "Recall@5"),
    ("mrr", "MRR"),
    ("pool_recall", "Pool recall"),
    ("no_answer_accuracy", "No-ans acc"),
    ("false_abstain_rate", "False abstain"),
    ("latency_p50_ms", "p50 ms*"),
]


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float) and v <= 1.0:
        return f"{v:.0%}"
    return str(v)


def estimate_cost(usage: dict[str, dict[str, int]]) -> float:
    total = 0.0
    for model, u in usage.items():
        price_in, price_out = PRICES.get(model, (0.0, 0.0))
        total += u["input"] / 1e6 * price_in + u["output"] / 1e6 * price_out
    return total


def render_report(reports: list[ConfigReport], meta: dict) -> str:
    lines = [
        "# RAG evaluation report",
        "",
        f"- Run: {meta['timestamp']}  ",
        f"- Cases: {meta['n_cases']} ({meta['case_types']})  ",
        f"- Generation: `{meta['generation_model']}` · system verifier: "
        f"`{meta['judge_model']}` · eval judge: `{meta['eval_judge_model']}`  ",
        f"- Tokens billed this run: {meta['usage_summary']} (≈ ${meta['cost_usd']:.2f})",
        "",
        "## Results by configuration",
        "",
        "Config = chunking strategy / retrieval mode / reranker. Retrieval metrics are over "
        "answerable cases; *Pool recall* is evidence recall over the 20 candidates handed "
        "to the reranker.",
        "",
        "| Config | " + " | ".join(label for _, label in MATRIX_COLUMNS) + " |",
        "|---|" + "---:|" * len(MATRIX_COLUMNS),
    ]
    best = max(reports, key=lambda r: r.summary["correctness"] or 0)
    for r in reports:
        name = f"**{r.config.name}**" if r is best else r.config.name
        cells = [_fmt(r.summary.get(key)) for key, _ in MATRIX_COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "\\* End-to-end latency including verification. LLM responses are cached, so configs "
        "re-run on cached prompts look faster; compare latency only within a fresh run.",
    ]

    types = sorted({t for r in reports for t in r.by_type})
    lines += [
        "",
        "## Correctness and Recall@5 by question type",
        "",
        "| Config | " + " | ".join(types) + " |",
        "|---|" + "---:|" * len(types),
    ]
    for r in reports:
        cells = []
        for t in types:
            s = r.by_type.get(t, {})
            rec = s.get("recall_at_k")
            cells.append(
                _fmt(s.get("correctness")) + (f" · R {_fmt(rec)}" if rec is not None else "")
            )
        lines.append(f"| {r.config.name} | " + " | ".join(cells) + " |")

    lines += _hybrid_vs_dense(reports)
    lines += _failures(best)
    return "\n".join(lines) + "\n"


def _hybrid_vs_dense(reports: list[ConfigReport]) -> list[str]:
    by_name = {r.config.name: r for r in reports}
    out: list[str] = []
    for r in reports:
        if r.config.mode != "hybrid":
            continue
        dense = by_name.get(f"{r.config.strategy}/dense/{r.config.reranker}")
        if not dense:
            continue
        d_by_id = {x.case_id: x for x in dense.results}
        gains, losses = [], []
        for h in r.results:
            d = d_by_id.get(h.case_id)
            if not d or h.recall_at_k is None or d.recall_at_k is None:
                continue
            if h.recall_at_k > d.recall_at_k:
                gains.append(f"{h.case_id} ({d.recall_at_k:.0%}→{h.recall_at_k:.0%}): {h.question}")
            elif h.recall_at_k < d.recall_at_k:
                losses.append(
                    f"{h.case_id} ({d.recall_at_k:.0%}→{h.recall_at_k:.0%}): {h.question}"
                )
        if not out:
            out += ["", "## Hybrid vs dense-only: where retrieval changed", ""]
        out.append(
            f"**{r.config.strategy} / {r.config.reranker}**: hybrid found more evidence in "
            f"{len(gains)} case(s), less in {len(losses)}."
        )
        out += [f"- ✅ {g}" for g in gains] + [f"- ❌ {loss}" for loss in losses] + [""]
    return out


def _failures(report: ConfigReport) -> list[str]:
    bad = [r for r in report.results if (r.correctness or 0) < 1.0 or r.error]
    out = ["", f"## Cases not fully correct in the best config ({report.config.name})", ""]
    if not bad:
        return out + ["None."]
    out += ["| Case | Type | Score | Recall@5 | Judge |", "|---|---|---:|---:|---|"]
    for r in bad:
        reason = (r.error or r.correctness_reason).replace("|", "/").replace("\n", " ")[:160]
        out.append(
            f"| {r.case_id} | {r.type} | {_fmt(r.correctness)} | {_fmt(r.recall_at_k)} | {reason} |"
        )
    return out
