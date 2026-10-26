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
    ("separation_auc", "Abstain AUC"),
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
        f"- LLM provider: `{meta.get('llm_provider', 'openai')}` · generation: "
        f"`{meta['generation_model']}` · system verifier: `{meta['judge_model']}` · "
        f"eval judge: `{meta['eval_judge_model']}`  ",
        f"- Tokens this run: {meta['usage_summary']} (≈ ${meta['cost_usd']:.2f})",
        "",
        "## Results by configuration",
        "",
        "Config = [embedder:]chunking strategy / retrieval mode[@dense weight] / reranker. "
        "Retrieval metrics are over answerable cases; *Pool recall* is evidence recall over "
        "the 20 candidates handed to the reranker; *Abstain AUC* is how well the top "
        "retrieval score separates answerable from unanswerable questions (1.0 = perfectly).",
        "",
    ]
    columns = [
        (k, label)
        for k, label in MATRIX_COLUMNS
        if any(r.summary.get(k) is not None for r in reports)
    ]
    lines += [
        "| Config | " + " | ".join(label for _, label in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    best = max(reports, key=_rank_key)
    for r in reports:
        name = f"**{r.config.name}**" if r is best else r.config.name
        cells = [_fmt(r.summary.get(key)) for key, _ in columns]
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

    lines += _score_separation(reports)
    lines += _hybrid_vs_dense(reports)
    lines += _failures(best)
    return "\n".join(lines) + "\n"


def _rank_key(r: ConfigReport) -> tuple[float, float]:
    """Best config: highest correctness; retrieval-only runs fall back to recall, MRR."""
    s = r.summary
    if s.get("correctness") is not None:
        return (s["correctness"], s.get("recall_at_k") or 0)
    return (s.get("recall_at_k") or 0, s.get("mrr") or 0)


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    return vals[min(int(q * len(vals)), len(vals) - 1)]


def _score_separation(reports: list[ConfigReport]) -> list[str]:
    """Raw score distributions per config: the data behind abstain-threshold calibration."""
    rows = []
    for r in reports:
        ans = [x.top_cosine for x in r.results if x.type != "no_answer" and x.top_cosine]
        no = [x.top_cosine for x in r.results if x.type == "no_answer" and x.top_cosine]
        ans_rel = [
            x.top_relevance
            for x in r.results
            if x.type != "no_answer" and x.top_relevance is not None
        ]
        no_rel = [
            x.top_relevance
            for x in r.results
            if x.type == "no_answer" and x.top_relevance is not None
        ]
        if not (ans or ans_rel):
            continue
        cells = [
            f"{_f2(_pct(ans, 0.1))} / {_f2(_pct(ans, 0.5))}",
            _f2(max(no, default=None)),
            f"{_f2(_pct(ans_rel, 0.1))} / {_f2(_pct(ans_rel, 0.5))}",
            _f2(max(no_rel, default=None)),
        ]
        rows.append(f"| {r.config.name} | " + " | ".join(cells) + " |")
    if not rows:
        return []
    return [
        "",
        "## Score separation (abstain-threshold calibration)",
        "",
        'Top dense cosine and top calibrated relevance per question. A good "I don\'t know" '
        "threshold sits above the unanswerable max and below the answerable p10.",
        "",
        "| Config | Cosine answerable p10 / p50 | Cosine unanswerable max "
        "| Relevance answerable p10 / p50 | Relevance unanswerable max |",
        "|---|---:|---:|---:|---:|",
        *rows,
    ]


def _f2(v: float | None) -> str:
    return "–" if v is None else f"{v:.2f}"


def _hybrid_vs_dense(reports: list[ConfigReport]) -> list[str]:
    by_name = {r.config.name: r for r in reports}
    out: list[str] = []
    for r in reports:
        if r.config.mode != "hybrid":
            continue
        dense_cfg = r.config.model_copy(update={"mode": "dense", "dense_weight": None})
        dense = by_name.get(dense_cfg.name)
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
            f"**{r.config.name}** vs dense-only: hybrid found more evidence in "
            f"{len(gains)} case(s), less in {len(losses)}."
        )
        out += [f"- ✅ {g}" for g in gains] + [f"- ❌ {loss}" for loss in losses] + [""]
    return out


def _failures(report: ConfigReport) -> list[str]:
    if report.summary.get("correctness") is None:  # retrieval-only run
        missed = [
            r
            for r in report.results
            if (r.recall_at_k is not None and r.recall_at_k < 1) or r.error
        ]
        out = ["", f"## Evidence missed in the best config ({report.config.name})", ""]
        if not missed:
            return out + ["None."]
        return out + [
            f"- {r.case_id} ({r.type}, recall {_fmt(r.recall_at_k)}): {r.error or r.question}"
            for r in missed
        ]
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
