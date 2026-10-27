"""Streamlit dashboard for the docrag API.

  uv run streamlit run dashboard/app.py

Talks to the API over HTTP only (DOCRAG_API_URL, default http://localhost:8000), so it
needs nothing from the docrag package and runs in its own small container.
"""

import os
import re
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("DOCRAG_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("DOCRAG_API_KEY")
REPORTS_DIR = Path(os.environ.get("DOCRAG_REPORTS_DIR", Path(__file__).parents[1] / "reports"))

STRATEGIES = ["recursive", "fixed", "semantic"]
MODES = ["hybrid", "dense", "sparse"]
RERANKERS = ["cross_encoder", "llm", "none"]
STATUS_ICON = {"supported": "✅", "partial": "🟡", "unsupported": "❌", "uncited": "⚪"}
LABEL_COLOR = {"high": "green", "medium": "orange", "low": "red"}
EXAMPLES = [
    "How do I add custom headers to an HTTPException in FastAPI?",
    "What is the default timeout in HTTPX?",
    "When does the code after yield run in a FastAPI dependency?",
    "Why should OAuth clients use PKCE, and how does FastAPI handle OAuth2 password flow?",
    "How do I configure Kafka consumer groups?",
]
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

st.set_page_config(page_title="docrag", page_icon="📚", layout="wide")


# ------------------------------------------------------------------ API access


@st.cache_resource
def _client() -> httpx.Client:
    # Answers from a small local model take 20-60 s; verification is included.
    return httpx.Client(base_url=API_URL, timeout=900)


def api(method: str, path: str, **kwargs) -> dict | list | None:
    """Call the API; show the error in the page and return None on failure."""
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    try:
        r = _client().request(method, path, headers=headers, **kwargs)
    except httpx.HTTPError as e:
        st.error(f"Could not reach the API at {API_URL}: {e}")
        return None
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(f"API error {r.status_code}: {detail}")
        return None
    return r.json()


@st.cache_data(ttl=15, show_spinner=False)
def health() -> dict | None:
    try:
        return _client().get("/healthz", timeout=10).json()
    except (httpx.HTTPError, ValueError):
        return None


# ------------------------------------------------------------------ sidebar


def sidebar() -> dict:
    h = health()
    with st.sidebar:
        st.title("📚 docrag")
        st.caption("Hybrid retrieval · reranking · verified citations")
        if h is None:
            st.error(f"API not reachable at `{API_URL}`")
            st.code("uv run uvicorn docrag.api.main:app", language="bash")
        else:
            (st.success if h["status"] == "ok" else st.warning)(
                f"API {h['status']}" + ("" if h["in_sync"] else " · indexes out of sync")
            )
            st.caption(
                f"LLM: `{h['llm_provider']}/{h['generation_model']}`  \n"
                f"Embedder: `{h['embedder']}`  \n"
                "Chunks: " + ", ".join(f"{k} {v:,}" for k, v in h["chunks"].items())
            )

        st.subheader("Retrieval")
        default_rr = h["reranker"] if h and h["reranker"] in RERANKERS else "cross_encoder"
        opts = {
            "strategy": st.selectbox("Chunking strategy", STRATEGIES),
            "mode": st.radio("Retrieval mode", MODES, horizontal=True),
            "reranker": st.selectbox("Reranker", RERANKERS, index=RERANKERS.index(default_rr)),
            "top_n": st.slider("Chunks given to the model", 1, 10, 5),
        }
        st.subheader("Answer")
        opts["verify"] = st.toggle("Verify citations", value=True)
        opts["compare"] = st.toggle(
            "Compare with " + ("hybrid" if opts["mode"] == "dense" else "dense-only"),
            help="Runs the same question twice and shows both side by side. "
            "With a local model this doubles the wait.",
        )
    return opts


# ------------------------------------------------------------------ answer rendering


def _link_citations(text: str, key: str) -> str:
    """Turn [1] / [1, 2] markers into links that jump to the source below."""

    def repl(m: re.Match) -> str:
        nums = [n.strip() for n in m.group(1).split(",")]
        return "".join(f"[[{n}]](#src-{key}-{n})" for n in nums)

    return CITE_RE.sub(repl, text)


def render_answer(ans: dict, key: str) -> None:
    conf = ans.get("confidence")
    meta = ans.get("retrieval", {})
    st.caption(
        f"`{meta.get('strategy')}` · `{meta.get('mode')}` · reranker `{meta.get('reranker')}`"
    )

    if ans["status"] == "insufficient_context":
        st.warning("**Not enough information in the indexed docs to answer this.**")
        if ans["found"]:
            st.markdown("**Found**\n" + "\n".join(f"- {f}" for f in ans["found"]))
        if ans["missing"]:
            st.markdown("**Missing**\n" + "\n".join(f"- {m}" for m in ans["missing"]))
        if ans["suggested_documents"]:
            st.markdown(
                "**Worth checking manually:** "
                + ", ".join(f"`{d}`" for d in ans["suggested_documents"])
            )
    else:
        st.markdown(_link_citations(ans["answer"], key))
        cited = sorted({n for c in ans["claims"] for n in c["citations"]})
        by_n = {s["n"]: s for s in ans["sources"]}
        if cited:
            cols = st.columns(min(len(cited), 6))
            for i, n in enumerate(cited):
                if n not in by_n:
                    continue
                with cols[i % len(cols)].popover(f"[{n}] {by_n[n]['heading'][:28] or 'source'}"):
                    st.caption(by_n[n]["label"])
                    st.markdown(by_n[n]["text"])
        for u in ans["unsupported_citations"]:
            st.error(f"Claim {u['claim_id']} cites [{u['source']}] without support: {u['reason']}")

    if conf:
        render_confidence(conf)
    if ans["status"] == "answered" and ans["claims"]:
        render_claims(ans["claims"])
    render_sources(ans["sources"], key)
    if ans.get("candidates"):
        render_candidates(ans["candidates"])
    timings = {k: f"{v / 1000:.1f}s" for k, v in ans.get("timings_ms", {}).items()}
    st.caption("Timings: " + " · ".join(f"{k} {v}" for k, v in timings.items()))


def render_confidence(conf: dict) -> None:
    label = conf["label"]
    st.markdown(f"**Confidence** :{LABEL_COLOR[label]}-badge[{label} · {conf['composite']:.2f}]")
    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieval", f"{conf['retrieval']:.2f}", help="How relevant the top chunks are")
    c2.metric(
        "Citation coverage",
        f"{conf['citation_coverage']:.2f}",
        help="Share of claims backed by a verified citation",
    )
    c3.metric(
        "Completeness",
        f"{conf['completeness']:.2f}",
        help="Share of the question's parts the answer addresses",
    )
    if conf.get("question_parts"):
        with st.expander("Question parts"):
            st.dataframe(pd.DataFrame(conf["question_parts"]), hide_index=True)


def render_claims(claims: list[dict]) -> None:
    rows = []
    for c in claims:
        status = c["status"]
        rows.append(
            {
                "": STATUS_ICON[status],
                "Claim": c["text"],
                "Cites": " ".join(f"[{n}]" for n in c["citations"] + c["invalid_citations"]),
                "Verifier": " · ".join(
                    f"[{k['source']}] {k['verdict']}: {k['reason']}" for k in c["checks"]
                ),
            }
        )
    with st.expander(f"Claims and citation checks ({len(claims)})", expanded=False):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_sources(sources: list[dict], key: str) -> None:
    st.markdown("**Sources**")
    for s in sources:
        st.html(f'<div id="src-{key}-{s["n"]}"></div>')
        ranks = " · ".join(
            f"{name} #{s[f'{name}_rank']}"
            for name in ("dense", "sparse", "fused")
            if s.get(f"{name}_rank") is not None
        )
        with st.expander(f"[{s['n']}] {s['label']}  —  relevance {s['relevance']:.2f}"):
            st.caption(ranks + (f" · rerank {s['rerank_score']:.2f}" if s["rerank_score"] else ""))
            st.markdown(s["text"])


def render_candidates(cands: list[dict]) -> None:
    df = pd.DataFrame(cands)
    df["final_rank"] = [str(int(v)) if pd.notna(v) else "" for v in df["final_rank"]]
    df = df[
        [
            "final_rank",
            "rerank_score",
            "fused_rank",
            "fused_score",
            "dense_rank",
            "dense_score",
            "sparse_rank",
            "sparse_score",
            "label",
        ]
    ]
    with st.expander(f"Retrieval stages: all {len(cands)} reranker candidates", expanded=False):
        st.caption(
            "Fused order (weighted RRF of dense + BM25). *Final* is the rank after reranking; "
            "blank = not passed to the model. Dense = cosine, sparse = BM25 score."
        )
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "final_rank": st.column_config.TextColumn("Final"),
                "rerank_score": st.column_config.ProgressColumn(
                    "Rerank", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "fused_rank": "Fused #",
                "fused_score": st.column_config.NumberColumn("RRF", format="%.4f"),
                "dense_rank": "Dense #",
                "dense_score": st.column_config.NumberColumn("Cosine", format="%.3f"),
                "sparse_rank": "BM25 #",
                "sparse_score": st.column_config.NumberColumn("BM25", format="%.1f"),
                "label": "Source",
            },
        )


def render_overlap(a: dict, b: dict, name_a: str, name_b: str) -> None:
    ids_a = {s["chunk_id"] for s in a["sources"]}
    ids_b = {s["chunk_id"] for s in b["sources"]}
    only_a = [s["label"] for s in a["sources"] if s["chunk_id"] not in ids_b]
    only_b = [s["label"] for s in b["sources"] if s["chunk_id"] not in ids_a]
    st.info(
        f"{len(ids_a & ids_b)} of the chunks given to the model are shared. "
        f"Only {name_a}: {len(only_a)} · only {name_b}: {len(only_b)}"
    )
    if only_a or only_b:
        with st.expander("Chunks found by only one mode"):
            for name, labels in ((name_a, only_a), (name_b, only_b)):
                for label in labels:
                    st.markdown(f"- **{name}** · {label}")


# ------------------------------------------------------------------ tabs


def ask_tab(opts: dict) -> None:
    example = st.selectbox("Example questions", [""] + EXAMPLES, index=0)
    with st.form("ask"):
        question = st.text_area("Question", value=example, height=80)
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and question.strip():
        body = {k: opts[k] for k in ("strategy", "mode", "reranker", "top_n", "verify")}
        body |= {"question": question.strip(), "debug": True}
        runs = [(opts["mode"], body)]
        if opts["compare"]:
            other = "hybrid" if opts["mode"] == "dense" else "dense"
            runs.append((other, body | {"mode": other}))
        results = []
        for mode, b in runs:
            with st.spinner(f"Answering ({mode})… a local model can take up to a minute"):
                results.append((mode, api("POST", "/v1/ask", json=b)))
        st.session_state["results"] = results

    with st.expander("Audit an answer: check any text's citations against the sources"):
        with st.form("verify"):
            st.caption(
                "Sources are retrieved for the question and numbered [1]..[n] exactly as for "
                "Ask (open *Sources* after a normal answer to see them). Paste or edit an "
                "answer with [n] markers, e.g. one with a made-up detail, and every claim is "
                "checked against the source it cites."
            )
            audit_q = st.text_input("Question", value=question)
            audit_a = st.text_area("Answer to check", height=120)
            if st.form_submit_button("Check citations") and audit_q.strip() and audit_a.strip():
                body = {k: opts[k] for k in ("strategy", "mode", "reranker", "top_n")}
                with st.spinner("Verifying every cited claim…"):
                    res = api(
                        "POST", "/v1/verify", json=body | {"question": audit_q, "answer": audit_a}
                    )
                st.session_state["results"] = [("audit", res)]

    results = [(m, r) for m, r in st.session_state.get("results", []) if r]
    if not results:
        return
    st.divider()
    if len(results) == 1:
        render_answer(results[0][1], "a")
    else:
        (name_a, a), (name_b, b) = results
        render_overlap(a, b, name_a, name_b)
        for col, (name, ans), key in zip(st.columns(2), results, "ab", strict=True):
            with col:
                st.subheader(name)
                render_answer(ans, key)


def documents_tab() -> None:
    docs = api("GET", "/v1/documents")
    if docs is None:
        return
    st.caption(f"{len(docs)} documents indexed")
    query = st.text_input("Filter", placeholder="doc id or title")
    rows = [
        {
            "doc_id": d["doc_id"],
            "title": d["title"],
            "format": d["format"],
            "sections": d["sections"],
            "chars": d["chars"],
            **{f"{k} chunks": v for k, v in d["chunks"].items()},
        }
        for d in docs
        if query.lower() in (d["doc_id"] + " " + d["title"]).lower()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left, st.form("upload", clear_on_submit=True):
        st.markdown("**Add a document**")
        file = st.file_uploader("File", type=["md", "markdown", "txt", "html", "htm", "pdf"])
        doc_id = st.text_input("Document id (optional)", placeholder="uploads/<file name>")
        chosen = st.multiselect("Chunking strategies", STRATEGIES, default=STRATEGIES)
        if st.form_submit_button("Upload and index") and file is not None:
            with st.spinner("Parsing, chunking and embedding…"):
                res = api(
                    "POST",
                    "/v1/ingest",
                    files={"file": (file.name, file.getvalue())},
                    data={"strategies": ",".join(chosen)} | ({"doc_id": doc_id} if doc_id else {}),
                )
            if res:
                stats = ", ".join(
                    f"{k}: {v['chunks']} chunks ({v['duplicates']} duplicates skipped)"
                    for k, v in res["strategies"].items()
                )
                st.success(f"Indexed `{res['doc_id']}` in {res['elapsed_ms'] / 1000:.1f}s. {stats}")
                health.clear()

    with right, st.form("delete"):
        st.markdown("**Remove a document**")
        target = st.selectbox("Document", [d["doc_id"] for d in docs])
        sure = st.checkbox("Remove it from every index")
        if st.form_submit_button("Remove") and target and sure:
            res = api("DELETE", f"/v1/documents/{target}")
            if res:
                st.success(f"Removed `{target}` ({res['chunks_removed']} chunks)")
                health.clear()


def eval_tab() -> None:
    reports = sorted(
        (p for p in REPORTS_DIR.glob("*.md") if not p.name.startswith(("eval_", "latest"))),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        st.info(f"No reports in {REPORTS_DIR}. Run `uv run python scripts/run_eval.py`.")
        return
    names = [p.name for p in reports]
    default = names.index("local_answer_eval.md") if "local_answer_eval.md" in names else 0
    chosen = st.selectbox("Report", names, index=default)
    st.markdown((REPORTS_DIR / chosen).read_text())


def main() -> None:
    opts = sidebar()
    ask, documents, evaluation = st.tabs(["Ask", "Documents", "Evaluation"])
    with ask:
        ask_tab(opts)
    with documents:
        documents_tab()
    with evaluation:
        eval_tab()


main()
