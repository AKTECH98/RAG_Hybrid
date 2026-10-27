# Demo walkthrough (under 4 minutes)

A script for recording the demo. Every step below was rehearsed against the running system;
the expected results are what it actually returned.

## Before recording

```bash
docker compose up --build            # or: uvicorn + streamlit, see the README
```

- Open the dashboard at http://localhost:8501. The sidebar should say **API ok** and show the
  chunk counts.
- Keep the defaults: `recursive` chunks, `hybrid` retrieval, the `cross_encoder` reranker,
  5 chunks, and verification on.
- **Warm the cache.** Ask each question below once before recording. LLM responses are cached,
  so the recorded run answers instantly instead of taking 20–60 s on a local model. Say on
  camera that answers are cached for the recording.
- Have [`docs/demo/acme-rate-limits.md`](demo/acme-rate-limits.md) ready to upload. It is
  a short, fictional internal doc that is *not* in the corpus.
- If you rehearsed the upload, remove the document afterwards (Documents → Remove), so that
  the first question in step 2 declines again.

## 0:00–0:20 · The pitch

> "This is a RAG system over 103 technical docs: FastAPI, Pydantic, HTTPX and the OAuth RFCs.
> On a 69-question eval it answers 96% correctly with 97% citation accuracy, and it declines
> every question the docs can't answer. It even runs fully locally, on an 8 GB Jetson."

Show the README's results table, or the Evaluation tab.

## 0:20–1:00 · "I don't know", then ingest

1. Ask: **"What should a client do when the Acme search API returns a 429?"**
   - Expected: *Not enough information*. Retrieval confidence is about 0.06, and the "Missing"
     list names exactly what it couldn't find.
   - Say: "It doesn't guess. It says what's missing."
2. Go to **Documents → Add a document** and upload `acme-rate-limits.md` with all three
   strategies.
   - Expected: indexed in about 2 s, with chunk counts for each strategy.
3. Back on **Ask**, ask the same question again.
   - Expected: wait for the `Retry-After` seconds and add up to 1 s of random jitter `[1]`.
     The claim is ✅ supported, and confidence is **high**.

## 1:00–2:00 · Questions of increasing difficulty

1. **Lookup:** "How do I add custom headers to an HTTPException in FastAPI?"
   - Click the `[1]` citation link: the page jumps to the source chunk. Click a source button
     under the answer: a popover shows the exact text.
2. **Multi-hop across two RFCs:** "For a public client using the authorization code grant,
   what does RFC 9700 require to protect the code, and how long should the code live
   according to RFC 6749?"
   - Show that the citations point to both RFCs: PKCE from RFC 9700, and the 10-minute
     maximum code lifetime from RFC 6749.
3. **Ambiguous:** "What happens to extra fields?"
   - It picks the Pydantic reading, and its sources show which documentation it used.

## 2:00–2:45 · Catching a made-up claim

Open **Audit an answer**.

- Question: **"What is the default timeout in HTTPX?"**
- Answer to check:
  > HTTPX times out after 5 seconds of network inactivity by default [1]. Timed-out requests are
  > automatically retried three times before an error is raised [1].
- Expected:
  - Claim 1 is ✅ supported.
  - Claim 2 is ❌ unsupported, with the reason "the source does not mention any automatic
    retries".
  - Confidence drops to **medium**, and a red banner names the bad citation.
- Say: "Models often attach a real-looking citation to a sentence the source never says. Every
  claim is checked against the exact chunk it cites."

## 2:45–3:30 · Hybrid vs dense-only

1. Turn on **Compare with dense-only** in the sidebar and ask the RFC multi-hop question
   from 1:00 again.
   - Expected: the blue banner shows the two modes gave the model different chunks.
   - Dense-only never retrieves the RFC 9700 passage "Public clients MUST use PKCE", so it
     can't answer the first half of the question.
   - Open **Retrieval stages** in the hybrid column. That chunk has no dense rank at all.
     BM25 finds it (#6), fusion brings it into the pool at #15, and the cross-encoder moves it
     to **final #1**.
2. For a single-identifier case, set **Chunking strategy** to `fixed` and ask **"How do I mark
   a FastAPI endpoint as deprecated without removing it?"**
   - The `deprecated=True` chunk is **BM25 #1**, but it isn't in dense search's top 20 at
     all. Hybrid ranks it final #1.
   - Say: "Embeddings blur exact identifiers; BM25 matches them literally. Hybrid search
     matters most for exactly the questions developers ask."

## 3:30–3:50 · Close

Open the **Evaluation** tab:

> "Every number here comes from the eval suite in the repo: chunking strategies, retrieval
> modes, rerankers and embedders, plus the bugs the eval caught along the way."
