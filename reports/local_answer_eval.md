# RAG evaluation report

## Findings (fully local run: nomic-embed-text + qwen3:4b-instruct on an 8 GB Jetson)

Compared with the OpenAI baseline (`baseline_strategies_k10.md`, gpt-4o generator, gpt-4o judge):

| | Correct | Faithful | Cite acc | Recall@5 | No-ans acc | False abstain |
|---|---:|---:|---:|---:|---:|---:|
| OpenAI, recursive/hybrid/llm (k=10) | 96% | 98% | 97% | 95% | 100% | 2% |
| Local, recursive/hybrid/cross_encoder | 96% | 88% | 93% | 91% | 100% | 2% |

- **Answer quality holds up.** 64 of 69 answers are judged fully correct, and all 8 unanswerable
  questions are declined. 3 are stopped by the retrieval-confidence gate and 5 by the model's
  INSUFFICIENT_CONTEXT marker, so both gates are needed.
- **Citation accuracy rose from 88% to 93% once skipped pairs were asked again.** On the first
  run, the 4B verifier returned no verdict for 16 of the 161 claim-source pairs, and those counted
  as unsupported. The verifier now asks again, up to two more times, for only the skipped pairs.
  After that every pair gets a verdict: 2 are unsupported (L48, A05) and the rest of the gap comes
  from "partial" verdicts.
- **The faithfulness score is noisy with a 4B judge.** L12 and L15 are short, correct answers whose
  citations the verifier confirmed, yet the eval judge scored them 0% faithful. Here the same model
  generates the answer and grades it, so treat Faithful as a lower bound, not as a
  like-for-like comparison with gpt-4o.
- **There is one false abstain.** A03 ("How do I turn off the docs?") is under-specified and gets
  low retrieval confidence (0.17), so the gate abstains.
- **Latency:** the median is 21 s per question and p90 is 54 s, including verification and the
  completeness check, on the Nano with one request at a time. These figures come from the first, uncached run; the
  run below reused cached LLM responses, so its p50 latency is not meaningful.

- Run: 20261026-175936  
- Cases: 69 (lookup 48, multi_hop 8, no_answer 8, ambiguous 5)  
- LLM provider: `ollama` · generation: `qwen3:4b-instruct-2507-q4_K_M` · system verifier: `qwen3:4b-instruct-2507-q4_K_M` · eval judge: `qwen3:4b-instruct-2507-q4_K_M`  
- Tokens this run: qwen3:4b-instruct-2507-q4_K_M: 8,440 in / 1,118 out (≈ $0.00)

## Results by configuration

Config = [embedder:]chunking strategy / retrieval mode[@dense weight] / reranker. Retrieval metrics are over answerable cases; *Pool recall* is evidence recall over the 20 candidates handed to the reranker; *Abstain AUC* is how well the top retrieval score separates answerable from unanswerable questions (1.0 = perfectly).

| Config | Correct | Faithful | Cite acc | Recall@5 | MRR | Pool recall | Abstain AUC | No-ans acc | False abstain | p50 ms* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **recursive/hybrid/cross_encoder** | 96% | 88% | 93% | 91% | 83% | 97% | 88% | 100% | 2% | 149 |

\* End-to-end latency including verification. LLM responses are cached, so configs re-run on cached prompts look faster; compare latency only within a fresh run.

## Correctness and Recall@5 by question type

| Config | ambiguous | lookup | multi_hop | no_answer |
|---|---:|---:|---:|---:|
| recursive/hybrid/cross_encoder | 80% · R 60% | 98% · R 98% | 94% · R 69% | 100% |

## Score separation (abstain-threshold calibration)

Top dense cosine and top calibrated relevance per question. A good "I don't know" threshold sits above the unanswerable max and below the answerable p10.

| Config | Cosine answerable p10 / p50 | Cosine unanswerable max | Relevance answerable p10 / p50 | Relevance unanswerable max |
|---|---:|---:|---:|---:|
| recursive/hybrid/cross_encoder | 0.75 / 0.79 | 0.75 | 0.53 / 0.99 | 0.99 |

## Cases not fully correct in the best config (recursive/hybrid/cross_encoder)

| Case | Type | Score | Recall@5 | Judge |
|---|---|---:|---:|---|
| L23 | lookup | 50% | 100% | partially_correct: The system answer correctly states that `max_connections=100` and `max_keepalive_connections=20`, which matches the key facts in the referenc |
| L41 | lookup | 50% | 0% | partially_correct: The system answer correctly identifies that clients should use PKCE code challenge methods that do not expose the PKCE verifier in the author |
| M04 | multi_hop | 50% | 0% | partially_correct: The system answer correctly identifies that FastAPI will accept a string '123' as an int field due to Pydantic's automatic conversion. Howeve |
| A03 | ambiguous | 0% | 0% | abstained |
