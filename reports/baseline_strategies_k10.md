# RAG evaluation report (baseline)

> **Baseline, superseded retrieval settings.** Run before the fetch-depth fix: hybrid searched 10 results per retriever (dense-only searched 20), fused with fixed 0.7/0.3 weights. This run is what exposed the depth and weighting issues; re-run with `scripts/run_eval.py --dense-weights 0.7,0.5,0.3` for current numbers.

- Run: 20261026-035717  
- Cases: 69 (lookup 48, multi_hop 8, no_answer 8, ambiguous 5)  
- Generation: `gpt-4o` · system verifier: `gpt-4o-mini` · eval judge: `gpt-4o`  
- Tokens billed this run: gpt-4o-mini: 1,628,125 in / 72,456 out, gpt-4o: 1,126,702 in / 68,228 out (≈ $3.79)

## Results by configuration

Config = chunking strategy / retrieval mode / reranker. Retrieval metrics are over answerable cases; *Pool recall* is evidence recall over the 20 candidates handed to the reranker.

| Config | Correct | Faithful | Cite acc | Recall@5 | MRR | Pool recall | No-ans acc | False abstain | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed/hybrid/llm | 93% | 98% | 96% | 89% | 76% | 94% | 100% | 0% | 31 |
| fixed/dense/llm | 92% | 96% | 94% | 83% | 70% | 90% | 100% | 3% | 15 |
| **recursive/hybrid/llm** | 96% | 98% | 97% | 95% | 87% | 96% | 100% | 2% | 6342 |
| recursive/dense/llm | 95% | 99% | 97% | 98% | 88% | 98% | 100% | 0% | 6178 |
| semantic/hybrid/llm | 95% | 99% | 95% | 91% | 73% | 97% | 100% | 2% | 6746 |
| semantic/dense/llm | 92% | 100% | 93% | 88% | 72% | 94% | 100% | 3% | 6538 |

## Correctness and Recall@5 by question type

| Config | ambiguous | lookup | multi_hop | no_answer |
|---|---:|---:|---:|---:|
| fixed/hybrid/llm | 90% · R 100% | 94% · R 92% | 81% · R 69% | 100% |
| fixed/dense/llm | 100% · R 100% | 92% · R 83% | 81% · R 69% | 100% |
| recursive/hybrid/llm | 100% · R 100% | 94% · R 98% | 100% · R 75% | 100% |
| recursive/dense/llm | 100% · R 100% | 93% · R 100% | 100% · R 88% | 100% |
| semantic/hybrid/llm | 100% · R 80% | 93% · R 94% | 100% · R 81% | 100% |
| semantic/dense/llm | 90% · R 80% | 91% · R 90% | 94% · R 81% | 100% |

## Hybrid vs dense-only: where retrieval changed

**fixed / llm**: hybrid found more evidence in 4 case(s), less in 0.
- ✅ L11 (0%→100%): If two FastAPI dependencies share a sub-dependency, is it called twice per request? How can I force it to run every time?
- ✅ L35 (0%→100%): What's the difference between alias, validation_alias and serialization_alias in Pydantic?
- ✅ L40 (0%→100%): Should OAuth clients still use the implicit grant according to RFC 9700?
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?

**recursive / llm**: hybrid found more evidence in 0 case(s), less in 3.
- ❌ L10 (100%→0%): What attributes does FastAPI's UploadFile provide?
- ❌ M03 (100%→50%): Which HTTP client library does FastAPI's TestClient require, and what is that library's default timeout?
- ❌ M04 (50%→0%): If a FastAPI request body model has an int field and the client sends "123" as a string, will FastAPI accept it? How could I make it strict?

**semantic / llm**: hybrid found more evidence in 2 case(s), less in 0.
- ✅ L11 (0%→100%): If two FastAPI dependencies share a sub-dependency, is it called twice per request? How can I force it to run every time?
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?


## Cases not fully correct in the best config (recursive/hybrid/llm)

| Case | Type | Score | Recall@5 | Judge |
|---|---|---:|---:|---|
| L10 | lookup | 0% | 0% | abstained |
| L22 | lookup | 50% | 100% | partially_correct: The system answer correctly states that HTTP/2 is not enabled by default and provides the correct command to install the optional dependencie |
| L27 | lookup | 50% | 100% | partially_correct: The system answer correctly mentions using the `NO_PROXY` environment variable to bypass the proxy for certain hosts, which aligns with the r |
| L38 | lookup | 50% | 100% | partially_correct: The system answer correctly explains how to limit the request body size using `max_body_size` on the application, router, mount, or route, an |
| L45 | lookup | 50% | 100% | partially_correct: The system answer correctly identifies the two client types as 'confidential' and 'public'. However, it lacks the explanation of what disting |
