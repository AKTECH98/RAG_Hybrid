# RAG evaluation report

- Run: 20261026-155558  
- Cases: 69 (lookup 48, multi_hop 8, no_answer 8, ambiguous 5)  
- LLM provider: `openai` · generation: `gpt-4o` · system verifier: `gpt-4o-mini` · eval judge: `n/a (retrieval only)`  
- Tokens this run: none (all cached) (≈ $0.00)

## Results by configuration

Config = [embedder:]chunking strategy / retrieval mode[@dense weight] / reranker. Retrieval metrics are over answerable cases; *Pool recall* is evidence recall over the 20 candidates handed to the reranker; *Abstain AUC* is how well the top retrieval score separates answerable from unanswerable questions (1.0 = perfectly).

| Config | Recall@5 | MRR | Pool recall | Abstain AUC | p50 ms* |
|---|---:|---:|---:|---:|---:|
| openai:fixed/hybrid@0.7/none | 83% | 69% | 90% | 94% | 9 |
| openai:fixed/hybrid@0.5/none | 84% | 69% | 96% | 93% | 9 |
| openai:fixed/hybrid@0.3/none | 84% | 68% | 93% | 92% | 10 |
| openai:fixed/hybrid@0.7/cross_encoder | 80% | 64% | 90% | 87% | 1988 |
| openai:fixed/hybrid@0.5/cross_encoder | 82% | 69% | 96% | 80% | 1543 |
| openai:fixed/hybrid@0.3/cross_encoder | 83% | 68% | 93% | 80% | 1654 |
| openai:fixed/dense/none | 80% | 67% | 90% | 91% | 8 |
| openai:fixed/dense/cross_encoder | 80% | 64% | 90% | 87% | 1426 |
| openai:recursive/hybrid@0.7/none | 88% | 78% | 98% | 96% | 12 |
| openai:recursive/hybrid@0.5/none | 88% | 79% | 98% | 95% | 13 |
| openai:recursive/hybrid@0.3/none | 88% | 75% | 95% | 93% | 18 |
| openai:recursive/hybrid@0.7/cross_encoder | 91% | 83% | 98% | 92% | 1500 |
| openai:recursive/hybrid@0.5/cross_encoder | 91% | 83% | 98% | 89% | 1293 |
| openai:recursive/hybrid@0.3/cross_encoder | 91% | 83% | 95% | 86% | 2012 |
| openai:recursive/dense/none | 86% | 71% | 98% | 96% | 13 |
| openai:recursive/dense/cross_encoder | 91% | 83% | 98% | 92% | 1473 |
| openai:semantic/hybrid@0.7/none | 86% | 71% | 94% | 93% | 10 |
| openai:semantic/hybrid@0.5/none | 89% | 73% | 98% | 92% | 11 |
| openai:semantic/hybrid@0.3/none | 89% | 74% | 96% | 88% | 12 |
| openai:semantic/hybrid@0.7/cross_encoder | 87% | 74% | 94% | 87% | 2201 |
| openai:semantic/hybrid@0.5/cross_encoder | 88% | 75% | 98% | 85% | 1449 |
| openai:semantic/hybrid@0.3/cross_encoder | 86% | 75% | 96% | 88% | 2174 |
| openai:semantic/dense/none | 86% | 65% | 94% | 92% | 14 |
| openai:semantic/dense/cross_encoder | 87% | 74% | 94% | 87% | 1964 |
| local:fixed/hybrid@0.7/none | 81% | 63% | 92% | 89% | 8 |
| local:fixed/hybrid@0.5/none | 88% | 66% | 97% | 90% | 13 |
| local:fixed/hybrid@0.3/none | 88% | 70% | 93% | 87% | 11 |
| local:fixed/hybrid@0.7/cross_encoder | 81% | 68% | 92% | 93% | 2553 |
| local:fixed/hybrid@0.5/cross_encoder | 82% | 69% | 97% | 86% | 2052 |
| local:fixed/hybrid@0.3/cross_encoder | 83% | 68% | 93% | 86% | 2173 |
| local:fixed/dense/none | 72% | 61% | 92% | 93% | 44 |
| local:fixed/dense/cross_encoder | 81% | 68% | 92% | 93% | 1762 |
| local:recursive/hybrid@0.7/none | 88% | 75% | 98% | 95% | 10 |
| local:recursive/hybrid@0.5/none | 88% | 75% | 98% | 94% | 10 |
| local:recursive/hybrid@0.3/none | 88% | 72% | 95% | 94% | 10 |
| **local:recursive/hybrid@0.7/cross_encoder** | 93% | 83% | 98% | 93% | 1674 |
| local:recursive/hybrid@0.5/cross_encoder | 91% | 83% | 98% | 91% | 1810 |
| local:recursive/hybrid@0.3/cross_encoder | 91% | 83% | 95% | 91% | 1933 |
| local:recursive/dense/none | 84% | 72% | 98% | 96% | 12 |
| local:recursive/dense/cross_encoder | 93% | 83% | 98% | 93% | 1278 |
| local:semantic/hybrid@0.7/none | 80% | 65% | 95% | 93% | 5 |
| local:semantic/hybrid@0.5/none | 85% | 69% | 98% | 93% | 5 |
| local:semantic/hybrid@0.3/none | 86% | 72% | 94% | 91% | 6 |
| local:semantic/hybrid@0.7/cross_encoder | 88% | 74% | 95% | 91% | 1014 |
| local:semantic/hybrid@0.5/cross_encoder | 89% | 74% | 98% | 92% | 1021 |
| local:semantic/hybrid@0.3/cross_encoder | 86% | 72% | 94% | 92% | 1059 |
| local:semantic/dense/none | 77% | 64% | 95% | 92% | 4 |
| local:semantic/dense/cross_encoder | 88% | 74% | 95% | 91% | 1014 |

\* End-to-end latency including verification. LLM responses are cached, so configs re-run on cached prompts look faster; compare latency only within a fresh run.

## Correctness and Recall@5 by question type

| Config | ambiguous | lookup | multi_hop | no_answer |
|---|---:|---:|---:|---:|
| openai:fixed/hybrid@0.7/none | – · R 80% | – · R 85% | – · R 69% | – |
| openai:fixed/hybrid@0.5/none | – · R 80% | – · R 88% | – · R 62% | – |
| openai:fixed/hybrid@0.3/none | – · R 80% | – · R 88% | – · R 62% | – |
| openai:fixed/hybrid@0.7/cross_encoder | – · R 80% | – · R 83% | – · R 56% | – |
| openai:fixed/hybrid@0.5/cross_encoder | – · R 60% | – · R 88% | – · R 62% | – |
| openai:fixed/hybrid@0.3/cross_encoder | – · R 60% | – · R 88% | – · R 69% | – |
| openai:fixed/dense/none | – · R 80% | – · R 81% | – · R 69% | – |
| openai:fixed/dense/cross_encoder | – · R 80% | – · R 83% | – · R 56% | – |
| openai:recursive/hybrid@0.7/none | – · R 60% | – · R 94% | – · R 69% | – |
| openai:recursive/hybrid@0.5/none | – · R 60% | – · R 94% | – · R 69% | – |
| openai:recursive/hybrid@0.3/none | – · R 40% | – · R 96% | – · R 69% | – |
| openai:recursive/hybrid@0.7/cross_encoder | – · R 60% | – · R 98% | – · R 69% | – |
| openai:recursive/hybrid@0.5/cross_encoder | – · R 60% | – · R 98% | – · R 69% | – |
| openai:recursive/hybrid@0.3/cross_encoder | – · R 60% | – · R 98% | – · R 69% | – |
| openai:recursive/dense/none | – · R 60% | – · R 92% | – · R 69% | – |
| openai:recursive/dense/cross_encoder | – · R 60% | – · R 98% | – · R 69% | – |
| openai:semantic/hybrid@0.7/none | – · R 80% | – · R 90% | – · R 69% | – |
| openai:semantic/hybrid@0.5/none | – · R 80% | – · R 94% | – · R 69% | – |
| openai:semantic/hybrid@0.3/none | – · R 80% | – · R 94% | – · R 69% | – |
| openai:semantic/hybrid@0.7/cross_encoder | – · R 80% | – · R 90% | – · R 75% | – |
| openai:semantic/hybrid@0.5/cross_encoder | – · R 80% | – · R 92% | – · R 69% | – |
| openai:semantic/hybrid@0.3/cross_encoder | – · R 60% | – · R 92% | – · R 69% | – |
| openai:semantic/dense/none | – · R 80% | – · R 88% | – · R 81% | – |
| openai:semantic/dense/cross_encoder | – · R 80% | – · R 90% | – · R 75% | – |
| local:fixed/hybrid@0.7/none | – · R 80% | – · R 85% | – · R 56% | – |
| local:fixed/hybrid@0.5/none | – · R 100% | – · R 92% | – · R 62% | – |
| local:fixed/hybrid@0.3/none | – · R 80% | – · R 94% | – · R 62% | – |
| local:fixed/hybrid@0.7/cross_encoder | – · R 60% | – · R 85% | – · R 69% | – |
| local:fixed/hybrid@0.5/cross_encoder | – · R 60% | – · R 88% | – · R 62% | – |
| local:fixed/hybrid@0.3/cross_encoder | – · R 60% | – · R 88% | – · R 69% | – |
| local:fixed/dense/none | – · R 80% | – · R 73% | – · R 62% | – |
| local:fixed/dense/cross_encoder | – · R 60% | – · R 85% | – · R 69% | – |
| local:recursive/hybrid@0.7/none | – · R 40% | – · R 96% | – · R 69% | – |
| local:recursive/hybrid@0.5/none | – · R 40% | – · R 96% | – · R 69% | – |
| local:recursive/hybrid@0.3/none | – · R 40% | – · R 96% | – · R 69% | – |
| local:recursive/hybrid@0.7/cross_encoder | – · R 80% | – · R 98% | – · R 69% | – |
| local:recursive/hybrid@0.5/cross_encoder | – · R 60% | – · R 98% | – · R 69% | – |
| local:recursive/hybrid@0.3/cross_encoder | – · R 60% | – · R 98% | – · R 69% | – |
| local:recursive/dense/none | – · R 60% | – · R 90% | – · R 69% | – |
| local:recursive/dense/cross_encoder | – · R 80% | – · R 98% | – · R 69% | – |
| local:semantic/hybrid@0.7/none | – · R 60% | – · R 88% | – · R 50% | – |
| local:semantic/hybrid@0.5/none | – · R 60% | – · R 92% | – · R 62% | – |
| local:semantic/hybrid@0.3/none | – · R 60% | – · R 92% | – · R 69% | – |
| local:semantic/hybrid@0.7/cross_encoder | – · R 80% | – · R 96% | – · R 50% | – |
| local:semantic/hybrid@0.5/cross_encoder | – · R 80% | – · R 96% | – · R 56% | – |
| local:semantic/hybrid@0.3/cross_encoder | – · R 60% | – · R 94% | – · R 56% | – |
| local:semantic/dense/none | – · R 80% | – · R 81% | – · R 50% | – |
| local:semantic/dense/cross_encoder | – · R 80% | – · R 96% | – · R 50% | – |

## Score separation (abstain-threshold calibration)

Top dense cosine and top calibrated relevance per question. A good "I don't know" threshold sits above the unanswerable max and below the answerable p10.

| Config | Cosine answerable p10 / p50 | Cosine unanswerable max | Relevance answerable p10 / p50 | Relevance unanswerable max |
|---|---:|---:|---:|---:|
| openai:fixed/hybrid@0.7/none | 0.53 / 0.60 | 0.55 | 0.21 / 0.66 | 0.22 |
| openai:fixed/hybrid@0.5/none | 0.53 / 0.60 | 0.55 | 0.11 / 0.66 | 0.22 |
| openai:fixed/hybrid@0.3/none | 0.53 / 0.59 | 0.55 | 0.02 / 0.55 | 0.22 |
| openai:fixed/hybrid@0.7/cross_encoder | 0.53 / 0.60 | 0.55 | 0.41 / 0.78 | 0.54 |
| openai:fixed/hybrid@0.5/cross_encoder | 0.53 / 0.60 | 0.55 | 0.48 / 0.79 | 0.94 |
| openai:fixed/hybrid@0.3/cross_encoder | 0.53 / 0.59 | 0.55 | 0.48 / 0.82 | 0.94 |
| openai:fixed/dense/none | 0.53 / 0.60 | 0.55 | 0.24 / 0.74 | 0.42 |
| openai:fixed/dense/cross_encoder | 0.53 / 0.60 | 0.55 | 0.41 / 0.78 | 0.54 |
| openai:recursive/hybrid@0.7/none | 0.57 / 0.67 | 0.56 | 0.49 / 1.00 | 0.48 |
| openai:recursive/hybrid@0.5/none | 0.57 / 0.67 | 0.56 | 0.49 / 1.00 | 0.48 |
| openai:recursive/hybrid@0.3/none | 0.57 / 0.67 | 0.56 | 0.24 / 1.00 | 0.48 |
| openai:recursive/hybrid@0.7/cross_encoder | 0.57 / 0.67 | 0.56 | 0.53 / 0.99 | 0.74 |
| openai:recursive/hybrid@0.5/cross_encoder | 0.57 / 0.67 | 0.56 | 0.53 / 0.99 | 0.74 |
| openai:recursive/hybrid@0.3/cross_encoder | 0.57 / 0.67 | 0.56 | 0.53 / 0.99 | 0.78 |
| openai:recursive/dense/none | 0.57 / 0.67 | 0.56 | 0.55 / 1.00 | 0.49 |
| openai:recursive/dense/cross_encoder | 0.57 / 0.67 | 0.56 | 0.53 / 0.99 | 0.74 |
| openai:semantic/hybrid@0.7/none | 0.53 / 0.65 | 0.58 | 0.28 / 0.97 | 0.45 |
| openai:semantic/hybrid@0.5/none | 0.53 / 0.65 | 0.58 | 0.28 / 0.96 | 0.45 |
| openai:semantic/hybrid@0.3/none | 0.53 / 0.65 | 0.58 | 0.00 / 0.95 | 0.45 |
| openai:semantic/hybrid@0.7/cross_encoder | 0.53 / 0.65 | 0.58 | 0.50 / 0.90 | 0.72 |
| openai:semantic/hybrid@0.5/cross_encoder | 0.53 / 0.65 | 0.58 | 0.50 / 0.90 | 0.98 |
| openai:semantic/hybrid@0.3/cross_encoder | 0.53 / 0.65 | 0.58 | 0.50 / 0.91 | 0.98 |
| openai:semantic/dense/none | 0.53 / 0.65 | 0.58 | 0.28 / 1.00 | 0.60 |
| openai:semantic/dense/cross_encoder | 0.53 / 0.65 | 0.58 | 0.50 / 0.90 | 0.72 |
| local:fixed/hybrid@0.7/none | 0.72 / 0.77 | 0.73 | 0.00 / 0.78 | 0.34 |
| local:fixed/hybrid@0.5/none | 0.72 / 0.77 | 0.73 | 0.00 / 0.73 | 0.34 |
| local:fixed/hybrid@0.3/none | 0.71 / 0.77 | 0.73 | 0.00 / 0.57 | 0.34 |
| local:fixed/hybrid@0.7/cross_encoder | 0.72 / 0.77 | 0.73 | 0.45 / 0.78 | 0.49 |
| local:fixed/hybrid@0.5/cross_encoder | 0.72 / 0.77 | 0.73 | 0.48 / 0.78 | 0.98 |
| local:fixed/hybrid@0.3/cross_encoder | 0.71 / 0.77 | 0.73 | 0.48 / 0.78 | 0.98 |
| local:fixed/dense/none | 0.72 / 0.77 | 0.73 | 0.22 / 0.84 | 0.34 |
| local:fixed/dense/cross_encoder | 0.72 / 0.77 | 0.73 | 0.45 / 0.78 | 0.49 |
| local:recursive/hybrid@0.7/none | 0.76 / 0.80 | 0.76 | 0.58 / 1.00 | 0.76 |
| local:recursive/hybrid@0.5/none | 0.76 / 0.80 | 0.76 | 0.48 / 1.00 | 0.76 |
| local:recursive/hybrid@0.3/none | 0.76 / 0.80 | 0.76 | 0.30 / 1.00 | 0.76 |
| local:recursive/hybrid@0.7/cross_encoder | 0.76 / 0.80 | 0.76 | 0.52 / 1.00 | 0.88 |
| local:recursive/hybrid@0.5/cross_encoder | 0.76 / 0.80 | 0.76 | 0.52 / 1.00 | 0.88 |
| local:recursive/hybrid@0.3/cross_encoder | 0.76 / 0.80 | 0.76 | 0.50 / 1.00 | 0.88 |
| local:recursive/dense/none | 0.76 / 0.80 | 0.76 | 0.69 / 1.00 | 0.76 |
| local:recursive/dense/cross_encoder | 0.76 / 0.80 | 0.76 | 0.52 / 1.00 | 0.88 |
| local:semantic/hybrid@0.7/none | 0.73 / 0.78 | 0.74 | 0.25 / 1.00 | 0.26 |
| local:semantic/hybrid@0.5/none | 0.73 / 0.78 | 0.74 | 0.25 / 1.00 | 0.26 |
| local:semantic/hybrid@0.3/none | 0.72 / 0.78 | 0.74 | 0.10 / 1.00 | 0.26 |
| local:semantic/hybrid@0.7/cross_encoder | 0.73 / 0.78 | 0.74 | 0.47 / 0.96 | 0.62 |
| local:semantic/hybrid@0.5/cross_encoder | 0.73 / 0.78 | 0.74 | 0.51 / 0.96 | 0.62 |
| local:semantic/hybrid@0.3/cross_encoder | 0.72 / 0.78 | 0.74 | 0.49 / 0.97 | 0.62 |
| local:semantic/dense/none | 0.73 / 0.78 | 0.74 | 0.35 / 1.00 | 0.56 |
| local:semantic/dense/cross_encoder | 0.73 / 0.78 | 0.74 | 0.47 / 0.96 | 0.62 |

## Hybrid vs dense-only: where retrieval changed

**openai:fixed/hybrid@0.7/none** vs dense-only: hybrid found more evidence in 4 case(s), less in 2.
- ✅ L31 (0%→100%): Is it safe to use a mutable default like a list for a Pydantic model field?
- ✅ L35 (0%→100%): What's the difference between alias, validation_alias and serialization_alias in Pydantic?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ A04 (0%→100%): What happens to extra fields?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ A05 (100%→0%): How should I handle errors?

**openai:fixed/hybrid@0.5/none** vs dense-only: hybrid found more evidence in 5 case(s), less in 3.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L31 (0%→100%): Is it safe to use a mutable default like a list for a Pydantic model field?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ L44 (0%→100%): How does RFC 8725 recommend preventing one kind of JWT from being confused with another?
- ✅ A04 (0%→100%): What happens to extra fields?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ M01 (50%→0%): FastAPI's security tutorial uses the OAuth2 password flow. What does RFC 9700 say about using the resource owner password credentials grant?
- ❌ A05 (100%→0%): How should I handle errors?

**openai:fixed/hybrid@0.3/none** vs dense-only: hybrid found more evidence in 6 case(s), less in 4.
- ✅ L11 (0%→100%): If two FastAPI dependencies share a sub-dependency, is it called twice per request? How can I force it to run every time?
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L31 (0%→100%): Is it safe to use a mutable default like a list for a Pydantic model field?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ L44 (0%→100%): How does RFC 8725 recommend preventing one kind of JWT from being confused with another?
- ✅ A04 (0%→100%): What happens to extra fields?
- ❌ L27 (100%→0%): How do I tell HTTPX not to use the proxy for certain hosts?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ M01 (50%→0%): FastAPI's security tutorial uses the OAuth2 password flow. What does RFC 9700 say about using the resource owner password credentials grant?
- ❌ A05 (100%→0%): How should I handle errors?

**openai:fixed/hybrid@0.7/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 0.

**openai:fixed/hybrid@0.5/cross_encoder** vs dense-only: hybrid found more evidence in 4 case(s), less in 2.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L20 (0%→100%): What must the JSON response of an OAuth2 token endpoint contain in FastAPI's tutorial?
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?
- ✅ M02 (0%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ❌ L16 (100%→0%): How can I make FastAPI reject requests that send unexpected query parameters?
- ❌ A05 (100%→0%): How should I handle errors?

**openai:fixed/hybrid@0.3/cross_encoder** vs dense-only: hybrid found more evidence in 5 case(s), less in 2.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L20 (0%→100%): What must the JSON response of an OAuth2 token endpoint contain in FastAPI's tutorial?
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?
- ✅ M02 (0%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ✅ M07 (0%→50%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?
- ❌ L16 (100%→0%): How can I make FastAPI reject requests that send unexpected query parameters?
- ❌ A05 (100%→0%): How should I handle errors?

**openai:recursive/hybrid@0.7/none** vs dense-only: hybrid found more evidence in 1 case(s), less in 0.
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?

**openai:recursive/hybrid@0.5/none** vs dense-only: hybrid found more evidence in 1 case(s), less in 0.
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?

**openai:recursive/hybrid@0.3/none** vs dense-only: hybrid found more evidence in 2 case(s), less in 1.
- ✅ L44 (0%→100%): How does RFC 8725 recommend preventing one kind of JWT from being confused with another?
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?
- ❌ A04 (100%→0%): What happens to extra fields?

**openai:recursive/hybrid@0.7/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 0.

**openai:recursive/hybrid@0.5/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 0.

**openai:recursive/hybrid@0.3/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 0.

**openai:semantic/hybrid@0.7/none** vs dense-only: hybrid found more evidence in 2 case(s), less in 3.
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?
- ✅ A04 (0%→100%): What happens to extra fields?
- ❌ M01 (100%→50%): FastAPI's security tutorial uses the OAuth2 password flow. What does RFC 9700 say about using the resource owner password credentials grant?
- ❌ M07 (100%→50%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?
- ❌ A05 (100%→0%): How should I handle errors?

**openai:semantic/hybrid@0.5/none** vs dense-only: hybrid found more evidence in 6 case(s), less in 5.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L20 (0%→100%): What must the JSON response of an OAuth2 token endpoint contain in FastAPI's tutorial?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?
- ✅ M02 (50%→100%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ✅ A04 (0%→100%): What happens to extra fields?
- ❌ L10 (100%→0%): What attributes does FastAPI's UploadFile provide?
- ❌ M01 (100%→50%): FastAPI's security tutorial uses the OAuth2 password flow. What does RFC 9700 say about using the resource owner password credentials grant?
- ❌ M03 (100%→50%): Which HTTP client library does FastAPI's TestClient require, and what is that library's default timeout?
- ❌ M07 (100%→50%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?
- ❌ A05 (100%→0%): How should I handle errors?

**openai:semantic/hybrid@0.3/none** vs dense-only: hybrid found more evidence in 6 case(s), less in 5.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L20 (0%→100%): What must the JSON response of an OAuth2 token endpoint contain in FastAPI's tutorial?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?
- ✅ M02 (50%→100%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ✅ A04 (0%→100%): What happens to extra fields?
- ❌ L10 (100%→0%): What attributes does FastAPI's UploadFile provide?
- ❌ M01 (100%→50%): FastAPI's security tutorial uses the OAuth2 password flow. What does RFC 9700 say about using the resource owner password credentials grant?
- ❌ M03 (100%→50%): Which HTTP client library does FastAPI's TestClient require, and what is that library's default timeout?
- ❌ M07 (100%→50%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?
- ❌ A05 (100%→0%): How should I handle errors?

**openai:semantic/hybrid@0.7/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 0.

**openai:semantic/hybrid@0.5/cross_encoder** vs dense-only: hybrid found more evidence in 1 case(s), less in 1.
- ✅ L11 (0%→100%): If two FastAPI dependencies share a sub-dependency, is it called twice per request? How can I force it to run every time?
- ❌ M02 (100%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?

**openai:semantic/hybrid@0.3/cross_encoder** vs dense-only: hybrid found more evidence in 2 case(s), less in 3.
- ✅ L11 (0%→100%): If two FastAPI dependencies share a sub-dependency, is it called twice per request? How can I force it to run every time?
- ✅ L47 (0%→100%): Can an authorization server issue a refresh token in the implicit grant flow per RFC 6749?
- ❌ L31 (100%→0%): Is it safe to use a mutable default like a list for a Pydantic model field?
- ❌ M02 (100%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ❌ A05 (100%→0%): How should I handle errors?

**local:fixed/hybrid@0.7/none** vs dense-only: hybrid found more evidence in 7 case(s), less in 2.
- ✅ L04 (0%→100%): I have both /users/me and /users/{user_id} routes in FastAPI. Does the order I declare them in matter?
- ✅ L06 (0%→100%): Which numeric validation parameters can be used with Query and Path in FastAPI?
- ✅ L11 (0%→100%): If two FastAPI dependencies share a sub-dependency, is it called twice per request? How can I force it to run every time?
- ✅ L31 (0%→100%): Is it safe to use a mutable default like a list for a Pydantic model field?
- ✅ L34 (0%→100%): What happens by default when a Pydantic model receives extra fields, and what are the options?
- ✅ L35 (0%→100%): What's the difference between alias, validation_alias and serialization_alias in Pydantic?
- ✅ L40 (0%→100%): Should OAuth clients still use the implicit grant according to RFC 9700?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ M07 (50%→0%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?

**local:fixed/hybrid@0.5/none** vs dense-only: hybrid found more evidence in 12 case(s), less in 2.
- ✅ L04 (0%→100%): I have both /users/me and /users/{user_id} routes in FastAPI. Does the order I declare them in matter?
- ✅ L06 (0%→100%): Which numeric validation parameters can be used with Query and Path in FastAPI?
- ✅ L11 (0%→100%): If two FastAPI dependencies share a sub-dependency, is it called twice per request? How can I force it to run every time?
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L16 (0%→100%): How can I make FastAPI reject requests that send unexpected query parameters?
- ✅ L20 (0%→100%): What must the JSON response of an OAuth2 token endpoint contain in FastAPI's tutorial?
- ✅ L31 (0%→100%): Is it safe to use a mutable default like a list for a Pydantic model field?
- ✅ L34 (0%→100%): What happens by default when a Pydantic model receives extra fields, and what are the options?
- ✅ L35 (0%→100%): What's the difference between alias, validation_alias and serialization_alias in Pydantic?
- ✅ L40 (0%→100%): Should OAuth clients still use the implicit grant according to RFC 9700?
- ✅ M02 (0%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ✅ A04 (0%→100%): What happens to extra fields?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ M07 (50%→0%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?

**local:fixed/hybrid@0.3/none** vs dense-only: hybrid found more evidence in 13 case(s), less in 3.
- ✅ L04 (0%→100%): I have both /users/me and /users/{user_id} routes in FastAPI. Does the order I declare them in matter?
- ✅ L06 (0%→100%): Which numeric validation parameters can be used with Query and Path in FastAPI?
- ✅ L11 (0%→100%): If two FastAPI dependencies share a sub-dependency, is it called twice per request? How can I force it to run every time?
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L16 (0%→100%): How can I make FastAPI reject requests that send unexpected query parameters?
- ✅ L20 (0%→100%): What must the JSON response of an OAuth2 token endpoint contain in FastAPI's tutorial?
- ✅ L31 (0%→100%): Is it safe to use a mutable default like a list for a Pydantic model field?
- ✅ L34 (0%→100%): What happens by default when a Pydantic model receives extra fields, and what are the options?
- ✅ L35 (0%→100%): What's the difference between alias, validation_alias and serialization_alias in Pydantic?
- ✅ L40 (0%→100%): Should OAuth clients still use the implicit grant according to RFC 9700?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ M02 (0%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ✅ A04 (0%→100%): What happens to extra fields?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ M07 (50%→0%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?
- ❌ A05 (100%→0%): How should I handle errors?

**local:fixed/hybrid@0.7/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 0.

**local:fixed/hybrid@0.5/cross_encoder** vs dense-only: hybrid found more evidence in 1 case(s), less in 1.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ❌ M03 (100%→50%): Which HTTP client library does FastAPI's TestClient require, and what is that library's default timeout?

**local:fixed/hybrid@0.3/cross_encoder** vs dense-only: hybrid found more evidence in 2 case(s), less in 1.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ M02 (0%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ❌ M03 (100%→50%): Which HTTP client library does FastAPI's TestClient require, and what is that library's default timeout?

**local:recursive/hybrid@0.7/none** vs dense-only: hybrid found more evidence in 3 case(s), less in 1.
- ✅ L06 (0%→100%): Which numeric validation parameters can be used with Query and Path in FastAPI?
- ✅ L12 (0%→100%): In a FastAPI dependency that uses yield, when does the code after the yield statement run?
- ✅ L40 (0%→100%): Should OAuth clients still use the implicit grant according to RFC 9700?
- ❌ A03 (100%→0%): How do I turn off the docs?

**local:recursive/hybrid@0.5/none** vs dense-only: hybrid found more evidence in 3 case(s), less in 1.
- ✅ L06 (0%→100%): Which numeric validation parameters can be used with Query and Path in FastAPI?
- ✅ L12 (0%→100%): In a FastAPI dependency that uses yield, when does the code after the yield statement run?
- ✅ L40 (0%→100%): Should OAuth clients still use the implicit grant according to RFC 9700?
- ❌ A03 (100%→0%): How do I turn off the docs?

**local:recursive/hybrid@0.3/none** vs dense-only: hybrid found more evidence in 4 case(s), less in 2.
- ✅ L06 (0%→100%): Which numeric validation parameters can be used with Query and Path in FastAPI?
- ✅ L12 (0%→100%): In a FastAPI dependency that uses yield, when does the code after the yield statement run?
- ✅ L40 (0%→100%): Should OAuth clients still use the implicit grant according to RFC 9700?
- ✅ M02 (50%→100%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ❌ M07 (100%→50%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?
- ❌ A03 (100%→0%): How do I turn off the docs?

**local:recursive/hybrid@0.7/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 0.

**local:recursive/hybrid@0.5/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 1.
- ❌ A03 (100%→0%): How do I turn off the docs?

**local:recursive/hybrid@0.3/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 1.
- ❌ A03 (100%→0%): How do I turn off the docs?

**local:semantic/hybrid@0.7/none** vs dense-only: hybrid found more evidence in 5 case(s), less in 3.
- ✅ L06 (0%→100%): Which numeric validation parameters can be used with Query and Path in FastAPI?
- ✅ L16 (0%→100%): How can I make FastAPI reject requests that send unexpected query parameters?
- ✅ L35 (0%→100%): What's the difference between alias, validation_alias and serialization_alias in Pydantic?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ M02 (0%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ M07 (100%→50%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?
- ❌ A05 (100%→0%): How should I handle errors?

**local:semantic/hybrid@0.5/none** vs dense-only: hybrid found more evidence in 8 case(s), less in 3.
- ✅ L06 (0%→100%): Which numeric validation parameters can be used with Query and Path in FastAPI?
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L16 (0%→100%): How can I make FastAPI reject requests that send unexpected query parameters?
- ✅ L18 (0%→100%): How can a FastAPI Server-Sent Events endpoint resume a stream after the browser reconnects?
- ✅ L35 (0%→100%): What's the difference between alias, validation_alias and serialization_alias in Pydantic?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ M02 (0%→100%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ✅ M05 (50%→100%): FastAPI's PATCH tutorial relies on exclude_unset. How does Pydantic decide which fields are 'unset'?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ M07 (100%→50%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?
- ❌ A05 (100%→0%): How should I handle errors?

**local:semantic/hybrid@0.3/none** vs dense-only: hybrid found more evidence in 9 case(s), less in 3.
- ✅ L06 (0%→100%): Which numeric validation parameters can be used with Query and Path in FastAPI?
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L16 (0%→100%): How can I make FastAPI reject requests that send unexpected query parameters?
- ✅ L18 (0%→100%): How can a FastAPI Server-Sent Events endpoint resume a stream after the browser reconnects?
- ✅ L35 (0%→100%): What's the difference between alias, validation_alias and serialization_alias in Pydantic?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ M01 (0%→50%): FastAPI's security tutorial uses the OAuth2 password flow. What does RFC 9700 say about using the resource owner password credentials grant?
- ✅ M02 (0%→100%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ✅ M05 (50%→100%): FastAPI's PATCH tutorial relies on exclude_unset. How does Pydantic decide which fields are 'unset'?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ M07 (100%→50%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?
- ❌ A05 (100%→0%): How should I handle errors?

**local:semantic/hybrid@0.7/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 0.

**local:semantic/hybrid@0.5/cross_encoder** vs dense-only: hybrid found more evidence in 1 case(s), less in 1.
- ✅ M08 (0%→100%): If I want my FastAPI app to tolerate slow upstream calls made with HTTPX, which timeout types can I tune and what exception tells me the server took too long to send the response body?
- ❌ M04 (50%→0%): If a FastAPI request body model has an int field and the client sends "123" as a string, will FastAPI accept it? How could I make it strict?

**local:semantic/hybrid@0.3/cross_encoder** vs dense-only: hybrid found more evidence in 1 case(s), less in 3.
- ✅ M08 (0%→100%): If I want my FastAPI app to tolerate slow upstream calls made with HTTPX, which timeout types can I tune and what exception tells me the server took too long to send the response body?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ M04 (50%→0%): If a FastAPI request body model has an int field and the client sends "123" as a string, will FastAPI accept it? How could I make it strict?
- ❌ A05 (100%→0%): How should I handle errors?


## Evidence missed in the best config (local:recursive/hybrid@0.7/cross_encoder)

- L41 (lookup, recall 0%): Which PKCE code challenge method does RFC 9700 say clients should use, and why?
- M01 (multi_hop, recall 50%): FastAPI's security tutorial uses the OAuth2 password flow. What does RFC 9700 say about using the resource owner password credentials grant?
- M02 (multi_hop, recall 50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- M03 (multi_hop, recall 50%): Which HTTP client library does FastAPI's TestClient require, and what is that library's default timeout?
- M04 (multi_hop, recall 0%): If a FastAPI request body model has an int field and the client sends "123" as a string, will FastAPI accept it? How could I make it strict?
- A05 (ambiguous, recall 0%): How should I handle errors?
