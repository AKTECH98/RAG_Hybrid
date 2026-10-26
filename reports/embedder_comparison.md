# RAG evaluation report

- Run: 20261026-160148  
- Cases: 69 (lookup 48, multi_hop 8, no_answer 8, ambiguous 5)  
- LLM provider: `ollama` · generation: `qwen3:4b-instruct-2507-q4_K_M` · system verifier: `qwen3:4b-instruct-2507-q4_K_M` · eval judge: `n/a (retrieval only)`  
- Tokens this run: none (all cached) (≈ $0.00)

## Results by configuration

Config = [embedder:]chunking strategy / retrieval mode[@dense weight] / reranker. Retrieval metrics are over answerable cases; *Pool recall* is evidence recall over the 20 candidates handed to the reranker; *Abstain AUC* is how well the top retrieval score separates answerable from unanswerable questions (1.0 = perfectly).

| Config | Recall@5 | MRR | Pool recall | Abstain AUC | p50 ms* |
|---|---:|---:|---:|---:|---:|
| openai:fixed/hybrid/none | 84% | 69% | 96% | 93% | 5 |
| openai:fixed/hybrid/cross_encoder | 82% | 69% | 96% | 80% | 652 |
| openai:fixed/dense/none | 80% | 67% | 90% | 91% | 5 |
| openai:fixed/dense/cross_encoder | 80% | 64% | 90% | 87% | 629 |
| openai:recursive/hybrid/none | 88% | 79% | 98% | 95% | 7 |
| openai:recursive/hybrid/cross_encoder | 91% | 83% | 98% | 89% | 602 |
| openai:recursive/dense/none | 86% | 71% | 98% | 96% | 6 |
| openai:recursive/dense/cross_encoder | 91% | 83% | 98% | 92% | 604 |
| openai:semantic/hybrid/none | 89% | 73% | 98% | 92% | 5 |
| openai:semantic/hybrid/cross_encoder | 88% | 75% | 98% | 85% | 606 |
| openai:semantic/dense/none | 86% | 65% | 94% | 92% | 5 |
| openai:semantic/dense/cross_encoder | 87% | 74% | 94% | 87% | 590 |
| local:fixed/hybrid/none | 88% | 66% | 97% | 90% | 4 |
| local:fixed/hybrid/cross_encoder | 82% | 69% | 97% | 86% | 584 |
| local:fixed/dense/none | 72% | 61% | 92% | 93% | 4 |
| local:fixed/dense/cross_encoder | 81% | 68% | 92% | 93% | 579 |
| local:recursive/hybrid/none | 88% | 75% | 98% | 94% | 6 |
| local:recursive/hybrid/cross_encoder | 91% | 83% | 98% | 91% | 551 |
| local:recursive/dense/none | 84% | 72% | 98% | 96% | 4 |
| **local:recursive/dense/cross_encoder** | 93% | 83% | 98% | 93% | 576 |
| local:semantic/hybrid/none | 85% | 69% | 98% | 93% | 5 |
| local:semantic/hybrid/cross_encoder | 89% | 74% | 98% | 92% | 661 |
| local:semantic/dense/none | 77% | 64% | 95% | 92% | 4 |
| local:semantic/dense/cross_encoder | 88% | 74% | 95% | 91% | 609 |
| ollama:fixed/hybrid/none | 80% | 65% | 93% | 61% | 59 |
| ollama:fixed/hybrid/cross_encoder | 80% | 67% | 93% | 86% | 605 |
| ollama:fixed/dense/none | 76% | 62% | 87% | 49% | 4 |
| ollama:fixed/dense/cross_encoder | 80% | 67% | 87% | 84% | 607 |
| ollama:recursive/hybrid/none | 89% | 76% | 97% | 49% | 7 |
| ollama:recursive/hybrid/cross_encoder | 91% | 83% | 97% | 89% | 560 |
| ollama:recursive/dense/none | 85% | 70% | 98% | 49% | 4 |
| ollama:recursive/dense/cross_encoder | 90% | 82% | 98% | 89% | 614 |
| ollama:semantic/hybrid/none | 83% | 65% | 95% | 61% | 5 |
| ollama:semantic/hybrid/cross_encoder | 90% | 73% | 95% | 88% | 619 |
| ollama:semantic/dense/none | 70% | 53% | 92% | 49% | 4 |
| ollama:semantic/dense/cross_encoder | 88% | 72% | 92% | 88% | 602 |

\* End-to-end latency including verification. LLM responses are cached, so configs re-run on cached prompts look faster; compare latency only within a fresh run.

## Correctness and Recall@5 by question type

| Config | ambiguous | lookup | multi_hop | no_answer |
|---|---:|---:|---:|---:|
| openai:fixed/hybrid/none | – · R 80% | – · R 88% | – · R 62% | – |
| openai:fixed/hybrid/cross_encoder | – · R 60% | – · R 88% | – · R 62% | – |
| openai:fixed/dense/none | – · R 80% | – · R 81% | – · R 69% | – |
| openai:fixed/dense/cross_encoder | – · R 80% | – · R 83% | – · R 56% | – |
| openai:recursive/hybrid/none | – · R 60% | – · R 94% | – · R 69% | – |
| openai:recursive/hybrid/cross_encoder | – · R 60% | – · R 98% | – · R 69% | – |
| openai:recursive/dense/none | – · R 60% | – · R 92% | – · R 69% | – |
| openai:recursive/dense/cross_encoder | – · R 60% | – · R 98% | – · R 69% | – |
| openai:semantic/hybrid/none | – · R 80% | – · R 94% | – · R 69% | – |
| openai:semantic/hybrid/cross_encoder | – · R 80% | – · R 92% | – · R 69% | – |
| openai:semantic/dense/none | – · R 80% | – · R 88% | – · R 81% | – |
| openai:semantic/dense/cross_encoder | – · R 80% | – · R 90% | – · R 75% | – |
| local:fixed/hybrid/none | – · R 100% | – · R 92% | – · R 62% | – |
| local:fixed/hybrid/cross_encoder | – · R 60% | – · R 88% | – · R 62% | – |
| local:fixed/dense/none | – · R 80% | – · R 73% | – · R 62% | – |
| local:fixed/dense/cross_encoder | – · R 60% | – · R 85% | – · R 69% | – |
| local:recursive/hybrid/none | – · R 40% | – · R 96% | – · R 69% | – |
| local:recursive/hybrid/cross_encoder | – · R 60% | – · R 98% | – · R 69% | – |
| local:recursive/dense/none | – · R 60% | – · R 90% | – · R 69% | – |
| local:recursive/dense/cross_encoder | – · R 80% | – · R 98% | – · R 69% | – |
| local:semantic/hybrid/none | – · R 60% | – · R 92% | – · R 62% | – |
| local:semantic/hybrid/cross_encoder | – · R 80% | – · R 96% | – · R 56% | – |
| local:semantic/dense/none | – · R 80% | – · R 81% | – · R 50% | – |
| local:semantic/dense/cross_encoder | – · R 80% | – · R 96% | – · R 50% | – |
| ollama:fixed/hybrid/none | – · R 60% | – · R 85% | – · R 62% | – |
| ollama:fixed/hybrid/cross_encoder | – · R 60% | – · R 85% | – · R 56% | – |
| ollama:fixed/dense/none | – · R 60% | – · R 81% | – · R 56% | – |
| ollama:fixed/dense/cross_encoder | – · R 80% | – · R 81% | – · R 69% | – |
| ollama:recursive/hybrid/none | – · R 60% | – · R 96% | – · R 69% | – |
| ollama:recursive/hybrid/cross_encoder | – · R 60% | – · R 98% | – · R 69% | – |
| ollama:recursive/dense/none | – · R 60% | – · R 92% | – · R 62% | – |
| ollama:recursive/dense/cross_encoder | – · R 60% | – · R 98% | – · R 62% | – |
| ollama:semantic/hybrid/none | – · R 80% | – · R 88% | – · R 56% | – |
| ollama:semantic/hybrid/cross_encoder | – · R 100% | – · R 94% | – · R 62% | – |
| ollama:semantic/dense/none | – · R 60% | – · R 75% | – · R 44% | – |
| ollama:semantic/dense/cross_encoder | – · R 100% | – · R 92% | – · R 62% | – |

## Score separation (abstain-threshold calibration)

Top dense cosine and top calibrated relevance per question. A good "I don't know" threshold sits above the unanswerable max and below the answerable p10.

| Config | Cosine answerable p10 / p50 | Cosine unanswerable max | Relevance answerable p10 / p50 | Relevance unanswerable max |
|---|---:|---:|---:|---:|
| openai:fixed/hybrid/none | 0.53 / 0.60 | 0.55 | 0.11 / 0.66 | 0.22 |
| openai:fixed/hybrid/cross_encoder | 0.53 / 0.60 | 0.55 | 0.48 / 0.79 | 0.94 |
| openai:fixed/dense/none | 0.53 / 0.60 | 0.55 | 0.24 / 0.74 | 0.42 |
| openai:fixed/dense/cross_encoder | 0.53 / 0.60 | 0.55 | 0.41 / 0.78 | 0.54 |
| openai:recursive/hybrid/none | 0.57 / 0.67 | 0.56 | 0.49 / 1.00 | 0.48 |
| openai:recursive/hybrid/cross_encoder | 0.57 / 0.67 | 0.56 | 0.53 / 0.99 | 0.74 |
| openai:recursive/dense/none | 0.57 / 0.67 | 0.56 | 0.55 / 1.00 | 0.49 |
| openai:recursive/dense/cross_encoder | 0.57 / 0.67 | 0.56 | 0.53 / 0.99 | 0.74 |
| openai:semantic/hybrid/none | 0.53 / 0.65 | 0.58 | 0.28 / 0.96 | 0.45 |
| openai:semantic/hybrid/cross_encoder | 0.53 / 0.65 | 0.58 | 0.50 / 0.90 | 0.98 |
| openai:semantic/dense/none | 0.53 / 0.65 | 0.58 | 0.28 / 1.00 | 0.60 |
| openai:semantic/dense/cross_encoder | 0.53 / 0.65 | 0.58 | 0.50 / 0.90 | 0.72 |
| local:fixed/hybrid/none | 0.72 / 0.77 | 0.73 | 0.00 / 0.73 | 0.34 |
| local:fixed/hybrid/cross_encoder | 0.72 / 0.77 | 0.73 | 0.48 / 0.78 | 0.98 |
| local:fixed/dense/none | 0.72 / 0.77 | 0.73 | 0.22 / 0.84 | 0.34 |
| local:fixed/dense/cross_encoder | 0.72 / 0.77 | 0.73 | 0.45 / 0.78 | 0.49 |
| local:recursive/hybrid/none | 0.76 / 0.80 | 0.76 | 0.48 / 1.00 | 0.76 |
| local:recursive/hybrid/cross_encoder | 0.76 / 0.80 | 0.76 | 0.52 / 1.00 | 0.88 |
| local:recursive/dense/none | 0.76 / 0.80 | 0.76 | 0.69 / 1.00 | 0.76 |
| local:recursive/dense/cross_encoder | 0.76 / 0.80 | 0.76 | 0.52 / 1.00 | 0.88 |
| local:semantic/hybrid/none | 0.73 / 0.78 | 0.74 | 0.25 / 1.00 | 0.26 |
| local:semantic/hybrid/cross_encoder | 0.73 / 0.78 | 0.74 | 0.51 / 0.96 | 0.62 |
| local:semantic/dense/none | 0.73 / 0.78 | 0.74 | 0.35 / 1.00 | 0.56 |
| local:semantic/dense/cross_encoder | 0.73 / 0.78 | 0.74 | 0.47 / 0.96 | 0.62 |
| ollama:fixed/hybrid/none | 0.71 / 0.77 | 0.72 | 1.00 / 1.00 | 1.00 |
| ollama:fixed/hybrid/cross_encoder | 0.71 / 0.77 | 0.72 | 0.88 / 0.99 | 0.99 |
| ollama:fixed/dense/none | 0.71 / 0.77 | 0.72 | 1.00 / 1.00 | 1.00 |
| ollama:fixed/dense/cross_encoder | 0.71 / 0.77 | 0.72 | 0.88 / 0.99 | 0.99 |
| ollama:recursive/hybrid/none | 0.75 / 0.79 | 0.75 | 1.00 / 1.00 | 1.00 |
| ollama:recursive/hybrid/cross_encoder | 0.75 / 0.79 | 0.75 | 0.95 / 1.00 | 1.00 |
| ollama:recursive/dense/none | 0.75 / 0.79 | 0.75 | 1.00 / 1.00 | 1.00 |
| ollama:recursive/dense/cross_encoder | 0.75 / 0.79 | 0.75 | 0.95 / 1.00 | 1.00 |
| ollama:semantic/hybrid/none | 0.72 / 0.78 | 0.75 | 1.00 / 1.00 | 1.00 |
| ollama:semantic/hybrid/cross_encoder | 0.72 / 0.78 | 0.75 | 0.92 / 1.00 | 0.99 |
| ollama:semantic/dense/none | 0.72 / 0.78 | 0.75 | 1.00 / 1.00 | 1.00 |
| ollama:semantic/dense/cross_encoder | 0.72 / 0.78 | 0.75 | 0.92 / 1.00 | 0.99 |

## Hybrid vs dense-only: where retrieval changed

**openai:fixed/hybrid/none** vs dense-only: hybrid found more evidence in 5 case(s), less in 3.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L31 (0%→100%): Is it safe to use a mutable default like a list for a Pydantic model field?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ L44 (0%→100%): How does RFC 8725 recommend preventing one kind of JWT from being confused with another?
- ✅ A04 (0%→100%): What happens to extra fields?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ M01 (50%→0%): FastAPI's security tutorial uses the OAuth2 password flow. What does RFC 9700 say about using the resource owner password credentials grant?
- ❌ A05 (100%→0%): How should I handle errors?

**openai:fixed/hybrid/cross_encoder** vs dense-only: hybrid found more evidence in 4 case(s), less in 2.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L20 (0%→100%): What must the JSON response of an OAuth2 token endpoint contain in FastAPI's tutorial?
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?
- ✅ M02 (0%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ❌ L16 (100%→0%): How can I make FastAPI reject requests that send unexpected query parameters?
- ❌ A05 (100%→0%): How should I handle errors?

**openai:recursive/hybrid/none** vs dense-only: hybrid found more evidence in 1 case(s), less in 0.
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?

**openai:recursive/hybrid/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 0.

**openai:semantic/hybrid/none** vs dense-only: hybrid found more evidence in 6 case(s), less in 5.
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

**openai:semantic/hybrid/cross_encoder** vs dense-only: hybrid found more evidence in 1 case(s), less in 1.
- ✅ L11 (0%→100%): If two FastAPI dependencies share a sub-dependency, is it called twice per request? How can I force it to run every time?
- ❌ M02 (100%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?

**local:fixed/hybrid/none** vs dense-only: hybrid found more evidence in 12 case(s), less in 2.
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

**local:fixed/hybrid/cross_encoder** vs dense-only: hybrid found more evidence in 1 case(s), less in 1.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ❌ M03 (100%→50%): Which HTTP client library does FastAPI's TestClient require, and what is that library's default timeout?

**local:recursive/hybrid/none** vs dense-only: hybrid found more evidence in 3 case(s), less in 1.
- ✅ L06 (0%→100%): Which numeric validation parameters can be used with Query and Path in FastAPI?
- ✅ L12 (0%→100%): In a FastAPI dependency that uses yield, when does the code after the yield statement run?
- ✅ L40 (0%→100%): Should OAuth clients still use the implicit grant according to RFC 9700?
- ❌ A03 (100%→0%): How do I turn off the docs?

**local:recursive/hybrid/cross_encoder** vs dense-only: hybrid found more evidence in 0 case(s), less in 1.
- ❌ A03 (100%→0%): How do I turn off the docs?

**local:semantic/hybrid/none** vs dense-only: hybrid found more evidence in 8 case(s), less in 3.
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

**local:semantic/hybrid/cross_encoder** vs dense-only: hybrid found more evidence in 1 case(s), less in 1.
- ✅ M08 (0%→100%): If I want my FastAPI app to tolerate slow upstream calls made with HTTPX, which timeout types can I tune and what exception tells me the server took too long to send the response body?
- ❌ M04 (50%→0%): If a FastAPI request body model has an int field and the client sends "123" as a string, will FastAPI accept it? How could I make it strict?

**ollama:fixed/hybrid/none** vs dense-only: hybrid found more evidence in 3 case(s), less in 0.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ M02 (0%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?

**ollama:fixed/hybrid/cross_encoder** vs dense-only: hybrid found more evidence in 2 case(s), less in 3.
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L48 (0%→100%): What does response_model_exclude_unset=True do in FastAPI?
- ❌ M03 (100%→50%): Which HTTP client library does FastAPI's TestClient require, and what is that library's default timeout?
- ❌ M07 (50%→0%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?
- ❌ A05 (100%→0%): How should I handle errors?

**ollama:recursive/hybrid/none** vs dense-only: hybrid found more evidence in 4 case(s), less in 1.
- ✅ L35 (0%→100%): What's the difference between alias, validation_alias and serialization_alias in Pydantic?
- ✅ L44 (0%→100%): How does RFC 8725 recommend preventing one kind of JWT from being confused with another?
- ✅ M03 (50%→100%): Which HTTP client library does FastAPI's TestClient require, and what is that library's default timeout?
- ✅ M06 (50%→100%): FastAPI's BackgroundTasks comes from Starlette. If I add several background tasks and one raises an exception, do the remaining ones still run?
- ❌ M04 (50%→0%): If a FastAPI request body model has an int field and the client sends "123" as a string, will FastAPI accept it? How could I make it strict?

**ollama:recursive/hybrid/cross_encoder** vs dense-only: hybrid found more evidence in 2 case(s), less in 1.
- ✅ M02 (0%→50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ✅ M07 (50%→100%): For a public client using the authorization code grant, what does RFC 9700 require to protect the code, and how long should the code live according to RFC 6749?
- ❌ M04 (50%→0%): If a FastAPI request body model has an int field and the client sends "123" as a string, will FastAPI accept it? How could I make it strict?

**ollama:semantic/hybrid/none** vs dense-only: hybrid found more evidence in 8 case(s), less in 0.
- ✅ L04 (0%→100%): I have both /users/me and /users/{user_id} routes in FastAPI. Does the order I declare them in matter?
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L16 (0%→100%): How can I make FastAPI reject requests that send unexpected query parameters?
- ✅ L40 (0%→100%): Should OAuth clients still use the implicit grant according to RFC 9700?
- ✅ L42 (0%→100%): What does RFC 9700 require for refresh tokens issued to public clients?
- ✅ L46 (0%→100%): What maximum authorization code lifetime does RFC 6749 recommend?
- ✅ M02 (0%→100%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- ✅ A04 (0%→100%): What happens to extra fields?

**ollama:semantic/hybrid/cross_encoder** vs dense-only: hybrid found more evidence in 4 case(s), less in 3.
- ✅ L06 (0%→100%): Which numeric validation parameters can be used with Query and Path in FastAPI?
- ✅ L15 (0%→100%): How do I mark a FastAPI endpoint as deprecated without removing it?
- ✅ L20 (0%→100%): What must the JSON response of an OAuth2 token endpoint contain in FastAPI's tutorial?
- ✅ M01 (50%→100%): FastAPI's security tutorial uses the OAuth2 password flow. What does RFC 9700 say about using the resource owner password credentials grant?
- ❌ L31 (100%→0%): Is it safe to use a mutable default like a list for a Pydantic model field?
- ❌ L45 (100%→0%): What are the two OAuth 2.0 client types defined in RFC 6749?
- ❌ M03 (100%→50%): Which HTTP client library does FastAPI's TestClient require, and what is that library's default timeout?


## Evidence missed in the best config (local:recursive/dense/cross_encoder)

- L41 (lookup, recall 0%): Which PKCE code challenge method does RFC 9700 say clients should use, and why?
- M01 (multi_hop, recall 50%): FastAPI's security tutorial uses the OAuth2 password flow. What does RFC 9700 say about using the resource owner password credentials grant?
- M02 (multi_hop, recall 50%): FastAPI's JWT tutorial signs tokens with HS256 and a SECRET_KEY. What does RFC 8725 say about the key used with HS256?
- M03 (multi_hop, recall 50%): Which HTTP client library does FastAPI's TestClient require, and what is that library's default timeout?
- M04 (multi_hop, recall 0%): If a FastAPI request body model has an int field and the client sends "123" as a string, will FastAPI accept it? How could I make it strict?
- A05 (ambiguous, recall 0%): How should I handle errors?
