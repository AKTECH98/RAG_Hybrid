"""LLM clients (OpenAI, Ollama): structured (pydantic) outputs, retries, response cache.

The cache key is the full request (model, messages, schema, temperature), so re-running an
eval or a demo with identical inputs costs nothing and returns identical outputs.
"""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from docrag.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def generate(self, model: str, system: str, user: str, temperature: float = 0.0) -> str: ...

    def structured(
        self, model: str, system: str, user: str, schema: type[T], temperature: float = 0.0
    ) -> T: ...


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("CREATE TABLE IF NOT EXISTS resp (key TEXT PRIMARY KEY, body TEXT)")

    @staticmethod
    def key(**parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT body FROM resp WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, body: str) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO resp VALUES (?, ?)", (key, body))
            self.conn.commit()


class OpenAILLM:
    def __init__(self, settings: Settings | None = None, use_cache: bool = True):
        from openai import OpenAI

        s = settings or get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (add it to .env)")
        self.client = OpenAI(api_key=s.openai_api_key, max_retries=5, timeout=120)
        self.cache = ResponseCache(s.cache_dir / "llm.sqlite") if use_cache else None
        # tokens actually billed (cache hits excluded), per model
        self.usage: dict[str, dict[str, int]] = {}
        self._usage_lock = threading.Lock()

    def _messages(self, system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _record(self, model: str, usage) -> None:
        if usage is None:
            return
        with self._usage_lock:
            u = self.usage.setdefault(model, {"input": 0, "output": 0, "calls": 0})
            u["input"] += usage.prompt_tokens
            u["output"] += usage.completion_tokens
            u["calls"] += 1

    def generate(self, model: str, system: str, user: str, temperature: float = 0.0) -> str:
        key = ResponseCache.key(kind="text", model=model, s=system, u=user, t=temperature)
        if self.cache and (hit := self.cache.get(key)) is not None:
            return hit
        resp = self.client.chat.completions.create(
            model=model, messages=self._messages(system, user), temperature=temperature
        )
        self._record(model, resp.usage)
        text = resp.choices[0].message.content or ""
        if self.cache:
            self.cache.put(key, text)
        return text

    def structured(
        self, model: str, system: str, user: str, schema: type[T], temperature: float = 0.0
    ) -> T:
        key = ResponseCache.key(
            kind="json",
            model=model,
            s=system,
            u=user,
            t=temperature,
            schema=schema.model_json_schema(),
        )
        if self.cache and (hit := self.cache.get(key)) is not None:
            return schema.model_validate_json(hit)
        resp = self.client.chat.completions.parse(
            model=model,
            messages=self._messages(system, user),
            response_format=schema,
            temperature=temperature,
        )
        self._record(model, resp.usage)
        msg = resp.choices[0].message
        if msg.parsed is None:
            raise RuntimeError(f"Model refused or returned no parsable output: {msg.refusal}")
        if self.cache:
            self.cache.put(key, msg.parsed.model_dump_json())
        return msg.parsed


RETRY_DELAYS_S = (5.0, 20.0)


def ollama_post(client, path: str, body: dict) -> dict:
    """POST to Ollama, retrying transient failures.

    On a small box (8 GB Jetson) the model runner can crash under memory pressure ("model
    runner has unexpectedly stopped", HTTP 500); Ollama reloads it on the next request, so
    a short wait and retry usually succeeds. 4xx errors (bad model name, bad schema) fail
    immediately.
    """
    import httpx

    for attempt in range(len(RETRY_DELAYS_S) + 1):
        try:
            resp = client.post(path, json=body)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            error: str = f"{type(e).__name__}: {e}"
        else:
            if resp.status_code == 200:
                return resp.json()
            error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code < 500:
                break
        if attempt < len(RETRY_DELAYS_S):
            time.sleep(RETRY_DELAYS_S[attempt])
    raise RuntimeError(f"Ollama {path} failed: {error}")


class OllamaLLM:
    """Ollama via its native /api/chat endpoint.

    Not the OpenAI-compatible /v1 layer, because only the native API lets each request set
    `num_ctx` (the server default of 2048 tokens silently drops the *start* of longer
    prompts, i.e. the system prompt and first sources) and `think`. Structured outputs use
    Ollama's `format` = JSON schema, which constrains decoding to valid JSON.
    """

    def __init__(self, settings: Settings | None = None, use_cache: bool = True):
        import httpx

        s = settings or get_settings()
        self.client = httpx.Client(base_url=s.ollama_base_url, timeout=s.ollama_timeout_s)
        self.num_ctx = s.ollama_num_ctx
        self.think = s.ollama_think
        self.cache = ResponseCache(s.cache_dir / "llm.sqlite") if use_cache else None
        self.usage: dict[str, dict[str, int]] = {}
        self._usage_lock = threading.Lock()
        self._slots = threading.Semaphore(s.ollama_max_parallel)

    def _chat(self, model: str, system: str, user: str, temperature: float, fmt=None) -> str:
        body = {
            "model": model,
            "stream": False,
            "think": self.think,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if fmt is not None:
            body["format"] = fmt
        with self._slots:
            data = ollama_post(self.client, "/api/chat", body)
        prompt_tokens = data.get("prompt_eval_count", 0)
        if prompt_tokens >= self.num_ctx:
            raise RuntimeError(
                f"Prompt filled the whole {self.num_ctx}-token context and was truncated; "
                "raise DOCRAG_OLLAMA_NUM_CTX"
            )
        with self._usage_lock:
            u = self.usage.setdefault(model, {"input": 0, "output": 0, "calls": 0})
            u["input"] += prompt_tokens
            u["output"] += data.get("eval_count", 0)
            u["calls"] += 1
        return data["message"]["content"]

    def generate(self, model: str, system: str, user: str, temperature: float = 0.0) -> str:
        key = ResponseCache.key(kind="text", model=model, s=system, u=user, t=temperature)
        if self.cache and (hit := self.cache.get(key)) is not None:
            return hit
        text = self._chat(model, system, user, temperature)
        if self.cache:
            self.cache.put(key, text)
        return text

    def structured(
        self, model: str, system: str, user: str, schema: type[T], temperature: float = 0.0
    ) -> T:
        json_schema = schema.model_json_schema()
        key = ResponseCache.key(
            kind="json", model=model, s=system, u=user, t=temperature, schema=json_schema
        )
        if self.cache and (hit := self.cache.get(key)) is not None:
            return schema.model_validate_json(hit)
        # The schema also goes in the prompt: with only the grammar constraint, qwen3-4b took
        # the cheapest valid path and returned {"scores": []} for every reranking call.
        system = f"{system}\n\nRespond only with JSON matching this JSON schema:\n" + json.dumps(
            json_schema
        )
        # Small models occasionally emit schema-valid JSON that breaks a field constraint
        # (e.g. a score of 11, an empty list); retry once with the validation error appended.
        prompt, last_error = user, None
        for _ in range(2):
            raw = self._chat(model, system, prompt, temperature, fmt=json_schema)
            try:
                parsed = schema.model_validate_json(raw)
                break
            except ValueError as e:
                last_error = e
                prompt = f"{user}\n\nYour previous output was invalid ({e}). Return valid JSON."
        else:
            raise RuntimeError(f"Model returned invalid JSON twice: {last_error}")
        if self.cache:
            self.cache.put(key, parsed.model_dump_json())
        return parsed


_llms: dict[str, LLM] = {}
_llms_lock = threading.Lock()


def get_llm(settings: Settings | None = None) -> LLM:
    s = settings or get_settings()
    with _llms_lock:
        if s.llm_provider not in _llms:
            _llms[s.llm_provider] = OllamaLLM(s) if s.llm_provider == "ollama" else OpenAILLM(s)
        return _llms[s.llm_provider]
