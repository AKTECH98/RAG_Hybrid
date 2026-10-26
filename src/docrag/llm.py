"""Thin OpenAI chat wrapper: structured (pydantic) outputs, retries, and a response cache.

The cache key is the full request (model, messages, schema, temperature), so re-running an
eval or a demo with identical inputs costs nothing and returns identical outputs.
"""

import hashlib
import json
import sqlite3
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
        self.conn.execute("CREATE TABLE IF NOT EXISTS resp (key TEXT PRIMARY KEY, body TEXT)")

    @staticmethod
    def key(**parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT body FROM resp WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, body: str) -> None:
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

    def _messages(self, system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def generate(self, model: str, system: str, user: str, temperature: float = 0.0) -> str:
        key = ResponseCache.key(kind="text", model=model, s=system, u=user, t=temperature)
        if self.cache and (hit := self.cache.get(key)) is not None:
            return hit
        resp = self.client.chat.completions.create(
            model=model, messages=self._messages(system, user), temperature=temperature
        )
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
        msg = resp.choices[0].message
        if msg.parsed is None:
            raise RuntimeError(f"Model refused or returned no parsable output: {msg.refusal}")
        if self.cache:
            self.cache.put(key, msg.parsed.model_dump_json())
        return msg.parsed


_llm: OpenAILLM | None = None


def get_llm(settings: Settings | None = None) -> OpenAILLM:
    global _llm
    if _llm is None:
        _llm = OpenAILLM(settings)
    return _llm
