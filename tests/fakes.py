"""Offline stand-ins for the OpenAI LLM."""

from collections.abc import Callable

from pydantic import BaseModel


class FakeLLM:
    """Routes calls to handlers keyed by schema name (structured) or returns fixed text."""

    def __init__(
        self,
        structured_handlers: dict[str, Callable[[str, str], BaseModel]] | None = None,
        text: str | Callable[[str, str], str] = "",
    ):
        self.structured_handlers = structured_handlers or {}
        self.text = text
        self.calls: list[tuple[str, str, str]] = []

    def generate(self, model: str, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append(("text", system, user))
        return self.text(system, user) if callable(self.text) else self.text

    def structured(self, model, system, user, schema, temperature: float = 0.0):
        self.calls.append((schema.__name__, system, user))
        return schema.model_validate(self.structured_handlers[schema.__name__](system, user))
