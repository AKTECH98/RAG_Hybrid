"""The "I don't know" path: say what was found, what wasn't, and where to look.

Used when retrieval confidence is below the threshold (generation is skipped entirely) or
when the model itself reports the context is insufficient. A structured non-answer is
more useful than a fluent fabrication.
"""

from pydantic import BaseModel

from docrag.generation.models import Source
from docrag.generation.prompt import format_context
from docrag.llm import LLM


class _Gaps(BaseModel):
    found: list[str]
    missing: list[str]


GAPS_SYSTEM = """A documentation search could not confidently answer a question.
Given the question and the closest passages that were retrieved, list:
- found: short statements of what the passages DO say that relates to the question \
(empty if nothing is related). Cite passages as [n].
- missing: the specific pieces of information the question needs that the passages do \
not contain.
Keep each item under 25 words. Do not answer the question from your own knowledge."""


def explain_gaps(llm: LLM, model: str, question: str, sources: list[Source]) -> _Gaps:
    return llm.structured(
        model,
        GAPS_SYSTEM,
        f"Question: {question}\n\nClosest passages:\n\n{format_context(sources)}",
        _Gaps,
    )


def suggest_documents(sources: list[Source], limit: int = 3) -> list[str]:
    """Distinct documents behind the closest passages, most relevant first."""
    seen: list[str] = []
    for s in sorted(sources, key=lambda s: -s.relevance):
        if s.doc_id not in seen:
            seen.append(s.doc_id)
    return seen[:limit]
