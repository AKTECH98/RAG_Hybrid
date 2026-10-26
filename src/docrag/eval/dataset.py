"""Golden Q&A dataset: loading, evidence matching, and validation against the corpus.

Relevance is defined by evidence *quotes* rather than chunk ids: a retrieved chunk is
relevant if it contains one of the case's quotes. That makes the same labels work for
every chunking strategy (chunk ids differ per strategy, the text doesn't).
"""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from docrag.config import PROJECT_ROOT
from docrag.models import Chunk

CaseType = Literal["lookup", "multi_hop", "no_answer", "ambiguous"]
DEFAULT_PATH = PROJECT_ROOT / "eval" / "golden.jsonl"


class Evidence(BaseModel):
    doc_id: str
    quote: str


class GoldenCase(BaseModel):
    id: str
    type: CaseType
    question: str
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return self.type != "no_answer"


def normalize(text: str) -> str:
    """Lowercase, drop all whitespace: robust to line wraps and PDF spacing quirks."""
    return re.sub(r"\s+", "", text).lower()


def chunk_matches(chunk: Chunk, ev: Evidence) -> bool:
    return chunk.doc_id == ev.doc_id and normalize(ev.quote) in normalize(chunk.text)


def load_golden(path: Path = DEFAULT_PATH) -> list[GoldenCase]:
    cases = [
        GoldenCase.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in golden set")
    return cases


def validate_against_corpus(cases: list[GoldenCase], doc_texts: dict[str, str]) -> list[str]:
    """Return problems: unknown documents or quotes that don't appear in their document."""
    problems = []
    norm_docs = {d: normalize(t) for d, t in doc_texts.items()}
    for case in cases:
        if case.answerable and not case.evidence:
            problems.append(f"{case.id}: answerable case without evidence")
        for ev in case.evidence:
            if ev.doc_id not in norm_docs:
                problems.append(f"{case.id}: unknown doc {ev.doc_id}")
            elif normalize(ev.quote) not in norm_docs[ev.doc_id]:
                problems.append(f"{case.id}: quote not found in {ev.doc_id}: {ev.quote[:60]!r}")
    return problems
