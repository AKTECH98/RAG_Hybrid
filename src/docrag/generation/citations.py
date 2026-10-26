"""Citation parsing and verification.

1. parse_claims: split the answer into claim sentences/bullets and pull out their [n] markers.
2. verify_claims: one batched LLM-as-judge call checks every (claim, cited source) pair
   and returns supported / partial / unsupported with a reason.

A citation is only as good as the chunk behind it: models routinely attach a real-looking
[2] to a sentence that block 2 never says. This layer catches exactly that.
"""

import re

from pydantic import BaseModel, Field

from docrag.generation.models import CitationCheck, Claim, Source, Verdict
from docrag.llm import LLM

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
FENCE_RE = re.compile(r"```.*?```", re.S)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*\"(\[])")
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
LEADING_CITES_RE = re.compile(r"^((?:\s*\[\d+(?:\s*,\s*\d+)*\])+)\s*")
# "Here is how you can do it for a single request:" introduces a code block, it isn't a fact
TRANSITION_RE = re.compile(
    r"^(here(?:'s| is| are)|for example|for instance|below|the following|this is how|"
    r"as shown|example)\b",
    re.I,
)
# "The provided documentation does not cover X" is a statement about the context, not a
# factual claim that needs a citation.
GAP_RE = re.compile(
    # "it"/"this" usually refers back to the context ("Therefore, it does not cover ...")
    r"(\b(context|documentation|docs|sources?|provided (?:text|information))\b.{0,60}"
    r"|^(?:\w+,\s*)?(?:it|this|they)\s)"
    r"\b(does not|doesn't|do not|don't|not)\b.{0,20}"
    r"\b(cover|mention|include|specify|say|describe|address|contain|provide|explain)",
    re.I,
)


def extract_citations(text: str) -> list[int]:
    nums: list[int] = []
    for group in CITATION_RE.findall(text):
        for n in group.split(","):
            if int(n) not in nums:
                nums.append(int(n))
    return nums


def _strip_markdown(text: str) -> str:
    text = CITATION_RE.sub("", text)
    # drop emphasis/heading/quote markers; keep "_" since identifiers like status_code use it
    text = re.sub(r"\*+|^[#>\s]+", "", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def parse_claims(answer: str, n_sources: int) -> list[Claim]:
    """Split an answer into claims with their citations (code blocks are skipped)."""
    segments: list[str] = []
    for line in FENCE_RE.sub("\n", answer).splitlines():
        line = BULLET_RE.sub("", line).strip()
        if not line or line.startswith("#"):
            continue
        for seg in SENTENCE_SPLIT_RE.split(line):
            # "... sentence. [2]" -> the marker belongs to the previous sentence
            if (m := LEADING_CITES_RE.match(seg)) and segments:
                segments[-1] += " " + m.group(1).strip()
                seg = seg[m.end() :]
            if seg.strip():
                segments.append(seg.strip())

    claims: list[Claim] = []
    for seg in segments:
        cites = extract_citations(seg)
        text = _strip_markdown(seg)
        if not cites:
            # transitions ("Here is an example:"), fragments and gap statements aren't
            # claims; a long sentence ending in ":" usually still states a fact
            words = len(text.split())
            transition = seg.rstrip().endswith(":") and (
                words < 7 or TRANSITION_RE.match(text) is not None
            )
            if transition or words < 4 or GAP_RE.search(text):
                continue
        claims.append(
            Claim(
                id=len(claims) + 1,
                text=text,
                citations=[n for n in cites if 1 <= n <= n_sources],
                invalid_citations=[n for n in cites if not 1 <= n <= n_sources],
            )
        )
    return claims


class _Verdict(BaseModel):
    claim_id: int
    source: int
    verdict: Verdict
    reason: str


class _Verdicts(BaseModel):
    verdicts: list[_Verdict] = Field(min_length=1)


VERIFY_SYSTEM = """You verify citations in an answer generated from documentation.
For every (claim, source) pair listed, decide whether that source supports that claim:
- "supported": the source states the claim, or it follows directly from the source.
- "partial": the source supports part of the claim, but the claim adds something the \
source does not say (an extra detail, value, condition or generalization).
- "unsupported": the source does not back the claim, or contradicts it.
Judge each source on its own text only, not on your own knowledge. Be strict: a claim \
that adds a default value, version, or recommendation absent from the source is not \
"supported". Give a one-sentence reason. Return exactly one verdict per listed pair."""


def verify_claims(
    llm: LLM, model: str, question: str, claims: list[Claim], sources: list[Source]
) -> list[CitationCheck]:
    pairs = [(c.id, n) for c in claims for n in c.citations]
    if not pairs:
        return []
    by_n = {s.n: s for s in sources}
    cited = sorted({n for _, n in pairs})
    source_text = "\n\n".join(f"[{n}] {by_n[n].label}\n{by_n[n].text}" for n in cited)
    claim_text = "\n".join(
        f"Claim {c.id}: {c.text}\n  -> check against sources {c.citations}"
        for c in claims
        if c.citations
    )
    result = llm.structured(
        model,
        VERIFY_SYSTEM,
        f"Question that was answered: {question}\n\nSources:\n\n{source_text}\n\n"
        f"Claims to verify:\n{claim_text}",
        _Verdicts,
    )
    got = {(v.claim_id, v.source): v for v in result.verdicts}
    checks = []
    for claim_id, n in pairs:
        v = got.get((claim_id, n))
        checks.append(
            CitationCheck(
                claim_id=claim_id,
                source=n,
                verdict=v.verdict if v else "unsupported",
                reason=v.reason if v else "judge returned no verdict for this pair",
            )
        )
    return checks


def attach_checks(claims: list[Claim], checks: list[CitationCheck]) -> None:
    for claim in claims:
        claim.checks = [c for c in checks if c.claim_id == claim.id]
