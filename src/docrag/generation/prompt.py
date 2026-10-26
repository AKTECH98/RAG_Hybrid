"""Grounded generation prompt: numbered context blocks + strict citation rules."""

from docrag.generation.models import Source

INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You answer questions about technical documentation using ONLY the \
numbered context blocks provided. You have no other knowledge.

Rules:
1. EVERY sentence or bullet that states a fact must carry its own citations of the blocks \
that support it, like [1] or [2][4], placed before the sentence's final punctuation. This \
includes the first sentence and sentences that introduce a code block. Do not collect all \
citations at the end of the answer.
2. Only cite a block if it directly states what the sentence says. Never cite a block for \
something it does not contain.
3. Do not add facts, defaults, version numbers or advice that are not in the context, even \
if you believe they are true.
4. If the context answers only part of the question, answer that part and then say \
explicitly which part the documentation provided does not cover.
5. If the context does not answer the question at all, reply with exactly \
"{INSUFFICIENT_MARKER}:" followed by one sentence describing what the context does cover.
6. Be concise and direct. Use short code snippets copied from the context when they help. \
Code blocks themselves do not need citations.

Example of the expected style:
Raise `HTTPException` to return an error response [2]. It takes a `status_code` and a \
`detail` [2][3]:
```python
raise HTTPException(status_code=404, detail="Item not found")
```
You can attach extra headers with the `headers` argument [4]."""


def format_context(sources: list[Source]) -> str:
    return "\n\n".join(f"[{s.n}] Source: {s.label}\n{s.text}" for s in sources)


def build_user_prompt(question: str, sources: list[Source]) -> str:
    return f"Context:\n\n{format_context(sources)}\n\n---\n\nQuestion: {question}"
