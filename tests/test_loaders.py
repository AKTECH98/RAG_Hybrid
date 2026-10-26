import pytest

from docrag.ingest.loaders import _numbered_sections, _strip_running_lines, load_document
from docrag.ingest.store import DocumentStore, validate_doc_id

MARKDOWN = """---
title: ignored front matter
---
# Request Body { #request-body }

Use [Pydantic](https://pydantic.dev) models, see [`Field`][pydantic.fields.Field].

/// note | Heads up

Sending a body with `GET` is discouraged.

///

## Declare it { #declare-it }

```python {lint="skip"}
# this comment is not a heading
class Item(BaseModel):
    name: str
```

!!! warning "Careful"
    Indented admonition body.

??? api "API Documentation"
    [`pydantic.fields.Field`][pydantic.fields.Field]<br>
"""


def test_markdown_structure_and_cleanup(tmp_path):
    path = tmp_path / "body.md"
    path.write_text(MARKDOWN)
    doc = load_document(path, "fastapi/body.md")

    assert doc.title == "Request Body"
    assert [s.heading_path for s in doc.sections] == [
        ["Request Body"],
        ["Request Body", "Declare it"],
    ]
    intro, declare = doc.sections[0].text, doc.sections[1].text
    assert "front matter" not in intro
    assert "Use Pydantic models, see `Field`." in intro  # links flattened
    assert "Note: Heads up" in intro and "///" not in intro
    assert "# this comment is not a heading" in declare  # fenced code kept verbatim
    assert "```python\n" in declare and "lint=" not in declare
    assert "Warning: Careful:" in declare and "Indented admonition body." in declare
    assert "API Documentation" not in declare and "<br>" not in declare


def test_markdown_without_h1_uses_filename_as_title(tmp_path):
    path = tmp_path / "strict_mode.md"
    path.write_text("Intro text.\n\n## Section\n\nBody.")
    doc = load_document(path, "pydantic/strict_mode.md")
    assert doc.title == "Strict Mode"
    assert all(s.heading_path[0] == "Strict Mode" for s in doc.sections)


def test_html_extracts_article_with_code(tmp_path):
    path = tmp_path / "requests.html"
    path.write_text(
        """<html><head><title>Requests - Starlette</title></head><body>
        <nav>Navigation junk</nav>
        <article><h1>Requests<a class="headerlink" href="#">¶</a></h1>
        <p>Starlette includes a <code>Request</code> class.</p>
        <h3>Headers</h3>
        <pre><code><span>request</span>.<span>headers</span>[<span>'x'</span>]</code></pre>
        <ul><li>first item</li></ul>
        </article></body></html>"""
    )
    doc = load_document(path, "starlette/requests.html")
    assert doc.title == "Requests"
    assert doc.sections[0].text == "Starlette includes a Request class."
    headers = doc.sections[1]
    assert headers.heading_path == ["Requests", "Headers"]
    assert "request.headers['x']" in headers.text  # highlighted spans rejoined
    assert "- first item" in headers.text
    assert "Navigation junk" not in doc.text and "¶" not in doc.text


def test_running_headers_and_footers_are_removed():
    topics = [
        "tokens",
        "clients",
        "scopes",
        "grants",
        "errors",
        "refresh",
        "pkce",
        "jwt",
        "redirects",
        "cookies",
        "csrf",
        "tls",
    ]
    bodies = [
        f"Opening line about {t}.\nMiddle text on {t}.\nClosing remark on {t}." for t in topics
    ]
    pages = [
        # the footer's padding shrinks by one space once page numbers reach two digits
        f"RFC 1 Title October 2012\n{body}\nAuthor   Standards Track   {' ' * (i < 10)}[Page {i}]"
        for i, body in enumerate(bodies, start=1)
    ]
    cleaned = _strip_running_lines(pages)
    assert [p.strip() for p in cleaned] == bodies


def test_numbered_sections_skip_toc_and_list_items():
    text = "\n".join(
        [
            "Abstract text.",
            "Table of Contents",
            "1. Introduction 3",
            "2. Attacks 4",
            "2.1. Mix-Up 4",
            "1. Introduction",
            "Intro body.",
            "2. Attacks",
            "Steps:",
            "3. The attacker therefore obtains the code",  # numbered list item, not in TOC
            "2.1. Mix-Up",
            "Mix-up body.",
        ]
    )
    sections = _numbered_sections([(text, 1)], "RFC X", require_col0=False)
    paths = [s.heading_path for s in sections]
    assert paths == [
        ["RFC X"],
        ["RFC X", "1. Introduction"],
        ["RFC X", "2. Attacks"],
        ["RFC X", "2. Attacks", "2.1. Mix-Up"],
    ]
    assert sections[0].text == "Abstract text."  # TOC lines dropped
    assert "3. The attacker" in sections[2].text


def test_rfc_text_requires_headings_at_column_zero(tmp_path):
    path = tmp_path / "rfc.txt"
    path.write_text(
        "Request for Comments: 9999\n\n          A Test Protocol\n\n"
        "1.  Introduction\n\n   Intro body.\n\n   2.  Indented, so not a heading\n\n"
        "2.  Terminology\n\n   Terms.\n"
    )
    doc = load_document(path, "rfc/rfc.txt")
    assert doc.title == "RFC 9999: A Test Protocol"
    headings = [s.heading_path[-1] for s in doc.sections]
    assert headings[-2:] == ["1. Introduction", "2. Terminology"]
    assert "Indented, so not a heading" in doc.sections[-2].text


@pytest.mark.parametrize("bad", ["../etc/passwd.md", "/abs/path.md", "a/../../b.md", "x.exe", ""])
def test_doc_id_validation_rejects_unsafe_ids(bad):
    with pytest.raises(ValueError):
        validate_doc_id(bad)


def test_document_store_roundtrip(settings, tmp_path):
    src = tmp_path / "note.md"
    src.write_text("# Note\n\nHello.")
    store = DocumentStore(settings)
    doc = store.ingest_file(src, "team/note.md")
    assert (settings.raw_dir / "team/note.md").read_text() == "# Note\n\nHello."
    assert store.get("team/note.md") == doc
    assert [d.doc_id for d in store.list()] == ["team/note.md"]
    assert store.delete("team/note.md") and store.get("team/note.md") is None


def test_markdown_strips_html_comments_and_anchor_markers(tmp_path):
    path = tmp_path / "strict.md"
    path.write_text(
        "# Strict\n\nIntro.\n\n<!-- old anchor -->\n<!-- markdownlint-disable -->\n"
        "[](){#strict-mode-with-field}\n\n## Field\n\nBody.\n"
    )
    doc = load_document(path, "pydantic/strict.md")
    assert "<!--" not in doc.text and "[](){#" not in doc.text
    assert [s.text for s in doc.sections] == ["Intro.", "Body."]
