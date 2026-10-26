"""Multi-format document loaders: markdown, text, HTML, PDF -> normalized Document.

Every loader produces sections that carry a heading path (and a page number for PDFs) so
chunks can always be traced back to where they came from.
"""

import logging
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup, NavigableString, Tag

from docrag.models import Document, Section

# pdfminer warns about malformed-but-harmless font maps (e.g. "ToUnicode" in RFC 9700)
logging.getLogger("pdfminer").setLevel(logging.ERROR)


class SectionBuilder:
    """Accumulates text under a heading stack and emits non-empty sections."""

    def __init__(self, root: list[str] | None = None):
        self.root = root or []
        self.stack: list[tuple[int, str]] = []
        self.sections: list[Section] = []
        self._buf: list[str] = []
        self._page: int | None = None

    def heading(self, level: int, title: str) -> None:
        self.flush()
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))

    def text(self, text: str, page: int | None = None) -> None:
        if page is not None and self._page is not None and page != self._page:
            self.flush()  # keep page numbers accurate: one section piece per page
        if page is not None:
            self._page = page
        self._buf.append(text)

    def flush(self) -> None:
        text = _tidy("\n".join(self._buf))
        if text:
            path = self.root + [t for _, t in self.stack]
            self.sections.append(Section(heading_path=path, text=text, page=self._page))
        self._buf = []

    def build(self) -> list[Section]:
        self.flush()
        return self.sections


def _tidy(text: str) -> str:
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


# --------------------------------------------------------------------------- markdown

FENCE_RE = re.compile(r"^\s*(```|~~~)")
ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ANCHOR_RE = re.compile(r"\s*\{\s*:?\s*#[\w-]+\s*\}")
ADMONITION_RE = re.compile(r'^(\s*)(!!!|\?\?\?\+?|===)\s*(\w[\w-]*)?\s*(?:"([^"]*)")?\s*$')
SLASH_BLOCK_RE = re.compile(r"^/{3,}\s*(\w+)?\s*(?:\|\s*(.*))?$")
INLINE_TAG_RE = re.compile(
    r"</?(abbr|br|a|span|div|img|details|summary|sup|sub|b|i|em|strong|p|kbd)\b[^>]*>", re.I
)


def _clean_md_inline(line: str) -> str:
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)  # images
    line = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", line)  # [text][ref]
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # [text](url)
    return INLINE_TAG_RE.sub("", line)


def _expand_admonitions(lines: list[str]) -> list[str]:
    """Flatten mkdocs-material admonitions / content tabs into plain labelled text.

    `!!! note "Title"` + indented body  ->  "Note: Title" + dedented body
    `??? api ...` blocks are pure API cross-reference links and are dropped.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = ADMONITION_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, marker, kind, title = len(m.group(1)), m.group(2), m.group(3), m.group(4)
        body: list[str] = []
        i += 1
        while i < len(lines) and (
            not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) >= indent + 4
        ):
            body.append(lines[i][indent + 4 :] if lines[i].strip() else "")
            i += 1
        if kind == "api":
            continue
        label = title if marker == "===" else f"{(kind or 'note').capitalize()}: {title or ''}"
        out.append(label.strip().rstrip(":") + ":" if marker != "===" else f"({label})")
        out.extend(_expand_admonitions(body))
    return out


def load_markdown(path: Path) -> tuple[str, list[Section]]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # front matter
    text = re.sub(r"<!--.*?-->\n?", "", text, flags=re.S)  # <!-- fmt:off --> etc.
    text = re.sub(r"^\[\]\(\)\{#[\w-]+\}\s*$\n?", "", text, flags=re.M)  # mkdocs anchors
    lines = _expand_admonitions(text.splitlines())

    builder = SectionBuilder()
    title = None
    in_fence = False
    for raw in lines:
        if FENCE_RE.match(raw):
            in_fence = not in_fence
            # drop fence attributes: ```python {lint="skip"} -> ```python
            raw = re.sub(r"^(\s*(?:```|~~~)\s*[\w+-]*).*$", r"\1", raw)
            builder.text(raw)
            continue
        if in_fence:
            builder.text(raw)
            continue
        if raw.startswith("--8<--"):
            continue
        slash = SLASH_BLOCK_RE.match(raw.strip())
        if slash:  # FastAPI "/// note | Title" admonition markers
            kind, extra = slash.group(1), slash.group(2)
            if kind == "tab" and extra:
                builder.text(f"({extra.strip()})")
            elif kind and kind != "details":
                builder.text(
                    f"{kind.capitalize()}: {extra.strip()}" if extra else f"{kind.capitalize()}:"
                )
            continue
        m = ATX_RE.match(raw)
        if m:
            heading = _clean_md_inline(ANCHOR_RE.sub("", m.group(2))).strip()
            level = len(m.group(1))
            if level == 1 and title is None:
                title = heading
            builder.heading(level, heading)
            continue
        builder.text(_clean_md_inline(raw))
    sections = builder.build()
    if title is None:  # e.g. Pydantic docs take their H1 from mkdocs nav, not the file
        title = path.stem.replace("-", " ").replace("_", " ").title()
        for s in sections:
            s.heading_path.insert(0, title)
    return title, sections


# --------------------------------------------------------------------------- html

HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
TEXT_BLOCKS = {"p", "dt", "dd", "figcaption", "summary", "caption"}
DROP_TAGS = ["script", "style", "nav", "header", "footer", "noscript", "svg", "form", "button"]


def _inline_text(tag: Tag) -> str:
    for br in tag.find_all("br"):
        br.replace_with("\n")
    return re.sub(r"[ \t\r\f\v]+", " ", tag.get_text("")).strip()


def _walk_html(node: Tag, builder: SectionBuilder) -> None:
    for child in node.children:
        if isinstance(child, NavigableString):
            if child.strip():
                builder.text(child.strip())
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in HEADING_TAGS:
            builder.heading(HEADING_TAGS[name], _inline_text(child))
        elif name == "pre":
            builder.text(f"```\n{child.get_text('').strip()}\n```")
        elif name == "table":
            for tr in child.find_all("tr"):
                cells = [_inline_text(c) for c in tr.find_all(["th", "td"])]
                builder.text(" | ".join(cells))
        elif name == "li":
            if child.find(["pre", "ul", "ol", "table", "p"]):
                _walk_html(child, builder)
            else:
                builder.text(f"- {_inline_text(child)}")
        elif name in TEXT_BLOCKS:
            builder.text(_inline_text(child))
        else:
            _walk_html(child, builder)


def load_html(path: Path) -> tuple[str, list[Section]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    for tag in soup.select("a.headerlink, .md-sidebar, .md-source, .md-search"):
        tag.decompose()
    container = soup.select_one("article") or soup.select_one("main") or soup.body or soup
    h1 = container.find("h1")
    title = (
        _inline_text(h1) if h1 else (soup.title.get_text(strip=True) if soup.title else path.stem)
    )
    builder = SectionBuilder()
    _walk_html(container, builder)
    return title, builder.build()


# --------------------------------------------------------------------------- pdf / txt

NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+([A-Z][^\n]{1,100}?)(?:\s+\d+)?$")
APPENDIX_RE = re.compile(r"^Appendix\s+([A-Z])\.\s+(\S.{1,100})$")


def _strip_running_lines(pages: list[str]) -> list[str]:
    """Remove running headers/footers: edge lines that repeat (modulo digits) on many pages."""
    if len(pages) < 3:
        return pages

    def norm(line: str) -> str:
        # digits vary (page numbers) and so does column padding around them
        return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip()))

    edge_counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln for ln in page.splitlines() if ln.strip()]
        for ln in set(lines[:2] + lines[-2:]):
            edge_counts[norm(ln)] += 1
    running = {k for k, v in edge_counts.items() if v >= len(pages) * 0.5}
    return ["\n".join(ln for ln in p.splitlines() if norm(ln) not in running) for p in pages]


def _is_successor(prev: tuple[int, ...], cand: tuple[int, ...]) -> bool:
    """True if `cand` can directly follow `prev` in a numbered outline (child/sibling/uncle)."""
    if not prev:
        return cand == (1,)
    if cand == prev + (1,):
        return True
    return any(cand == prev[:i] + (prev[i] + 1,) for i in range(len(prev)))


TOC_LEADER_RE = re.compile(r"\.{4,}\s*\d+\s*$")


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()[:20]


def _numbered_sections(
    pages: list[tuple[str, int | None]], title: str, require_col0: bool
) -> list[Section]:
    """Split RFC-style text on numbered headings ("4.5.3. Countermeasures").

    Pass 1 finds heading candidates. The table of contents repeats the outline, so the body
    starts at the *last* "1." heading whose title matches the first one ("Introduction").
    Pass 2 accepts a body heading only if it continues the outline in order and, when a TOC
    was found, matches the TOC title for that number (filters numbered list items).
    """
    lines = [(line, page) for text, page in pages for line in text.splitlines()]
    cands: dict[int, tuple[tuple[int, ...], str]] = {}
    for i, (line, _) in enumerate(lines):
        if require_col0 and (not line[:1].strip()):
            continue
        stripped = line.strip()
        if (m := NUMBERED_HEADING_RE.match(stripped)) and not stripped.endswith("."):
            nums = tuple(int(n) for n in m.group(1).split("."))
            cands[i] = (nums, m.group(2).strip())

    ones = [i for i, (nums, _) in cands.items() if nums == (1,)]
    body_start = 0
    if ones:
        first_title = _norm_title(cands[ones[0]][1])
        body_start = max(i for i in ones if _norm_title(cands[i][1]) == first_title)
    toc = {nums: _norm_title(t) for i, (nums, t) in cands.items() if i < body_start}

    builder = SectionBuilder(root=[title])
    prev: tuple[int, ...] = ()
    for i, (line, page) in enumerate(lines):
        stripped = line.strip()
        if i < body_start:
            if i in cands or stripped == "Table of Contents" or TOC_LEADER_RE.search(stripped):
                continue
            builder.text(line, page=page)
            continue
        if i in cands:
            nums, heading = cands[i]
            if _is_successor(prev, nums) and (not toc or toc.get(nums) == _norm_title(heading)):
                prev = nums
                builder.heading(len(nums), f"{'.'.join(map(str, nums))}. {heading}")
                continue
        col0_ok = not require_col0 or bool(line[:1].strip())
        if col0_ok and (m := APPENDIX_RE.match(stripped)):
            builder.heading(1, f"Appendix {m.group(1)}. {m.group(2).strip()}")
            prev = (10_000,)  # appendix sub-numbering (A.1) is not tracked
            continue
        builder.text(line, page=page)
    return builder.build()


def load_pdf(path: Path) -> tuple[str, list[Section]]:
    with pdfplumber.open(path) as pdf:
        raw_pages = [p.extract_text() or "" for p in pdf.pages]
        meta_title = (pdf.metadata or {}).get("Title")
    pages = _strip_running_lines(raw_pages)
    first = next((ln.strip() for ln in pages[0].splitlines() if ln.strip()), "") if pages else ""
    title = meta_title or first or path.stem
    if pages and first and not meta_title:
        # RFC PDFs: the first line is the number, the second the real title
        lines = [ln.strip() for ln in pages[0].splitlines() if ln.strip()]
        title = (
            f"{lines[0]}: {lines[1]}" if len(lines) > 1 and lines[0].startswith("RFC") else first
        )
    return title, _numbered_sections([(t, i + 1) for i, t in enumerate(pages)], title, False)


def load_text(path: Path) -> tuple[str, list[Section]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pages = _strip_running_lines(text.split("\f")) if "\f" in text else [text]
    title = _txt_title(pages[0]) or path.stem
    sections = _numbered_sections([(p, None) for p in pages], title, require_col0=True)
    if len(sections) <= 1:  # plain text without an outline
        sections = [Section(heading_path=[title], text=_tidy(text))]
    return title, sections


def _txt_title(first_page: str) -> str | None:
    """RFC .txt files put the centered title after the header block."""
    m = re.search(r"Request for Comments:\s*(\d+)", first_page)
    for line in first_page.splitlines():
        if line.startswith(" " * 10) and line.strip() and ":" not in line:
            title = line.strip()
            return f"RFC {m.group(1)}: {title}" if m else title
    return None


# --------------------------------------------------------------------------- dispatch

LOADERS: dict[str, tuple[str, Callable[[Path], tuple[str, list[Section]]]]] = {
    ".md": ("md", load_markdown),
    ".markdown": ("md", load_markdown),
    ".txt": ("txt", load_text),
    ".html": ("html", load_html),
    ".htm": ("html", load_html),
    ".pdf": ("pdf", load_pdf),
}


def supported(path: Path) -> bool:
    return path.suffix.lower() in LOADERS


def load_document(path: Path, doc_id: str, source_path: str | None = None) -> Document:
    try:
        fmt, loader = LOADERS[path.suffix.lower()]
    except KeyError as e:
        raise ValueError(f"Unsupported file type: {path.suffix}") from e
    title, sections = loader(path)
    return Document(
        doc_id=doc_id,
        source_path=source_path or str(path),
        format=fmt,
        title=title,
        sections=sections,
    )
