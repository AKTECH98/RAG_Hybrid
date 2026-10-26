"""Download the sample documentation corpus into ./corpus.

Sources (all permissively licensed or freely redistributable):
  - FastAPI tutorial docs (MIT)          -> corpus/fastapi/*.md   (code samples inlined)
  - Pydantic concept docs (MIT)          -> corpus/pydantic/*.md
  - httpx docs (BSD-3)                   -> corpus/httpx/*.md
  - Starlette docs pages, rendered (BSD) -> corpus/starlette/*.html
  - IETF RFCs 9700 + 8725 (PDF), 6749 (TXT) -> corpus/rfc/

Usage: uv run python scripts/fetch_corpus.py [--force]
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"

GIT_SOURCES = [
    {
        "name": "fastapi",
        "repo": "https://github.com/fastapi/fastapi",
        "license": "MIT",
        "docs_dir": "docs/en/docs/tutorial",
        "extra_dirs": ["docs_src"],  # needed to inline {* ... *} code includes
    },
    {
        "name": "pydantic",
        "repo": "https://github.com/pydantic/pydantic",
        "license": "MIT",
        "docs_dir": "docs/concepts",
        "extra_dirs": [],
    },
    {
        "name": "httpx",
        "repo": "https://github.com/encode/httpx",
        "license": "BSD-3-Clause",
        "docs_dir": "docs",
        "extra_dirs": [],
        "exclude": ["code_of_conduct.md", "contributing.md", "third_party_packages.md"],
    },
]

STARLETTE_PAGES = [
    "requests",
    "responses",
    "middleware",
    "background",
    "exceptions",
    "testclient",
    "lifespan",
    "routing",
    "websockets",
    "staticfiles",
]

RFC_FILES = [
    ("rfc9700.pdf", "https://www.rfc-editor.org/rfc/rfc9700.pdf"),  # OAuth 2.0 Security BCP
    ("rfc8725.pdf", "https://www.rfc-editor.org/rfc/rfc8725.pdf"),  # JWT Best Current Practices
    ("rfc6749.txt", "https://www.rfc-editor.org/rfc/rfc6749.txt"),  # OAuth 2.0 Framework
]

# FastAPI docs embed code like: {* ../../docs_src/body/tutorial001_py310.py hl[5:9] *}
INCLUDE_RE = re.compile(r"\{\*\s+(?P<path>\S+\.py)(?P<opts>[^*]*)\*\}")


def sparse_clone(repo: str, dirs: list[str], dest: Path) -> None:
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", repo, str(dest)],
        check=True,
    )
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *dirs], check=True)


def inline_code_includes(text: str, repo_dir: Path) -> str:
    """Replace FastAPI's {* path *} include macros with fenced python code.

    Paths look like ../../docs_src/x.py and are relative to the mkdocs config dir, so
    anchor everything after "docs_src/" at the repo root instead of resolving them.
    """

    def repl(m: re.Match) -> str:
        path = m.group("path")
        if "docs_src/" not in path:
            return ""
        src = repo_dir / "docs_src" / path.split("docs_src/", 1)[1]
        if not src.exists():
            return ""
        return f"```python\n{src.read_text().rstrip()}\n```"

    return INCLUDE_RE.sub(repl, text)


def fetch_git_source(src: dict) -> int:
    out_dir = CORPUS / src["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / src["name"]
        sparse_clone(src["repo"], [src["docs_dir"], *src["extra_dirs"]], repo_dir)
        docs_root = repo_dir / src["docs_dir"]
        count = 0
        for md in sorted(docs_root.rglob("*.md")):
            rel = md.relative_to(docs_root)
            if str(rel) in src.get("exclude", []):
                continue
            text = inline_code_includes(md.read_text(), repo_dir)
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
            count += 1
    return count


def fetch_starlette(client: httpx.Client) -> int:
    out_dir = CORPUS / "starlette"
    out_dir.mkdir(parents=True, exist_ok=True)
    for page in STARLETTE_PAGES:
        resp = client.get(f"https://starlette.dev/{page}/")
        resp.raise_for_status()
        (out_dir / f"{page}.html").write_text(resp.text)
    return len(STARLETTE_PAGES)


def fetch_rfcs(client: httpx.Client) -> int:
    out_dir = CORPUS / "rfc"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in RFC_FILES:
        resp = client.get(url)
        resp.raise_for_status()
        (out_dir / name).write_bytes(resp.content)
    return len(RFC_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="delete and re-download the corpus")
    args = parser.parse_args()

    if CORPUS.exists() and any(CORPUS.iterdir()):
        if not args.force:
            print(f"{CORPUS} already populated; pass --force to re-download.")
            return
        shutil.rmtree(CORPUS)
    CORPUS.mkdir(exist_ok=True)

    manifest = {}
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for src in GIT_SOURCES:
            n = fetch_git_source(src)
            manifest[src["name"]] = {"source": src["repo"], "license": src["license"], "files": n}
            print(f"{src['name']:<10} {n:>4} markdown files")
        n = fetch_starlette(client)
        manifest["starlette"] = {
            "source": "https://starlette.dev",
            "license": "BSD-3-Clause",
            "files": n,
        }
        print(f"{'starlette':<10} {n:>4} html pages")
        n = fetch_rfcs(client)
        manifest["rfc"] = {
            "source": "https://www.rfc-editor.org",
            "license": "IETF Trust",
            "files": n,
        }
        print(f"{'rfc':<10} {n:>4} pdf/txt files")

    (CORPUS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done -> {CORPUS}")


if __name__ == "__main__":
    main()
