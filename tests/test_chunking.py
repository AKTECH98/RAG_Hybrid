import tiktoken

from docrag.chunking import FixedChunker, RecursiveChunker, SemanticChunker
from docrag.chunking.semantic import split_units

from .conftest import make_doc

ENC = tiktoken.get_encoding("cl100k_base")


def test_fixed_windows_respect_size_and_overlap():
    words = " ".join(f"word{i}" for i in range(2000))
    doc = make_doc("d.md", [(["Doc"], words)])
    chunks = FixedChunker(chunk_tokens=100, overlap_tokens=20).chunk(doc)
    assert len(chunks) > 5
    assert all(len(ENC.encode(c.text)) <= 100 for c in chunks)
    # consecutive windows share their overlap
    tail = ENC.decode(ENC.encode(chunks[0].text)[-20:])
    assert chunks[1].text.startswith(tail)
    assert all(c.strategy == "fixed" and not c.embed_with_heading for c in chunks)


def test_fixed_reports_heading_of_window_start():
    doc = make_doc("d.md", [(["A"], "alpha " * 300), (["A", "B"], "beta " * 300)])
    chunks = FixedChunker(chunk_tokens=200, overlap_tokens=0).chunk(doc)
    assert chunks[0].heading == "A"
    assert chunks[-1].heading == "A > B"


def test_recursive_never_straddles_sections_and_merges_tiny_ones():
    doc = make_doc(
        "d.md",
        [
            (["Guide"], "Short intro."),  # < min_chars: folded into the next section
            (["Guide", "Install"], "Install it with pip. " * 30),
            (["Guide", "Usage"], "Call the function. " * 120),
        ],
    )
    chunks = RecursiveChunker(chunk_chars=500, overlap_chars=50, min_chars=100).chunk(doc)
    assert chunks[0].heading == "Guide > Install"
    assert chunks[0].text.startswith("Guide\nShort intro.")
    for c in chunks:
        assert ("Install it" in c.text) != ("Call the function" in c.text)
        assert 100 <= c.char_count <= 500 + 100  # slivers are absorbed by a neighbour
    assert chunks[-1].embed_text.startswith("Guide > Usage\n\n")


def test_chunk_ids_are_deterministic():
    doc = make_doc("d.md", [(["A"], "some text " * 200)])
    a = [c.chunk_id for c in RecursiveChunker().chunk(doc)]
    b = [c.chunk_id for c in RecursiveChunker().chunk(doc)]
    assert a == b and len(set(a)) == len(a)


def test_split_units_keeps_code_blocks_whole():
    text = "First sentence. Second one.\n\n```python\nx = 1\n\ny = 2\n```\n\n- item a\n- item b"
    units = split_units(text)
    assert units == [
        "First sentence.",
        "Second one.",
        "```python\nx = 1\n\ny = 2\n```",
        "- item a",
        "- item b",
    ]


def test_semantic_splits_on_topic_shift(embedder):
    cooking = " ".join(f"Bake the bread with flour yeast and oven heat {i}." for i in range(8))
    network = " ".join(
        f"Route packets through tcp sockets and network ports {i}." for i in range(8)
    )
    doc = make_doc("d.md", [(["Mixed"], f"{cooking} {network}")])
    chunks = SemanticChunker(embedder, breakpoint_percentile=90, min_chars=50).chunk(doc)
    assert len(chunks) >= 2
    first_network = next(i for i, c in enumerate(chunks) if "tcp" in c.text)
    assert "flour" not in chunks[first_network].text  # the topic boundary is a chunk boundary
    assert all("tcp" not in c.text for c in chunks[:first_network])


def test_semantic_respects_max_chars(embedder):
    doc = make_doc("d.md", [(["X"], "Same words repeated here. " * 400)])
    chunks = SemanticChunker(embedder, min_chars=100, max_chars=800).chunk(doc)
    assert all(c.char_count <= 800 + 100 for c in chunks)  # tail fold may add < min_chars
