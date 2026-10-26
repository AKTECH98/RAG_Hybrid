from docrag.index.bm25_store import tokenize

from .conftest import make_doc

SHARED = "Raise HTTPException with status_code 404 when the item is not found in the database. " * 5


def _docs():
    return [
        make_doc("a.md", [(["Errors"], SHARED), (["Errors", "Custom"], "Custom handlers " * 60)]),
        make_doc("b.md", [(["Copy"], SHARED)]),  # same passage in a second doc
        make_doc("c.md", [(["Timeouts"], "Configure httpx timeout values per client. " * 20)]),
    ]


def test_tokenizer_keeps_identifiers_and_their_parts():
    tokens = tokenize("Use response_model and HTTPException; see request.url.path for 422")
    for t in [
        "response_model",
        "response",
        "model",
        "httpexception",
        "http",
        "exception",
        "request.url.path",
        "url",
        "path",
        "422",
    ]:
        assert t in tokens
    assert "and" not in tokens and "for" not in tokens


def test_index_dedups_across_documents_and_stays_in_sync(indexer):
    (stats,) = indexer.index_documents(_docs(), ("recursive",))
    assert stats.duplicates == 1
    dup = stats.duplicate_examples[0]
    assert (dup.dropped_doc, dup.kept_doc) == ("b.md", "a.md") and dup.similarity > 0.95
    assert indexer.dedup_log.exists()

    report = indexer.check_sync("recursive")
    assert report.in_sync and report.chunk_store == stats.chunks == report.vector_store


def test_reindexing_a_document_replaces_its_chunks(indexer):
    indexer.index_documents(_docs(), ("recursive",))
    idx = indexer.index("recursive")
    before = set(idx.chunks.ids_for_doc("c.md"))

    updated = make_doc("c.md", [(["Timeouts"], "Timeouts now default to five seconds. " * 20)])
    indexer.index_documents([updated], ("recursive",))
    after = set(idx.chunks.ids_for_doc("c.md"))
    assert after and not (before & after)
    assert indexer.check_sync("recursive").in_sync
    assert idx.bm25.search("five seconds", 1)[0][0] in after  # BM25 rebuilt automatically


def test_remove_document_and_repair_drift(indexer):
    indexer.index_documents(_docs(), ("recursive", "fixed"))
    assert indexer.remove_document("a.md") > 0
    for s in ("recursive", "fixed"):
        assert indexer.check_sync(s).in_sync
        assert "a.md" not in indexer.index(s).chunks.doc_ids()

    # simulate a crash between the Chroma write and the chunk-store write
    idx = indexer.index("recursive")
    lost = next(iter(idx.chunks.chunks))
    idx.vectors.delete_ids([lost])
    assert indexer.check_sync("recursive").missing_in_vectors == [lost]
    assert indexer.repair("recursive").in_sync


def test_chunk_store_persists_to_jsonl(indexer, settings):
    indexer.index_documents(_docs(), ("semantic",))
    from docrag.index.chunk_store import ChunkStore

    reloaded = ChunkStore("semantic", settings)
    assert set(reloaded.chunks) == set(indexer.index("semantic").chunks.chunks)


def test_concurrent_queries_do_not_deadlock(indexer, embedder):
    """Regression: 8 threads querying Chroma's embedded client at once used to hang."""
    from concurrent.futures import ThreadPoolExecutor

    indexer.index_documents(_docs(), ("recursive",))
    vectors = indexer.index("recursive").vectors
    queries = [embedder.embed([f"timeout handlers query {i}"])[0] for i in range(64)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda v: vectors.query(v, 3), queries))
    assert len(results) == 64 and all(len(r) == 3 for r in results)
