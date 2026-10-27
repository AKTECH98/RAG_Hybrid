"""Ingest the sample corpus and build all indexes.

Usage:
  uv run python scripts/seed.py                       # all strategies
  uv run python scripts/seed.py --strategy recursive  # just one
  uv run python scripts/seed.py --reset               # wipe indexes first
  uv run python scripts/seed.py --embedder local      # index with another embedder
  uv run python scripts/seed.py --reuse-chunks        # only sync vectors to existing chunks

Each embedder gets its own index under data/index/<provider>-<model>/.
"""

import argparse
import time
from collections import Counter
from pathlib import Path

from docrag.config import CHUNK_STRATEGIES, EMBEDDING_PROVIDERS, get_settings
from docrag.index.embedder import get_embedder
from docrag.index.indexer import Indexer
from docrag.ingest.loaders import supported
from docrag.ingest.store import DocumentStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=[*CHUNK_STRATEGIES, "all"], default="all")
    parser.add_argument("--corpus", type=Path, default=None, help="defaults to ./corpus")
    parser.add_argument("--reset", action="store_true", help="drop existing indexes first")
    parser.add_argument(
        "--embedder", choices=EMBEDDING_PROVIDERS, help="defaults to DOCRAG_EMBEDDING_PROVIDER"
    )
    parser.add_argument("--embedding-model", help="override the provider's default model")
    parser.add_argument(
        "--reuse-chunks",
        action="store_true",
        help="strategies that already have a chunk store are not re-chunked; their vectors are "
        "synced to it instead (fast with a warm embedding cache, e.g. a fresh Chroma server)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.embedder:
        settings = settings.with_embedder(args.embedder, args.embedding_model)
    print(f"Embedder: {settings.embedding_provider} / {settings.embedding_model_name}")
    corpus = args.corpus or settings.corpus_dir
    strategies = CHUNK_STRATEGIES if args.strategy == "all" else (args.strategy,)

    t0 = time.time()
    indexer = Indexer(get_embedder(settings), settings)
    if args.reset:
        settings.index_dir.joinpath("dedup_log.jsonl").unlink(missing_ok=True)
        for s in strategies:
            indexer.reset(s)

    if args.reuse_chunks and not args.reset:
        reuse = tuple(s for s in strategies if indexer.index(s).chunks.chunks)
        for strategy in reuse:
            sync = indexer.repair(strategy)
            print(
                f"{strategy:<10} reused {sync.chunk_store} chunks; vectors {sync.vector_store} "
                f"({'in sync' if sync.in_sync else 'DRIFT'})"
            )
        strategies = tuple(s for s in strategies if s not in reuse)
        if not strategies:
            print(f"\nDone in {time.time() - t0:.1f}s. Data in {settings.data_dir}")
            return

    files = sorted(p for p in corpus.rglob("*") if p.is_file() and supported(p))
    if not files:
        raise SystemExit(f"No supported files in {corpus}. Run scripts/fetch_corpus.py first.")
    store = DocumentStore(settings)
    docs = [store.ingest_file(p, p.relative_to(corpus).as_posix()) for p in files]
    formats = Counter(d.format for d in docs)
    print(f"Loaded {len(docs)} documents {dict(formats)} in {time.time() - t0:.1f}s")

    print(f"\n{'strategy':<10} {'chunks':>7} {'dupes':>6} {'vectors':>8} {'bm25':>6}  sync   time")
    for strategy in strategies:
        t = time.time()
        (stats,) = indexer.index_documents(docs, (strategy,))
        sync = indexer.check_sync(strategy)
        print(
            f"{strategy:<10} {stats.chunks:>7} {stats.duplicates:>6} {sync.vector_store:>8} "
            f"{sync.bm25:>6}  {'ok' if sync.in_sync else 'DRIFT':<5} {time.time() - t:>5.1f}s"
        )
        for d in stats.duplicate_examples[:3]:
            print(f"    dup {d.similarity:.3f}: {d.dropped_doc} ~ {d.kept_doc}")
    print(f"\nDone in {time.time() - t0:.1f}s. Data in {settings.data_dir}")


if __name__ == "__main__":
    main()
