"""Ingest the sample corpus and build all indexes.

Usage:
  uv run python scripts/seed.py                       # all strategies
  uv run python scripts/seed.py --strategy recursive  # just one
  uv run python scripts/seed.py --reset               # wipe indexes first
"""

import argparse
import time
from collections import Counter
from pathlib import Path

from docrag.config import CHUNK_STRATEGIES, get_settings
from docrag.index.embedder import get_embedder
from docrag.index.indexer import Indexer
from docrag.ingest.loaders import supported
from docrag.ingest.store import DocumentStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=[*CHUNK_STRATEGIES, "all"], default="all")
    parser.add_argument("--corpus", type=Path, default=None, help="defaults to ./corpus")
    parser.add_argument("--reset", action="store_true", help="drop existing indexes first")
    args = parser.parse_args()

    settings = get_settings()
    corpus = args.corpus or settings.corpus_dir
    strategies = CHUNK_STRATEGIES if args.strategy == "all" else (args.strategy,)
    files = sorted(p for p in corpus.rglob("*") if p.is_file() and supported(p))
    if not files:
        raise SystemExit(f"No supported files in {corpus}. Run scripts/fetch_corpus.py first.")

    t0 = time.time()
    store = DocumentStore(settings)
    docs = [store.ingest_file(p, p.relative_to(corpus).as_posix()) for p in files]
    formats = Counter(d.format for d in docs)
    print(f"Loaded {len(docs)} documents {dict(formats)} in {time.time() - t0:.1f}s")

    indexer = Indexer(get_embedder(settings), settings)
    if args.reset:
        settings.data_dir.joinpath("dedup_log.jsonl").unlink(missing_ok=True)
        for s in strategies:
            indexer.reset(s)

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
