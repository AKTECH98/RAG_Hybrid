"""Raw + processed document storage.

data/raw/<doc_id>              original file, byte for byte
data/processed/<doc_id>.json   normalized Document

Keeping the originals means re-indexing (new chunker, new loader fix) never needs a re-upload.
"""

import shutil
from pathlib import Path, PurePosixPath

from docrag.config import Settings, get_settings
from docrag.ingest.loaders import load_document, supported
from docrag.models import Document


def validate_doc_id(doc_id: str) -> str:
    """doc_ids become file paths, so reject anything that could escape the data dir."""
    p = PurePosixPath(doc_id)
    if not doc_id or p.is_absolute() or ".." in p.parts or "\\" in doc_id:
        raise ValueError(f"Invalid doc_id: {doc_id!r}")
    if not supported(Path(doc_id)):
        raise ValueError(f"Unsupported file type: {doc_id!r}")
    return str(p)


class DocumentStore:
    def __init__(self, settings: Settings | None = None):
        s = settings or get_settings()
        self.raw_dir = s.raw_dir
        self.processed_dir = s.processed_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _raw_path(self, doc_id: str) -> Path:
        return self.raw_dir / doc_id

    def _processed_path(self, doc_id: str) -> Path:
        return self.processed_dir / f"{doc_id}.json"

    def ingest_file(self, path: Path, doc_id: str) -> Document:
        """Copy a file into raw storage, parse it, and save the processed document."""
        doc_id = validate_doc_id(doc_id)
        raw = self._raw_path(doc_id)
        raw.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != raw.resolve():
            shutil.copyfile(path, raw)
        return self.process(doc_id)

    def ingest_bytes(self, data: bytes, doc_id: str) -> Document:
        doc_id = validate_doc_id(doc_id)
        raw = self._raw_path(doc_id)
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(data)
        return self.process(doc_id)

    def process(self, doc_id: str) -> Document:
        """(Re)parse the raw copy of a document."""
        raw = self._raw_path(doc_id)
        doc = load_document(raw, doc_id, source_path=str(raw.relative_to(self.raw_dir.parent)))
        out = self._processed_path(doc_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc.model_dump_json(indent=1))
        return doc

    def get(self, doc_id: str) -> Document | None:
        path = self._processed_path(doc_id)
        return Document.model_validate_json(path.read_text()) if path.exists() else None

    def list(self) -> list[Document]:
        return [
            Document.model_validate_json(p.read_text())
            for p in sorted(self.processed_dir.rglob("*.json"))
        ]

    def delete(self, doc_id: str) -> bool:
        found = False
        for path in (self._raw_path(doc_id), self._processed_path(doc_id)):
            if path.exists():
                path.unlink()
                found = True
        return found
