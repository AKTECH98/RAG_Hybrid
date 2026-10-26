import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from docrag.config import Settings
from docrag.index.embedder import HashEmbedder
from docrag.index.indexer import Indexer
from docrag.models import Document, Section


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "data", openai_api_key=None)


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def indexer(settings, embedder, tmp_path):
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
    )
    return Indexer(embedder, settings, client=client)


def make_doc(doc_id: str, sections: list[tuple[list[str], str]], fmt: str = "md") -> Document:
    return Document(
        doc_id=doc_id,
        source_path=doc_id,
        format=fmt,
        title=sections[0][0][0] if sections and sections[0][0] else doc_id,
        sections=[Section(heading_path=h, text=t) for h, t in sections],
    )
