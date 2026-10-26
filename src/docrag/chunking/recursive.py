"""Structure-aware: split within each section, so chunks never straddle two headings."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from docrag.chunking.base import Chunker, merge_page_pieces
from docrag.models import Chunk, Document, Section

# Prefer breaking between code blocks / paragraphs, then lines, then sentences.
SEPARATORS = ["\n```\n", "\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(Chunker):
    name = "recursive"

    def __init__(self, chunk_chars: int = 1500, overlap_chars: int = 200, min_chars: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_chars,
            chunk_overlap=overlap_chars,
            separators=SEPARATORS,
            keep_separator="end",
        )
        self.min_chars = min_chars

    def _merge_small(self, sections: list[Section]) -> list[Section]:
        """Fold tiny sections (a one-line intro under a heading) into the next section.

        The small section's heading is kept inline so its content stays findable.
        """
        out: list[Section] = []
        carry = ""
        for s in sections:
            text = f"{carry}{s.text}" if carry else s.text
            if len(text) < self.min_chars and s is not sections[-1]:
                carry = f"{s.heading_path[-1] if s.heading_path else ''}\n{text}\n\n"
                continue
            out.append(Section(heading_path=s.heading_path, text=text, page=s.page))
            carry = ""
        return out

    def _absorb_small(self, splits: list[str]) -> list[str]:
        """The splitter can still leave slivers (a carried intro, a section's last line);
        attach each one to its following split, or to the previous one at the end."""
        out: list[str] = []
        pending = ""
        for text in splits:
            text = f"{pending}\n{text}" if pending else text
            pending = ""
            if len(text) < self.min_chars:
                pending = text
            else:
                out.append(text)
        if pending:
            if out:
                out[-1] = f"{out[-1]}\n{pending}"
            else:
                out.append(pending)
        return out

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = []
        for section in self._merge_small(merge_page_pieces(doc.sections)):
            for text in self._absorb_small(self.splitter.split_text(section.text)):
                pieces.append((text, section.heading, section.page))
        return self.make_chunks(doc, pieces)
