from typing import Any, Optional

from app.ingestion.chunkers.base import BaseChunker
from app.ingestion.parsers.base import ParsedDocument
from app.models.domain import Chunk


class RecursiveCharacterChunker(BaseChunker):
    """
    Strategy A: Recursive Character Chunking.
    Recursively splits text on natural delimiters (paragraphs -> lines -> words -> chars)
    to achieve uniform chunk sizes with configurable sliding window overlap.

    Trade-off:
    - High throughput, predictable token lengths, ideal for continuous prose.
    - May split tables, bullet lists, or structural units mid-sentence.
    """

    def __init__(self, separators: list[str] | None = None):
        self.separators = separators or ["\n\n", "\n", " ", ""]

    @property
    def strategy_name(self) -> str:
        return "recursive"

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        if not text:
            return []

        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        step = max(1, chunk_size - chunk_overlap)
        chunks: list[str] = []

        # Find best separator
        separator = self.separators[-1]
        for sep in self.separators:
            if sep in text:
                separator = sep
                break

        if separator:
            splits = text.split(separator)
            current_chunk: list[str] = []
            current_len = 0

            for part in splits:
                part_len = len(part) + len(separator)
                if current_len + part_len > chunk_size and current_chunk:
                    joined = separator.join(current_chunk).strip()
                    if joined:
                        chunks.append(joined)
                    # Retain overlap from end of current_chunk
                    overlap_parts: list[str] = []
                    overlap_len = 0
                    for op in reversed(current_chunk):
                        if overlap_len + len(op) + len(separator) <= chunk_overlap:
                            overlap_parts.insert(0, op)
                            overlap_len += len(op) + len(separator)
                        else:
                            break
                    current_chunk = overlap_parts
                    current_len = overlap_len

                current_chunk.append(part)
                current_len += part_len

            if current_chunk:
                joined = separator.join(current_chunk).strip()
                if joined:
                    chunks.append(joined)
        else:
            # Fallback character window
            for i in range(0, len(text), step):
                c = text[i : i + chunk_size].strip()
                if c:
                    chunks.append(c)

        return chunks

    def chunk(
        self,
        parsed_doc: ParsedDocument,
        kb_id: str,
        doc_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        custom_fields: Optional[dict[str, Any]] = None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_idx = 0

        # If pages exist (e.g. PDF), chunk per page to maintain accurate citations
        if parsed_doc.pages and len(parsed_doc.pages) > 1:
            for page in parsed_doc.pages:
                page_chunks = self._split_text(page.text, chunk_size, chunk_overlap)
                for txt in page_chunks:
                    c_id = f"{doc_id}_{chunk_idx}"
                    metadata = {
                        "strategy": self.strategy_name,
                        "filename": parsed_doc.filename,
                        "page_number": page.page_number,
                    }
                    if custom_fields:
                        metadata.update(custom_fields)

                    chunks.append(
                        Chunk(
                            id=c_id,
                            document_id=doc_id,
                            knowledge_base_id=kb_id,
                            text=txt,
                            chunk_index=chunk_idx,
                            page_number=page.page_number,
                            section_title=None,
                            metadata=metadata,
                        )
                    )
                    chunk_idx += 1
        else:
            # Document without multiple pages
            raw_chunks = self._split_text(parsed_doc.full_text, chunk_size, chunk_overlap)
            for txt in raw_chunks:
                c_id = f"{doc_id}_{chunk_idx}"
                metadata = {
                    "strategy": self.strategy_name,
                    "filename": parsed_doc.filename,
                    "page_number": 1,
                }
                if custom_fields:
                    metadata.update(custom_fields)

                chunks.append(
                    Chunk(
                        id=c_id,
                        document_id=doc_id,
                        knowledge_base_id=kb_id,
                        text=txt,
                        chunk_index=chunk_idx,
                        page_number=1,
                        section_title=None,
                        metadata=metadata,
                    )
                )
                chunk_idx += 1

        return chunks
