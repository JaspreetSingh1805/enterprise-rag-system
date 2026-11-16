from typing import Any, Optional

from app.ingestion.chunkers.base import BaseChunker
from app.ingestion.parsers.base import ParsedDocument
from app.models.domain import Chunk


class StructureAwareChunker(BaseChunker):
    """
    Strategy B: Structure/Markdown-Aware Hierarchical Chunking.
    Splits text along document hierarchy (Markdown headers, DOCX styles, sections)
    and prefixes parent section titles to child chunks.

    Trade-off:
    - Superior semantic cohesion; preserves document context in citations.
    - Chunks have variable length (short sections produce smaller chunks).
    - Best suited for technical manuals, policies, API docs, and FAQs.
    """

    @property
    def strategy_name(self) -> str:
        return "structure"

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

        sections = parsed_doc.sections
        # If no explicit sections found, split into paragraphs as fallback sections
        if not sections:
            paragraphs = [p.strip() for p in parsed_doc.full_text.split("\n\n") if p.strip()]
            for p_idx, p in enumerate(paragraphs):
                c_id = f"{doc_id}_{chunk_idx}"
                metadata = {
                    "strategy": self.strategy_name,
                    "filename": parsed_doc.filename,
                    "section_title": f"Paragraph {p_idx + 1}",
                }
                if custom_fields:
                    metadata.update(custom_fields)

                chunks.append(
                    Chunk(
                        id=c_id,
                        document_id=doc_id,
                        knowledge_base_id=kb_id,
                        text=p,
                        chunk_index=chunk_idx,
                        page_number=1,
                        section_title=f"Paragraph {p_idx + 1}",
                        metadata=metadata,
                    )
                )
                chunk_idx += 1
            return chunks

        for sec in sections:
            header_prefix = f"[{sec.title}]\n"
            sec_full = f"{header_prefix}{sec.text}".strip()

            if len(sec_full) <= chunk_size:
                c_id = f"{doc_id}_{chunk_idx}"
                metadata = {
                    "strategy": self.strategy_name,
                    "filename": parsed_doc.filename,
                    "section_title": sec.title,
                    "section_level": sec.level,
                }
                if custom_fields:
                    metadata.update(custom_fields)

                chunks.append(
                    Chunk(
                        id=c_id,
                        document_id=doc_id,
                        knowledge_base_id=kb_id,
                        text=sec_full,
                        chunk_index=chunk_idx,
                        page_number=1,
                        section_title=sec.title,
                        metadata=metadata,
                    )
                )
                chunk_idx += 1
            else:
                # Sub-split large section into paragraphs or smaller blocks
                paragraphs = [p.strip() for p in sec.text.split("\n\n") if p.strip()]
                current_block = []
                current_len = len(header_prefix)

                for p in paragraphs:
                    p_len = len(p) + 2
                    if current_len + p_len > chunk_size and current_block:
                        c_id = f"{doc_id}_{chunk_idx}"
                        block_text = f"{header_prefix}" + "\n\n".join(current_block)
                        metadata = {
                            "strategy": self.strategy_name,
                            "filename": parsed_doc.filename,
                            "section_title": sec.title,
                            "section_level": sec.level,
                        }
                        if custom_fields:
                            metadata.update(custom_fields)

                        chunks.append(
                            Chunk(
                                id=c_id,
                                document_id=doc_id,
                                knowledge_base_id=kb_id,
                                text=block_text,
                                chunk_index=chunk_idx,
                                page_number=1,
                                section_title=sec.title,
                                metadata=metadata,
                            )
                        )
                        chunk_idx += 1
                        current_block = [p]
                        current_len = len(header_prefix) + p_len
                    else:
                        current_block.append(p)
                        current_len += p_len

                if current_block:
                    c_id = f"{doc_id}_{chunk_idx}"
                    block_text = f"{header_prefix}" + "\n\n".join(current_block)
                    metadata = {
                        "strategy": self.strategy_name,
                        "filename": parsed_doc.filename,
                        "section_title": sec.title,
                        "section_level": sec.level,
                    }
                    if custom_fields:
                        metadata.update(custom_fields)

                    chunks.append(
                        Chunk(
                            id=c_id,
                            document_id=doc_id,
                            knowledge_base_id=kb_id,
                            text=block_text,
                            chunk_index=chunk_idx,
                            page_number=1,
                            section_title=sec.title,
                            metadata=metadata,
                        )
                    )
                    chunk_idx += 1

        return chunks
