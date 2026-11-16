import uuid
from typing import Any, Optional

from app.adapters.embeddings.base import BaseEmbeddingAdapter
from app.adapters.vector_store.base import BaseVectorStoreAdapter
from app.core.exceptions import (
    DuplicateFileError,
    KnowledgeBaseNotFoundError,
    RAGException,
)
from app.core.logging import get_logger
from app.ingestion.chunkers import get_chunker
from app.ingestion.deduplication import validate_file_bytes
from app.ingestion.parsers import get_parser_for_file
from app.models.domain import Document, DocumentType
from app.models.pipeline_config import PipelineConfig
from app.storage.metadata_store import MetadataStore

logger = get_logger("ingestion_pipeline")


def _detect_doc_type(filename: str) -> DocumentType:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return DocumentType.PDF
    elif lower.endswith(".docx"):
        return DocumentType.DOCX
    elif lower.endswith(".md") or lower.endswith(".markdown"):
        return DocumentType.MARKDOWN
    elif lower.endswith(".txt"):
        return DocumentType.TXT
    return DocumentType.UNKNOWN


class IngestionPipeline:
    """
    Coordinates file validation, parsing, chunking, embedding, vector storage,
    and metadata persistence.
    """

    def __init__(
        self,
        metadata_store: MetadataStore,
        embedding_adapter: BaseEmbeddingAdapter,
        vector_adapter: BaseVectorStoreAdapter,
    ):
        self.store = metadata_store
        self.embedding_adapter = embedding_adapter
        self.vector_adapter = vector_adapter

    async def ingest_file(
        self,
        kb_id: str,
        filename: str,
        file_bytes: bytes,
        custom_fields: Optional[dict[str, Any]] = None,
        config: Optional[PipelineConfig] = None,
    ) -> Document:
        kb = self.store.get_kb(kb_id)
        if not kb:
            raise KnowledgeBaseNotFoundError(kb_id)

        if not config:
            config = self.store.get_active_config()

        logger.info(f"Ingesting file '{filename}' into KB '{kb_id}' using config {config.version}")

        # 1. Validation & Deduplication Check
        content_hash = validate_file_bytes(
            file_bytes=file_bytes,
            filename=filename,
            max_size_mb=config.ingestion.max_file_size_mb,
        )

        existing_doc = self.store.get_document_by_hash(kb_id, content_hash)
        if existing_doc:
            mode = config.ingestion.deduplication_mode
            if mode == "reject":
                raise DuplicateFileError(filename, content_hash)
            elif mode == "skip":
                logger.info(f"Skipping duplicate file '{filename}' (Doc ID: {existing_doc.id})")
                return existing_doc
            elif mode == "update":
                logger.info(f"Updating existing document '{existing_doc.id}' for file '{filename}'")
                await self.vector_adapter.delete_by_document(kb_id, existing_doc.id)
                self.store.delete_document(existing_doc.id)

        # 2. Parsing
        parser = get_parser_for_file(filename)
        parsed_doc = parser.parse(file_bytes, filename)

        doc_id = str(uuid.uuid4())
        doc_type = _detect_doc_type(filename)

        # 3. Chunking
        chunker = get_chunker(config.chunking.strategy)
        chunks = chunker.chunk(
            parsed_doc=parsed_doc,
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            custom_fields=custom_fields,
        )

        if not chunks:
            raise RAGException(f"Chunking yielded 0 chunks for file '{filename}'", status_code=400)

        # 4. Generate Embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = await self.embedding_adapter.embed_documents(chunk_texts)
        for c, emb in zip(chunks, embeddings):
            c.embedding = emb

        # 5. Upsert to Vector Store
        await self.vector_adapter.upsert_chunks(kb_id, chunks)

        # 6. Save Document First (satisfying foreign key), then Chunks
        doc = Document(
            id=doc_id,
            knowledge_base_id=kb_id,
            filename=filename,
            content_type=doc_type,
            content_hash=content_hash,
            file_size_bytes=len(file_bytes),
            page_count=len(parsed_doc.pages) if parsed_doc.pages else 1,
            section_count=len(parsed_doc.sections) if parsed_doc.sections else 1,
            custom_fields=custom_fields or {},
            status="indexed",
        )
        self.store.create_document(doc)
        self.store.save_chunks(chunks)

        logger.info(
            f"Successfully ingested '{filename}' (Doc: {doc_id}): {len(chunks)} chunks created."
        )
        return doc
