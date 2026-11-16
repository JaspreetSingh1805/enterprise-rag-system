import json
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status

from app.adapters.vector_store.base import BaseVectorStoreAdapter
from app.api.dependencies import (
    get_ingestion_pipeline,
    get_job_queue,
    get_metadata_store,
    get_vector_adapter,
)
from app.core.exceptions import DocumentNotFoundError, KnowledgeBaseNotFoundError, RAGException
from app.ingestion.pipeline import IngestionPipeline
from app.models.api_schemas import (
    DocumentListResponse,
    DocumentResponse,
    JobResponse,
    ReindexDocumentResponse,
)
from app.models.domain import Job, JobStatus
from app.storage.job_queue import BackgroundJobQueue
from app.storage.metadata_store import MetadataStore

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    knowledge_base_id: str = Form(...),
    file: UploadFile = File(...),
    custom_fields: Optional[str] = Form(
        default=None, description="JSON string of custom fields e.g. {'dept': 'HR'}"
    ),
    store: MetadataStore = Depends(get_metadata_store),
    job_queue: BackgroundJobQueue = Depends(get_job_queue),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    kb = store.get_kb(knowledge_base_id)
    if not kb:
        raise KnowledgeBaseNotFoundError(knowledge_base_id)

    custom_dict: dict[str, Any] = {}
    if custom_fields:
        try:
            custom_dict = json.loads(custom_fields)
        except json.JSONDecodeError:
            raise RAGException("Invalid JSON format in custom_fields", status_code=400)

    file_bytes = await file.read()
    filename = file.filename or "uploaded_file"

    active_cfg = store.get_active_config()
    job_id = str(uuid.uuid4())

    job = Job(
        id=job_id,
        knowledge_base_id=knowledge_base_id,
        job_type="document_ingestion",
        status=JobStatus.QUEUED,
        config_version=active_cfg.version,
        details={"filename": filename, "file_size": len(file_bytes), "custom_fields": custom_dict},
    )

    # Define background execution coroutine
    async def _ingest_task(j: Job):
        doc = await pipeline.ingest_file(
            kb_id=knowledge_base_id,
            filename=filename,
            file_bytes=file_bytes,
            custom_fields=custom_dict,
            config=active_cfg,
        )
        j.details["document_id"] = doc.id
        j.details["chunks_count"] = doc.page_count

    job_queue.submit_job(job, _ingest_task)
    return JobResponse(job=job)


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    knowledge_base_id: str = Query(..., description="Filter documents by Knowledge Base ID"),
    store: MetadataStore = Depends(get_metadata_store),
):
    kb = store.get_kb(knowledge_base_id)
    if not kb:
        raise KnowledgeBaseNotFoundError(knowledge_base_id)

    docs = store.list_documents(knowledge_base_id)
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    store: MetadataStore = Depends(get_metadata_store),
):
    doc = store.get_document(doc_id)
    if not doc:
        raise DocumentNotFoundError(doc_id)
    return DocumentResponse(document=doc)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    store: MetadataStore = Depends(get_metadata_store),
    vector_adapter: BaseVectorStoreAdapter = Depends(get_vector_adapter),
):
    doc = store.get_document(doc_id)
    if not doc:
        raise DocumentNotFoundError(doc_id)

    # Delete all vectors in vector store
    await vector_adapter.delete_by_document(doc.knowledge_base_id, doc.id)
    # Delete metadata and chunks in SQLite
    store.delete_document(doc.id)
    return None


@router.post("/{doc_id}/reindex", response_model=ReindexDocumentResponse)
async def reindex_document(
    doc_id: str,
    store: MetadataStore = Depends(get_metadata_store),
    vector_adapter: BaseVectorStoreAdapter = Depends(get_vector_adapter),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    doc = store.get_document(doc_id)
    if not doc:
        raise DocumentNotFoundError(doc_id)

    # 1. Fetch all stored chunks for this document
    chunks = store.get_chunks_for_doc(doc_id)
    if not chunks:
        raise RAGException(f"No chunks found for document '{doc_id}' to reindex", status_code=400)

    # 2. Delete existing vectors
    await vector_adapter.delete_by_document(doc.knowledge_base_id, doc_id)

    # 3. Generate new embeddings and re-upsert
    texts = [c.text for c in chunks]
    embeddings = await pipeline.embedding_adapter.embed_documents(texts)
    for c, emb in zip(chunks, embeddings):
        c.embedding = emb

    await vector_adapter.upsert_chunks(doc.knowledge_base_id, chunks)

    return ReindexDocumentResponse(
        document_id=doc_id,
        status="reindexed",
        chunks_count=len(chunks),
        message=f"Successfully reindexed {len(chunks)} chunks for document '{doc.filename}'",
    )
