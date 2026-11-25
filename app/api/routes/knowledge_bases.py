import uuid
from typing import Optional

from fastapi import APIRouter, Depends, status

from app.adapters.embeddings.base import BaseEmbeddingAdapter
from app.adapters.factory import AdapterRegistry
from app.adapters.vector_store.base import BaseVectorStoreAdapter
from app.api.dependencies import (
    get_adapter_registry,
    get_config_manager,
    get_embedding_adapter,
    get_job_queue,
    get_metadata_store,
    get_vector_adapter,
)
from app.core.exceptions import KnowledgeBaseNotFoundError
from app.core.logging import get_logger
from app.models.api_schemas import (
    CreateKnowledgeBaseRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    ReindexKnowledgeBaseRequest,
    ReindexKnowledgeBaseResponse,
    UpdateKnowledgeBaseRequest,
)
from app.models.domain import EmbeddingProviderType, Job, JobStatus, KnowledgeBase
from app.models.pipeline_config import EmbeddingConfig
from app.pipeline.config_manager import DynamicConfigManager
from app.storage.job_queue import BackgroundJobQueue
from app.storage.metadata_store import MetadataStore

logger = get_logger("knowledge_bases_api")

router = APIRouter(prefix="/knowledge-bases", tags=["Knowledge Bases"])


@router.post("/", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    req: CreateKnowledgeBaseRequest,
    store: MetadataStore = Depends(get_metadata_store),
    vector_adapter: BaseVectorStoreAdapter = Depends(get_vector_adapter),
    embedding_adapter: BaseEmbeddingAdapter = Depends(get_embedding_adapter),
):
    model_name = req.embedding_model or embedding_adapter.model_name
    dimension = req.embedding_dimension or embedding_adapter.dimension

    kb = KnowledgeBase(
        id=req.id,
        name=req.name,
        description=req.description,
        embedding_model=model_name,
        embedding_dimension=dimension,
    )
    created_kb = store.create_kb(kb)
    await vector_adapter.create_collection(kb.id, dimension)
    return KnowledgeBaseResponse(knowledge_base=created_kb)


@router.get("/", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    store: MetadataStore = Depends(get_metadata_store),
):
    kbs = store.list_kbs()
    return KnowledgeBaseListResponse(knowledge_bases=kbs, total=len(kbs))


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: str,
    store: MetadataStore = Depends(get_metadata_store),
):
    kb = store.get_kb(kb_id)
    if not kb:
        raise KnowledgeBaseNotFoundError(kb_id)
    return KnowledgeBaseResponse(knowledge_base=kb)


@router.put("/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: str,
    req: UpdateKnowledgeBaseRequest,
    store: MetadataStore = Depends(get_metadata_store),
):
    updated_kb = store.update_kb(kb_id, name=req.name, description=req.description)
    return KnowledgeBaseResponse(knowledge_base=updated_kb)


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    kb_id: str,
    store: MetadataStore = Depends(get_metadata_store),
    vector_adapter: BaseVectorStoreAdapter = Depends(get_vector_adapter),
):
    kb = store.get_kb(kb_id)
    if not kb:
        raise KnowledgeBaseNotFoundError(kb_id)

    # Delete all vectors in vector store
    await vector_adapter.delete_collection(kb_id)
    # Delete metadata, documents, and chunks
    store.delete_kb(kb_id)
    return None


@router.post(
    "/{kb_id}/reindex",
    response_model=ReindexKnowledgeBaseResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reindex_knowledge_base(
    kb_id: str,
    req: Optional[ReindexKnowledgeBaseRequest] = None,
    store: MetadataStore = Depends(get_metadata_store),
    config_mgr: DynamicConfigManager = Depends(get_config_manager),
    adapter_registry: AdapterRegistry = Depends(get_adapter_registry),
    job_queue: BackgroundJobQueue = Depends(get_job_queue),
):
    """
    Triggers an asynchronous re-indexing workflow for the Knowledge Base.
    Safely drops old vectors, recreates collection with target embedding dimension,
    re-embeds all existing chunks from SQLite, and updates Knowledge Base schema metadata.
    """
    kb = store.get_kb(kb_id)
    if not kb:
        raise KnowledgeBaseNotFoundError(kb_id)

    active_cfg = store.get_active_config()

    # Determine target embedding provider, model, and dimension
    target_provider = (
        EmbeddingProviderType(req.embedding_provider)
        if req and req.embedding_provider
        else active_cfg.embedding.provider
    )
    target_model = (
        req.embedding_model if req and req.embedding_model else active_cfg.embedding.model_name
    )
    target_dim = (
        req.embedding_dimension
        if req and req.embedding_dimension
        else (
            1536
            if "openai" in str(target_provider).lower() or "text-embedding" in target_model
            else 384
        )
    )

    target_adapter = adapter_registry.get_embedding_adapter(
        EmbeddingConfig(
            provider=target_provider,
            model_name=target_model,
            dimension=target_dim,
        )
    )

    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        knowledge_base_id=kb_id,
        job_type="knowledge_base_reindex",
        status=JobStatus.QUEUED,
        progress_percent=0,
        config_version=active_cfg.version,
        details={
            "old_model": kb.embedding_model,
            "old_dimension": kb.embedding_dimension,
            "target_model": target_model,
            "target_dimension": target_dim,
            "target_provider": target_provider.value,
        },
    )

    async def _run_reindex(j: Job):
        logger.info(f"Executing reindex workflow for KB '{kb_id}' to model '{target_model}'")
        store.update_job(j.id, status=JobStatus.PROCESSING, progress_percent=25)
        count = await config_mgr.safe_reindex_knowledge_base(kb_id, target_adapter)
        store.update_job(
            j.id,
            status=JobStatus.COMPLETED,
            progress_percent=100,
            details={**j.details, "reindexed_chunks": count},
        )

    job_queue.submit_job(job, _run_reindex)

    return ReindexKnowledgeBaseResponse(
        job=job,
        knowledge_base_id=kb_id,
        target_embedding_model=target_model,
        target_embedding_dimension=target_dim,
        message=f"Reindexing job queued for Knowledge Base '{kb_id}'. Track status at /api/v1/jobs/{job_id}",
    )
