from fastapi import APIRouter, Depends, status

from app.adapters.embeddings.base import BaseEmbeddingAdapter
from app.adapters.vector_store.base import BaseVectorStoreAdapter
from app.api.dependencies import (
    get_embedding_adapter,
    get_metadata_store,
    get_vector_adapter,
)
from app.core.exceptions import KnowledgeBaseNotFoundError
from app.models.api_schemas import (
    CreateKnowledgeBaseRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    UpdateKnowledgeBaseRequest,
)
from app.models.domain import KnowledgeBase
from app.storage.metadata_store import MetadataStore

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
