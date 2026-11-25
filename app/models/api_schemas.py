from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.domain import Citation, Document, ExecutionTrace, Job, KnowledgeBase
from app.models.pipeline_config import PipelineConfig


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ==============================================================================
# Knowledge Base Schemas
# ==============================================================================


class CreateKnowledgeBaseRequest(BaseModel):
    id: str = Field(..., pattern=r"^[a-z0-9_-]+$", description="Unique lowercase slug identifier")
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    embedding_model: Optional[str] = Field(
        default=None, description="Override active default embedding model"
    )
    embedding_dimension: Optional[int] = Field(
        default=None, description="Dimension matching embedding model"
    )


class UpdateKnowledgeBaseRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)


class KnowledgeBaseResponse(BaseModel):
    knowledge_base: KnowledgeBase


class KnowledgeBaseListResponse(BaseModel):
    knowledge_bases: list[KnowledgeBase]
    total: int


class ReindexKnowledgeBaseRequest(BaseModel):
    embedding_provider: Optional[str] = Field(
        default=None,
        description="Optional new embedding provider ('local', 'openai', 'mock'). Defaults to active config.",
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Optional new embedding model name (e.g. 'all-MiniLM-L6-v2', 'text-embedding-3-small').",
    )
    embedding_dimension: Optional[int] = Field(
        default=None,
        description="Dimension matching the new embedding model (e.g. 384, 1536).",
    )


class ReindexKnowledgeBaseResponse(BaseModel):
    job: Job
    knowledge_base_id: str
    target_embedding_model: str
    target_embedding_dimension: int
    message: str


# ==============================================================================
# Document Schemas
# ==============================================================================


class DocumentResponse(BaseModel):
    document: Document


class DocumentListResponse(BaseModel):
    documents: list[Document]
    total: int


class ReindexDocumentResponse(BaseModel):
    document_id: str
    status: str
    chunks_count: int
    message: str


# ==============================================================================
# Pipeline Config Schemas
# ==============================================================================


class ConfigResponse(BaseModel):
    config: PipelineConfig


class ConfigListResponse(BaseModel):
    configs: list[PipelineConfig]
    active_version: str


class ActivateConfigResponse(BaseModel):
    version: str
    is_active: bool
    message: str


# ==============================================================================
# Query Schemas
# ==============================================================================


class QueryRequest(BaseModel):
    knowledge_base_id: str = Field(..., description="Target knowledge base slug")
    question: str = Field(..., min_length=2, max_length=1000)
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Metadata filters e.g. {'department': 'HR', 'content_type': 'pdf'}",
    )
    config_overrides: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional overrides for testing (e.g. search_mode, top_k, rerank_enabled)",
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    is_grounded: bool
    is_found: bool
    citations: list[Citation] = Field(default_factory=list)
    trace: ExecutionTrace
    config_version: str


# ==============================================================================
# Background Jobs & Health Schemas
# ==============================================================================


class JobResponse(BaseModel):
    job: Job


class JobListResponse(BaseModel):
    jobs: list[Job]
    total: int


class HealthResponse(BaseModel):
    status: str
    environment: str
    active_config_version: str
    vector_store_type: str
    vector_store_status: str
    timestamp: datetime = Field(default_factory=utc_now)
