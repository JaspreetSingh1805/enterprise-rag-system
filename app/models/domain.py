from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class SearchMode(str, Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class ChunkingStrategyType(str, Enum):
    RECURSIVE = "recursive"
    STRUCTURE = "structure"


class LLMProviderType(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    MOCK = "mock"


class EmbeddingProviderType(str, Enum):
    LOCAL = "local"
    OPENAI = "openai"
    MOCK = "mock"


class VectorStoreType(str, Enum):
    QDRANT = "qdrant"
    MEMORY = "memory"


class KnowledgeBase(BaseModel):
    id: str
    name: str
    description: str = ""
    embedding_model: str
    embedding_dimension: int
    document_count: int = 0
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Document(BaseModel):
    id: str
    knowledge_base_id: str
    filename: str
    content_type: DocumentType
    content_hash: str  # SHA-256
    file_size_bytes: int
    page_count: int = 1
    section_count: int = 1
    custom_fields: dict[str, Any] = Field(default_factory=dict)
    uploaded_at: datetime = Field(default_factory=utc_now)
    status: str = "indexed"


class Chunk(BaseModel):
    id: str
    document_id: str
    knowledge_base_id: str
    text: str
    chunk_index: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None


class Citation(BaseModel):
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    snippet: str
    score: float


class ExecutionStageTrace(BaseModel):
    stage_name: str
    duration_ms: float
    details: dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    trace_id: str
    total_latency_ms: float
    config_version: str
    stages: list[ExecutionStageTrace] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class QueryResult(BaseModel):
    question: str
    answer: str
    is_grounded: bool
    is_found: bool
    citations: list[Citation] = Field(default_factory=list)
    trace: ExecutionTrace
    config_version: str


class Job(BaseModel):
    id: str
    knowledge_base_id: str
    job_type: str
    status: JobStatus = JobStatus.QUEUED
    progress_percent: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    config_version: str = "v1"
    details: dict[str, Any] = Field(default_factory=dict)
