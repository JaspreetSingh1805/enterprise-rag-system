from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.models.domain import (
    ChunkingStrategyType,
    EmbeddingProviderType,
    LLMProviderType,
    SearchMode,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


DEFAULT_RAG_SYSTEM_PROMPT = """You are an accurate, honest AI assistant answering questions based strictly on the provided company knowledge base.
Rules:
1. Base your answer ONLY on the provided context facts.
2. If the context does not contain enough information to answer the question with certainty, you MUST state clearly: "I could not find information about this in the knowledge base."
3. Never invent facts, policies, or numbers not present in the context.
4. For every claim you make, cite the source document name and page/section in format [Doc: <filename>, Page: <page>, Section: <section>].
"""

DEFAULT_REWRITE_PROMPT = """You are a query reformulation assistant for a technical knowledge base.
Your job is to rewrite the user's question into an optimal, standalone search query.
Clarify pronouns, expand company abbreviations, and remove filler words.
Do not answer the question; output only the rewritten search query.
"""


class IngestionConfig(BaseModel):
    allowed_extensions: list[str] = Field(default=[".pdf", ".docx", ".txt", ".md"])
    max_file_size_mb: int = Field(default=50, ge=1, le=500)
    deduplication_mode: Literal["reject", "update", "skip"] = "reject"


class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategyType = ChunkingStrategyType.RECURSIVE
    chunk_size: int = Field(default=500, ge=50, le=4000)
    chunk_overlap: int = Field(default=50, ge=0, le=1000)
    separators: list[str] = Field(default=["\n\n", "\n", " ", ""])
    preserve_headings: bool = True

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )
        return self


class EmbeddingConfig(BaseModel):
    provider: EmbeddingProviderType = EmbeddingProviderType.LOCAL
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = Field(default=384, ge=16, le=4096)
    batch_size: int = Field(default=32, ge=1, le=256)


class RetrievalConfig(BaseModel):
    search_mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=200)


class RerankConfig(BaseModel):
    enabled: bool = False
    model_name: str = "flashrank"
    top_n: int = Field(default=3, ge=1, le=20)


class LLMConfig(BaseModel):
    provider: LLMProviderType = LLMProviderType.OPENAI
    model_name: str = "gpt-4o-mini"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=50, le=4096)
    system_prompt: str = DEFAULT_RAG_SYSTEM_PROMPT
    query_rewrite_enabled: bool = False
    query_rewrite_prompt: str = DEFAULT_REWRITE_PROMPT


class PipelineConfig(BaseModel):
    version: str = Field(pattern=r"^v[0-9]+(\.[0-9]+)*$")
    description: str = ""
    is_active: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
