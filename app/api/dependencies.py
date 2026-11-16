from typing import Optional

from fastapi import Depends

from app.adapters.embeddings.base import BaseEmbeddingAdapter
from app.adapters.embeddings.local_adapter import LocalEmbeddingAdapter
from app.adapters.embeddings.mock_adapter import MockEmbeddingAdapter
from app.adapters.embeddings.openai_adapter import OpenAIEmbeddingAdapter
from app.adapters.llm.base import BaseLLMAdapter
from app.adapters.llm.mock_adapter import MockLLMAdapter
from app.adapters.llm.ollama_adapter import OllamaAdapter
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.adapters.vector_store.base import BaseVectorStoreAdapter
from app.adapters.vector_store.in_memory_adapter import InMemoryVectorStoreAdapter
from app.adapters.vector_store.qdrant_adapter import QdrantVectorStoreAdapter
from app.core.config import settings
from app.ingestion.pipeline import IngestionPipeline
from app.pipeline.config_manager import DynamicConfigManager
from app.pipeline.rag_engine import RAGEngine
from app.storage.job_queue import BackgroundJobQueue
from app.storage.metadata_store import MetadataStore

# Global singletons
_metadata_store: Optional[MetadataStore] = None
_vector_adapter: Optional[BaseVectorStoreAdapter] = None
_embedding_adapter: Optional[BaseEmbeddingAdapter] = None
_llm_adapter: Optional[BaseLLMAdapter] = None
_job_queue: Optional[BackgroundJobQueue] = None


def get_metadata_store() -> MetadataStore:
    global _metadata_store
    if _metadata_store is None:
        db_path = settings.database_url.replace("sqlite:///", "")
        _metadata_store = MetadataStore(db_path=db_path)
    return _metadata_store


def get_vector_adapter() -> BaseVectorStoreAdapter:
    global _vector_adapter
    if _vector_adapter is None:
        if settings.vector_store_type == "qdrant":
            _vector_adapter = QdrantVectorStoreAdapter(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                path=settings.qdrant_path,
            )
        else:
            _vector_adapter = InMemoryVectorStoreAdapter()
    return _vector_adapter


def set_vector_adapter(adapter: BaseVectorStoreAdapter) -> None:
    """Override vector adapter (used in tests)."""
    global _vector_adapter
    _vector_adapter = adapter


def get_embedding_adapter() -> BaseEmbeddingAdapter:
    global _embedding_adapter
    if _embedding_adapter is None:
        provider = settings.default_embedding_provider
        if provider == "openai":
            _embedding_adapter = OpenAIEmbeddingAdapter(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                model_name=settings.openai_embedding_model,
            )
        elif provider == "local":
            _embedding_adapter = LocalEmbeddingAdapter(
                model_name=settings.local_embedding_model,
            )
        else:
            _embedding_adapter = MockEmbeddingAdapter()
    return _embedding_adapter


def set_embedding_adapter(adapter: BaseEmbeddingAdapter) -> None:
    """Override embedding adapter (used in tests)."""
    global _embedding_adapter
    _embedding_adapter = adapter


def get_llm_adapter() -> BaseLLMAdapter:
    global _llm_adapter
    if _llm_adapter is None:
        provider = settings.default_llm_provider
        if provider == "openai":
            _llm_adapter = OpenAIAdapter(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                model_name=settings.openai_model,
            )
        elif provider == "ollama":
            _llm_adapter = OllamaAdapter(
                base_url=settings.ollama_base_url,
                model_name=settings.ollama_model,
            )
        else:
            _llm_adapter = MockLLMAdapter()
    return _llm_adapter


def set_llm_adapter(adapter: BaseLLMAdapter) -> None:
    """Override LLM adapter (used in tests)."""
    global _llm_adapter
    _llm_adapter = adapter


def get_job_queue(
    store: MetadataStore = Depends(get_metadata_store),
) -> BackgroundJobQueue:
    return BackgroundJobQueue(store)


def get_ingestion_pipeline(
    store: MetadataStore = Depends(get_metadata_store),
    embedding_adapter: BaseEmbeddingAdapter = Depends(get_embedding_adapter),
    vector_adapter: BaseVectorStoreAdapter = Depends(get_vector_adapter),
) -> IngestionPipeline:
    return IngestionPipeline(
        metadata_store=store,
        embedding_adapter=embedding_adapter,
        vector_adapter=vector_adapter,
    )


def get_rag_engine(
    store: MetadataStore = Depends(get_metadata_store),
    llm_adapter: BaseLLMAdapter = Depends(get_llm_adapter),
    embedding_adapter: BaseEmbeddingAdapter = Depends(get_embedding_adapter),
    vector_adapter: BaseVectorStoreAdapter = Depends(get_vector_adapter),
) -> RAGEngine:
    return RAGEngine(
        metadata_store=store,
        llm_adapter=llm_adapter,
        embedding_adapter=embedding_adapter,
        vector_adapter=vector_adapter,
    )


def get_config_manager(
    store: MetadataStore = Depends(get_metadata_store),
    vector_adapter: BaseVectorStoreAdapter = Depends(get_vector_adapter),
) -> DynamicConfigManager:
    return DynamicConfigManager(
        metadata_store=store,
        vector_adapter=vector_adapter,
    )
