import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.adapters.embeddings.mock_adapter import MockEmbeddingAdapter
from app.adapters.llm.mock_adapter import MockLLMAdapter
from app.adapters.vector_store.in_memory_adapter import InMemoryVectorStoreAdapter
from app.api.dependencies import (
    get_embedding_adapter,
    get_llm_adapter,
    get_metadata_store,
    get_vector_adapter,
)
from app.ingestion.pipeline import IngestionPipeline
from app.main import app
from app.pipeline.config_manager import DynamicConfigManager
from app.pipeline.rag_engine import RAGEngine
from app.storage.job_queue import BackgroundJobQueue
from app.storage.metadata_store import MetadataStore


@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


@pytest.fixture
def metadata_store(temp_db_path):
    return MetadataStore(db_path=temp_db_path)


@pytest.fixture
def mock_embedding():
    return MockEmbeddingAdapter(model_name="mock-minilm", dimension=384)


@pytest.fixture
def mock_llm():
    return MockLLMAdapter(model_name="mock-gpt4o")


@pytest.fixture
def in_memory_vector_store():
    return InMemoryVectorStoreAdapter()


@pytest.fixture
def client(metadata_store, mock_embedding, mock_llm, in_memory_vector_store):
    # Override FastAPI dependencies for testing
    app.dependency_overrides[get_metadata_store] = lambda: metadata_store
    app.dependency_overrides[get_embedding_adapter] = lambda: mock_embedding
    app.dependency_overrides[get_llm_adapter] = lambda: mock_llm
    app.dependency_overrides[get_vector_adapter] = lambda: in_memory_vector_store
    app.dependency_overrides[DynamicConfigManager] = lambda: DynamicConfigManager(
        metadata_store, in_memory_vector_store
    )
    app.dependency_overrides[IngestionPipeline] = lambda: IngestionPipeline(
        metadata_store, mock_embedding, in_memory_vector_store
    )
    app.dependency_overrides[RAGEngine] = lambda: RAGEngine(
        metadata_store, mock_llm, mock_embedding, in_memory_vector_store
    )
    app.dependency_overrides[BackgroundJobQueue] = lambda: BackgroundJobQueue(metadata_store)

    from app.api.dependencies import (
        get_config_manager,
        get_ingestion_pipeline,
        get_job_queue,
        get_rag_engine,
    )

    app.dependency_overrides[get_config_manager] = lambda: DynamicConfigManager(
        metadata_store, in_memory_vector_store
    )
    app.dependency_overrides[get_ingestion_pipeline] = lambda: IngestionPipeline(
        metadata_store, mock_embedding, in_memory_vector_store
    )
    app.dependency_overrides[get_rag_engine] = lambda: RAGEngine(
        metadata_store, mock_llm, mock_embedding, in_memory_vector_store
    )
    app.dependency_overrides[get_job_queue] = lambda: BackgroundJobQueue(metadata_store)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
