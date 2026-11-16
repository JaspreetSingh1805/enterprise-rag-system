import pytest

from app.adapters.embeddings.local_adapter import LocalEmbeddingAdapter
from app.adapters.embeddings.mock_adapter import MockEmbeddingAdapter
from app.adapters.llm.mock_adapter import MockLLMAdapter
from app.adapters.vector_store.qdrant_adapter import QdrantVectorStoreAdapter
from app.ingestion.pipeline import IngestionPipeline
from app.models.domain import Chunk, KnowledgeBase
from app.pipeline.config_manager import DynamicConfigManager


@pytest.mark.asyncio
async def test_mock_llm_adapter():
    adapter = MockLLMAdapter(model_name="mock-model")
    assert adapter.provider_name == "mock"

    res = await adapter.generate(prompt="Question: What is 2 + 2?")
    assert res.model_name == "mock-model"
    assert res.total_tokens > 0


@pytest.mark.asyncio
async def test_embedding_adapters():
    mock_emb = MockEmbeddingAdapter(dimension=384)
    assert mock_emb.dimension == 384
    vec = await mock_emb.embed_query("testing query embedding")
    assert len(vec) == 384

    local_emb = LocalEmbeddingAdapter(model_name="all-MiniLM-L6-v2", dimension=384)
    assert local_emb.dimension == 384
    vec_local = await local_emb.embed_query("testing local vectorizer")
    assert len(vec_local) == 384


@pytest.mark.asyncio
async def test_qdrant_in_memory_adapter():
    # Test Qdrant in-memory mode
    qdrant = QdrantVectorStoreAdapter(in_memory=True)
    assert qdrant.store_type == "qdrant"

    # Create collection
    await qdrant.create_collection("kb_qdrant_test", dimension=384)

    # Upsert chunk
    chunk = Chunk(
        id="c_q1",
        document_id="doc_q1",
        knowledge_base_id="kb_qdrant_test",
        text="Qdrant in-memory test chunk content.",
        chunk_index=0,
        embedding=[0.1] * 384,
    )
    await qdrant.upsert_chunks("kb_qdrant_test", [chunk])

    # Search
    results = await qdrant.search("kb_qdrant_test", query_vector=[0.1] * 384, top_k=1)
    assert len(results) == 1
    assert results[0][0].id == "c_q1"

    # Delete
    deleted = await qdrant.delete_by_document("kb_qdrant_test", "doc_q1")
    assert deleted == 1


@pytest.mark.asyncio
async def test_safe_reindexing_when_embedding_model_changes(metadata_store, in_memory_vector_store):
    # Initial setup with 384-dimension embedding
    emb_v1 = MockEmbeddingAdapter(model_name="mock-384", dimension=384)
    kb = KnowledgeBase(
        id="kb_reindex_test",
        name="Reindex KB",
        embedding_model="mock-384",
        embedding_dimension=384,
    )
    metadata_store.create_kb(kb)
    pipeline = IngestionPipeline(metadata_store, emb_v1, in_memory_vector_store)

    await pipeline.ingest_file(
        kb_id="kb_reindex_test",
        filename="test_file.txt",
        file_bytes=b"Content that will be reindexed with a new embedding model.",
    )

    chunks_before = metadata_store.get_all_chunks_for_kb("kb_reindex_test")
    assert len(chunks_before) > 0

    # Model changes to 768 dimensions
    emb_v2 = MockEmbeddingAdapter(model_name="mock-768", dimension=768)
    config_mgr = DynamicConfigManager(metadata_store, in_memory_vector_store)

    reindexed_count = await config_mgr.safe_reindex_knowledge_base("kb_reindex_test", emb_v2)
    assert reindexed_count == len(chunks_before)

    # Verify KB updated in metadata
    updated_kb = metadata_store.get_kb("kb_reindex_test")
    assert updated_kb.embedding_model == "mock-768"
    assert updated_kb.embedding_dimension == 768

    # Verify vector store collection dimension updated
    new_dim = await in_memory_vector_store.get_collection_dimension("kb_reindex_test")
    assert new_dim == 768
