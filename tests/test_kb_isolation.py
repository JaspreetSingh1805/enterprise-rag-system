import pytest

from app.ingestion.pipeline import IngestionPipeline
from app.models.domain import KnowledgeBase
from app.pipeline.rag_engine import RAGEngine


@pytest.mark.asyncio
async def test_knowledge_base_isolation(
    metadata_store, mock_embedding, mock_llm, in_memory_vector_store
):
    # 1. Create Tenant A and Tenant B KBs
    kb_a = KnowledgeBase(
        id="tenant_a", name="Tenant A", embedding_model="mock", embedding_dimension=384
    )
    kb_b = KnowledgeBase(
        id="tenant_b", name="Tenant B", embedding_model="mock", embedding_dimension=384
    )
    metadata_store.create_kb(kb_a)
    metadata_store.create_kb(kb_b)

    pipeline = IngestionPipeline(metadata_store, mock_embedding, in_memory_vector_store)
    rag_engine = RAGEngine(metadata_store, mock_llm, mock_embedding, in_memory_vector_store)

    # Ingest document for Tenant A
    await pipeline.ingest_file(
        kb_id="tenant_a",
        filename="secret_a.txt",
        file_bytes=b"Confidential project Alpha details for Tenant A only.",
    )

    # Ingest document for Tenant B
    await pipeline.ingest_file(
        kb_id="tenant_b",
        filename="secret_b.txt",
        file_bytes=b"Confidential project Beta details for Tenant B only.",
    )

    # 2. Query Tenant A for Project Alpha -> should find
    res_a = await rag_engine.execute_query(
        kb_id="tenant_a",
        question="What are the details for project Alpha?",
    )
    assert res_a.is_found is True
    assert "secret_a.txt" in [c.document_name for c in res_a.citations]

    # 3. Query Tenant B for Project Alpha -> should NOT find Project Alpha!
    res_b = await rag_engine.execute_query(
        kb_id="tenant_b",
        question="What are the details for project Alpha?",
    )
    # Must not leak Tenant A's document into Tenant B's results
    for c in res_b.citations:
        assert c.document_name != "secret_a.txt"


@pytest.mark.asyncio
async def test_document_and_vector_cascading_deletion(
    metadata_store, mock_embedding, in_memory_vector_store
):
    kb = KnowledgeBase(
        id="kb_del", name="Del Test", embedding_model="mock", embedding_dimension=384
    )
    metadata_store.create_kb(kb)
    pipeline = IngestionPipeline(metadata_store, mock_embedding, in_memory_vector_store)

    doc = await pipeline.ingest_file(
        kb_id="kb_del",
        filename="temp_policy.txt",
        file_bytes=b"Temporary policy that will be deleted soon.",
    )

    # Verify vectors exist
    initial_count = await in_memory_vector_store.count("kb_del")
    assert initial_count > 0

    # Delete vectors for this doc
    deleted_vectors = await in_memory_vector_store.delete_by_document("kb_del", doc.id)
    assert deleted_vectors > 0

    # Delete document record in SQLite
    metadata_store.delete_document(doc.id)

    # Check vector count is now 0
    final_count = await in_memory_vector_store.count("kb_del")
    assert final_count == 0
    assert metadata_store.get_document(doc.id) is None
