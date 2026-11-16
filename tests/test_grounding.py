import pytest

from app.models.domain import Chunk, KnowledgeBase
from app.pipeline.grounding import NOT_FOUND_MESSAGE, GroundingEngine
from app.pipeline.rag_engine import RAGEngine


def test_grounding_engine_extract_citations():
    engine = GroundingEngine(similarity_threshold=0.35)
    sample_chunk = Chunk(
        id="c1",
        document_id="doc_hr",
        knowledge_base_id="kb1",
        text="Permanent employees get 25 annual leave days.",
        chunk_index=0,
        page_number=2,
        section_title="Leave Entitlement",
        metadata={"filename": "hr_policy.md"},
    )
    answer = "Employees receive 25 days of paid leave [Doc: hr_policy.md, Page: 2, Section: Leave Entitlement]."
    citations = engine.extract_citations(answer, [(sample_chunk, 0.95)])

    assert len(citations) == 1
    assert citations[0].document_name == "hr_policy.md"
    assert citations[0].page_number == 2
    assert citations[0].section_title == "Leave Entitlement"
    assert "25 annual leave days" in citations[0].snippet


def test_grounding_engine_not_found_detection():
    engine = GroundingEngine(similarity_threshold=0.35)
    sample_chunk = Chunk(
        id="c1",
        document_id="doc1",
        knowledge_base_id="kb1",
        text="Random unrelated text.",
        chunk_index=0,
    )

    # Answer says cannot find
    ans_not_found = "I could not find information about this in the knowledge base."
    assert engine.check_is_found(ans_not_found, [(sample_chunk, 0.8)]) is False

    # Low score below threshold
    assert engine.check_is_found("Some answer", [(sample_chunk, 0.2)]) is False


@pytest.mark.asyncio
async def test_rag_engine_unanswerable_query(
    metadata_store, mock_embedding, mock_llm, in_memory_vector_store
):
    kb = KnowledgeBase(
        id="kb_grounding",
        name="Grounding KB",
        embedding_model="mock",
        embedding_dimension=384,
    )
    metadata_store.create_kb(kb)
    await in_memory_vector_store.create_collection(kb.id, 384)

    rag = RAGEngine(metadata_store, mock_llm, mock_embedding, in_memory_vector_store)

    # Empty KB query must return NOT_FOUND_MESSAGE
    res = await rag.execute_query(
        kb_id="kb_grounding",
        question="What is the CEO personal cell phone number?",
    )
    assert res.is_found is False
    assert res.answer == NOT_FOUND_MESSAGE
    assert len(res.citations) == 0
