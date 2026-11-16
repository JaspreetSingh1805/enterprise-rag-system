import pytest

from app.models.domain import Chunk
from app.retrieval.bm25 import BM25SearchEngine
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            id="c1",
            document_id="doc1",
            knowledge_base_id="kb1",
            text="Employees receive 25 days of annual paid leave per calendar year.",
            chunk_index=0,
            page_number=1,
            section_title="Annual Leave",
            metadata={"department": "HR", "type": "policy"},
            embedding=[0.8, 0.2, 0.1, 0.0],
        ),
        Chunk(
            id="c2",
            document_id="doc1",
            knowledge_base_id="kb1",
            text="Sick leave of more than 2 consecutive business days requires a doctor certificate.",
            chunk_index=1,
            page_number=1,
            section_title="Sick Leave",
            metadata={"department": "HR", "type": "policy"},
            embedding=[0.1, 0.9, 0.1, 0.0],
        ),
        Chunk(
            id="c3",
            document_id="doc2",
            knowledge_base_id="kb1",
            text="Corporate laptops must be encrypted with BitLocker or FileVault at all times.",
            chunk_index=0,
            page_number=1,
            section_title="Encryption",
            metadata={"department": "IT", "type": "security"},
            embedding=[0.0, 0.1, 0.9, 0.2],
        ),
    ]


def test_bm25_keyword_search(sample_chunks):
    engine = BM25SearchEngine(sample_chunks)

    # Search for annual leave
    results = engine.search("annual paid leave", top_k=2)
    assert len(results) > 0
    assert results[0][0].id == "c1"
    assert results[0][1] > 0.0

    # Search with filter for IT department
    results_filtered = engine.search("leave", top_k=5, filters={"department": "IT"})
    assert len(results_filtered) == 0


def test_hybrid_reciprocal_rank_fusion(sample_chunks):
    vector_results = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.4)]
    bm25_results = [(sample_chunks[1], 0.8), (sample_chunks[0], 0.3)]

    hybrid = reciprocal_rank_fusion(vector_results, bm25_results, top_k=2, rrf_k=60)
    assert len(hybrid) == 2
    # Both candidates present and scored
    ids = [c.id for c, _ in hybrid]
    assert "c1" in ids and "c2" in ids
    assert hybrid[0][1] >= hybrid[1][1]


@pytest.mark.asyncio
async def test_reranker_reordering(sample_chunks):
    reranker = Reranker()
    candidates = [(sample_chunks[0], 0.5), (sample_chunks[2], 0.6)]

    # Query specifically matching BitLocker encryption
    reranked = await reranker.rerank("BitLocker encryption corporate laptop", candidates, top_n=2)
    assert len(reranked) == 2
    # Chunk c3 (Security/BitLocker) should be boosted to the top
    assert reranked[0][0].id == "c3"
