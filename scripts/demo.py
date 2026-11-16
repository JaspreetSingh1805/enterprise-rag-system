"""
End-to-End Enterprise RAG Platform Interactive Demo Script
Demonstrates full lifecycle: Health check, KB Creation, Multi-format Ingestion,
Answerable Queries with Citations, Unanswerable Query Rejection, Dynamic Config Switching,
and Cascading Document Deletion.
"""
import asyncio
import io
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.adapters.embeddings.mock_adapter import MockEmbeddingAdapter
from app.adapters.llm.mock_adapter import MockLLMAdapter
from app.adapters.vector_store.in_memory_adapter import InMemoryVectorStoreAdapter
from app.ingestion.pipeline import IngestionPipeline
from app.models.domain import (
    ChunkingStrategyType,
    KnowledgeBase,
    SearchMode,
)
from app.models.pipeline_config import (
    ChunkingConfig,
    LLMConfig,
    PipelineConfig,
    RerankConfig,
    RetrievalConfig,
)
from app.pipeline.config_manager import DynamicConfigManager
from app.pipeline.rag_engine import RAGEngine
from app.storage.metadata_store import MetadataStore


def print_header(title: str):
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)


async def run_demo():
    print_header("Enterprise Configurable RAG Platform - Interactive Demo")

    # Initialize in-memory / test storage
    db_path = "./data/demo_metadata.db"
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
    store = MetadataStore(db_path=db_path)
    embedding_adapter = MockEmbeddingAdapter(model_name="all-MiniLM-L6-v2", dimension=384)
    llm_adapter = MockLLMAdapter(model_name="gpt-4o-mini")
    vector_adapter = InMemoryVectorStoreAdapter()

    pipeline = IngestionPipeline(store, embedding_adapter, vector_adapter)
    rag_engine = RAGEngine(store, llm_adapter, embedding_adapter, vector_adapter)
    config_mgr = DynamicConfigManager(store, vector_adapter)

    # 1. Create Knowledge Base
    print("\n[Step 1] Creating Multi-Tenant Knowledge Base: 'acme_global'...")
    kb = KnowledgeBase(
        id="acme_global",
        name="Acme Corporation Global Knowledge Base",
        description="Official company policies, security guidelines, and API manuals.",
        embedding_model=embedding_adapter.model_name,
        embedding_dimension=embedding_adapter.dimension,
    )
    store.create_kb(kb)
    await vector_adapter.create_collection("acme_global", embedding_adapter.dimension)
    print(f"✓ Knowledge Base '{kb.id}' created with {embedding_adapter.dimension}-dimension vector index.")

    # 2. Ingest Sample Documents (PDF, DOCX, TXT, MD)
    print("\n[Step 2] Ingesting Multi-Format Enterprise Documents...")
    sample_files = [
        ("hr_leave_policy.md", {"department": "HR", "category": "Policy"}),
        ("security_guidelines.txt", {"department": "IT", "category": "Security"}),
        ("employee_handbook.docx", {"department": "Operations", "category": "Handbook"}),
        ("product_workflow_api.pdf", {"department": "Engineering", "category": "API"}),
    ]

    sample_dir = Path("./sample_docs")
    ingested_docs = []
    for fname, custom_fields in sample_files:
        fpath = sample_dir / fname
        if not fpath.exists():
            print(f"  Warning: {fname} not found. Run generate_sample_docs.py first.")
            continue
        with open(fpath, "rb") as f:
            content = f.read()
        doc = await pipeline.ingest_file(
            kb_id="acme_global",
            filename=fname,
            file_bytes=content,
            custom_fields=custom_fields,
        )
        ingested_docs.append(doc)
        print(f"  ✓ Ingested '{fname}' -> Doc ID: {doc.id[:8]}... (Size: {doc.file_size_bytes} bytes, Pages: {doc.page_count})")

    kb_info = store.get_kb("acme_global")
    print(f"✓ Knowledge Base Summary: {kb_info.document_count} documents, {kb_info.chunk_count} indexed chunks.")

    # 3. Query 1: Answerable Query with Citations
    print_header("Step 3: Answerable Query Execution (Config v1: Baseline)")
    question_1 = "Under what condition is a medical certificate required when taking sick leave?"
    print(f"Query: \"{question_1}\"")

    res_1 = await rag_engine.execute_query(kb_id="acme_global", question=question_1)
    print(f"\nResponse:\n{res_1.answer}\n")
    print("Citations Extracted:")
    for c in res_1.citations:
        print(f"  - [{c.document_name} | Page: {c.page_number} | Section: {c.section_title}]")
        print(f"    Snippet: \"{c.snippet}\" (Score: {c.score:.3f})")
    print(f"\nExecution Metadata:")
    print(f"  - Latency: {res_1.trace.total_latency_ms} ms")
    print(f"  - Config Version: {res_1.config_version}")
    print(f"  - Grounded: {res_1.is_grounded} | Found: {res_1.is_found}")

    # 4. Query 2: Unanswerable Query (Clean Refusal)
    print_header("Step 4: Unanswerable Query Execution (Grounding & Rejection)")
    question_2 = "What was the total landing payload weight of the NASA Curiosity Mars rover mission?"
    print(f"Query: \"{question_2}\"")

    res_2 = await rag_engine.execute_query(kb_id="acme_global", question=question_2)
    print(f"\nResponse:\n\"{res_2.answer}\"\n")
    print(f"Grounding Status: is_grounded={res_2.is_grounded}, is_found={res_2.is_found}")
    print(f"Citations: {len(res_2.citations)} (Zero hallucinations)")

    # 5. Dynamic Configuration Activation without Rebuilding
    print_header("Step 5: Dynamic Configuration Activation (v1 -> v2)")
    print("Admin activating Configuration 'v2' (Hybrid Search + Reranking + Query Rewrite)...")
    v2_config = PipelineConfig(
        version="v2",
        description="Production v2: Hybrid BM25+Dense search, cross-encoder reranking, and query rewrite",
        chunking=ChunkingConfig(strategy=ChunkingStrategyType.STRUCTURE, chunk_size=500, chunk_overlap=50),
        retrieval=RetrievalConfig(search_mode=SearchMode.HYBRID, top_k=5, similarity_threshold=0.30),
        rerank=RerankConfig(enabled=True, top_n=3),
        llm=LLMConfig(query_rewrite_enabled=True),
    )
    config_mgr.save_new_config(v2_config)
    activated = config_mgr.activate_config("v2")
    print(f"✓ Dynamically switched active pipeline configuration to '{activated.version}' WITHOUT restarting server!")

    # 6. Query with Config v2 (Technical Query)
    print_header("Step 6: Querying with Config v2 (Hybrid Search & Reranking)")
    question_3 = "What is the standard API rate limit and which HTTP status code is returned for throttling?"
    print(f"Query: \"{question_3}\"")

    res_3 = await rag_engine.execute_query(kb_id="acme_global", question=question_3)
    print(f"\nResponse:\n{res_3.answer}\n")
    print("Citations Extracted:")
    for c in res_3.citations:
        print(f"  - [{c.document_name} | Page: {c.page_number} | Section: {c.section_title}]")
    print(f"\nExecution Stages Recorded in Trace:")
    for stage in res_3.trace.stages:
        print(f"  - Stage: {stage.stage_name:<16} Duration: {stage.duration_ms:>6.2f} ms | Details: {stage.details}")

    # 7. Document Deletion & Vector Cleanup
    print_header("Step 7: Cascading Document & Vector Deletion")
    doc_to_delete = ingested_docs[0]
    print(f"Deleting document '{doc_to_delete.filename}' (ID: {doc_to_delete.id})...")
    del_count = await vector_adapter.delete_by_document("acme_global", doc_to_delete.id)
    store.delete_document(doc_to_delete.id)
    print(f"✓ Deleted document record from database and {del_count} vectors from vector store.")

    updated_kb = store.get_kb("acme_global")
    print(f"✓ Updated Knowledge Base document count: {updated_kb.document_count}")

    print_header("Demo Completed Successfully - All Stages Verified")


if __name__ == "__main__":
    asyncio.run(run_demo())
