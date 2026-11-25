import time
from typing import Any, Optional

from app.adapters.embeddings.base import BaseEmbeddingAdapter
from app.adapters.factory import AdapterRegistry
from app.adapters.llm.base import BaseLLMAdapter
from app.adapters.vector_store.base import BaseVectorStoreAdapter
from app.core.exceptions import EmbeddingModelMismatchError, KnowledgeBaseNotFoundError
from app.core.logging import get_current_trace_id, get_logger
from app.models.domain import (
    Chunk,
    ExecutionStageTrace,
    ExecutionTrace,
    QueryResult,
    SearchMode,
)
from app.models.pipeline_config import PipelineConfig
from app.pipeline.grounding import NOT_FOUND_MESSAGE, GroundingEngine
from app.retrieval.bm25 import BM25SearchEngine
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import Reranker
from app.storage.metadata_store import MetadataStore

logger = get_logger("rag_engine")


class RAGEngine:
    """
    End-to-end RAG Query Execution Pipeline.
    """

    def __init__(
        self,
        metadata_store: MetadataStore,
        llm_adapter: Optional[BaseLLMAdapter] = None,
        embedding_adapter: Optional[BaseEmbeddingAdapter] = None,
        vector_adapter: Optional[BaseVectorStoreAdapter] = None,
        reranker: Optional[Reranker] = None,
        adapter_registry: Optional[AdapterRegistry] = None,
    ):
        self.store = metadata_store
        self._default_llm_adapter = llm_adapter
        self._default_embedding_adapter = embedding_adapter
        self.vector_adapter = vector_adapter
        self.reranker = reranker or Reranker()
        self.adapter_registry = adapter_registry or AdapterRegistry()

    async def execute_query(
        self,
        kb_id: str,
        question: str,
        filters: Optional[dict[str, Any]] = None,
        config: Optional[PipelineConfig] = None,
    ) -> QueryResult:
        start_time = time.perf_counter()
        trace_id = get_current_trace_id()
        stages: list[ExecutionStageTrace] = []

        # 1. Resolve Active Config
        if not config:
            config = self.store.get_active_config()

        logger.info(f"Executing query for KB '{kb_id}' using config {config.version}")

        # 2. Validate KB Existence
        kb = self.store.get_kb(kb_id)
        if not kb:
            raise KnowledgeBaseNotFoundError(kb_id)

        # 3. Dynamically resolve LLM and Embedding adapters based on runtime config
        llm_adapter = (
            self._default_llm_adapter
            if self._default_llm_adapter is not None
            else self.adapter_registry.get_llm_adapter(config.llm)
        )
        embedding_adapter = (
            self._default_embedding_adapter
            if self._default_embedding_adapter is not None
            else self.adapter_registry.get_embedding_adapter_for_kb(kb)
        )

        # Check embedding compatibility
        if kb.embedding_dimension != embedding_adapter.dimension:
            raise EmbeddingModelMismatchError(
                expected_model=kb.embedding_model,
                expected_dim=kb.embedding_dimension,
                actual_model=embedding_adapter.model_name,
                actual_dim=embedding_adapter.dimension,
            )

        # 4. Stage: Query Rewrite (Optional)
        search_query = question
        rewrite_duration = 0.0
        if config.llm.query_rewrite_enabled:
            rw_start = time.perf_counter()
            query_rewriter = QueryRewriter(llm_adapter)
            search_query = await query_rewriter.rewrite(question, config.llm.query_rewrite_prompt)
            rewrite_duration = (time.perf_counter() - rw_start) * 1000
            stages.append(
                ExecutionStageTrace(
                    stage_name="query_rewrite",
                    duration_ms=round(rewrite_duration, 2),
                    details={
                        "original_query": question,
                        "rewritten_query": search_query,
                        "llm_provider": llm_adapter.provider_name,
                        "llm_model": llm_adapter.model_name,
                    },
                )
            )

        # 5. Stage: Retrieval (Vector / Keyword / Hybrid)
        retrieval_start = time.perf_counter()
        candidates: list[tuple[Chunk, float]] = []
        mode = config.retrieval.search_mode

        if mode in (SearchMode.VECTOR, SearchMode.HYBRID):
            query_vector = await embedding_adapter.embed_query(search_query)
            vector_results = await self.vector_adapter.search(
                kb_id=kb_id,
                query_vector=query_vector,
                top_k=config.retrieval.top_k,
                filters=filters,
            )
        else:
            vector_results = []

        if mode in (SearchMode.KEYWORD, SearchMode.HYBRID):
            all_chunks = self.store.get_all_chunks_for_kb(kb_id)
            bm25_engine = BM25SearchEngine(all_chunks)
            bm25_results = bm25_engine.search(
                query=search_query,
                top_k=config.retrieval.top_k,
                filters=filters,
            )
        else:
            bm25_results = []

        if mode == SearchMode.HYBRID:
            candidates = reciprocal_rank_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                top_k=config.retrieval.top_k,
                rrf_k=config.retrieval.rrf_k,
                dense_weight=config.retrieval.dense_weight,
                bm25_weight=config.retrieval.bm25_weight,
            )
        elif mode == SearchMode.VECTOR:
            candidates = vector_results
        else:
            candidates = bm25_results

        retrieval_duration = (time.perf_counter() - retrieval_start) * 1000
        stages.append(
            ExecutionStageTrace(
                stage_name="retrieval",
                duration_ms=round(retrieval_duration, 2),
                details={
                    "search_mode": mode.value,
                    "retrieved_count": len(candidates),
                    "top_score": candidates[0][1] if candidates else 0.0,
                },
            )
        )

        # 5. Stage: Optional Rerank
        rerank_duration = 0.0
        final_chunks = candidates
        if config.rerank.enabled and candidates:
            rr_start = time.perf_counter()
            final_chunks = await self.reranker.rerank(
                query=search_query,
                candidates=candidates,
                top_n=config.rerank.top_n,
            )
            rerank_duration = (time.perf_counter() - rr_start) * 1000
            stages.append(
                ExecutionStageTrace(
                    stage_name="rerank",
                    duration_ms=round(rerank_duration, 2),
                    details={"reranked_top_n": len(final_chunks)},
                )
            )

        # 6. Stage: Grounding, Citation & LLM Generation
        grounding_start = time.perf_counter()
        grounding_engine = GroundingEngine(
            similarity_threshold=config.retrieval.similarity_threshold
        )

        retrieval_top_score = candidates[0][1] if candidates else 0.0

        # Confidence gate: if below similarity threshold, reject immediately
        if not candidates or retrieval_top_score < config.retrieval.similarity_threshold:
            logger.info(
                f"Query '{question}' rejected by similarity threshold "
                f"({retrieval_top_score:.3f} < {config.retrieval.similarity_threshold})"
            )
            grounding_duration = (time.perf_counter() - grounding_start) * 1000
            stages.append(
                ExecutionStageTrace(
                    stage_name="generation",
                    duration_ms=round(grounding_duration, 2),
                    details={"rejected_by_threshold": True, "top_score": retrieval_top_score},
                )
            )

            total_latency = (time.perf_counter() - start_time) * 1000
            return QueryResult(
                question=question,
                answer=NOT_FOUND_MESSAGE,
                is_grounded=True,
                is_found=False,
                citations=[],
                trace=ExecutionTrace(
                    trace_id=trace_id,
                    total_latency_ms=round(total_latency, 2),
                    config_version=config.version,
                    stages=stages,
                    prompt_tokens=0,
                    completion_tokens=len(NOT_FOUND_MESSAGE.split()),
                    total_tokens=len(NOT_FOUND_MESSAGE.split()),
                ),
                config_version=config.version,
            )

        # Format context for LLM
        formatted_context = grounding_engine.format_context(final_chunks)
        llm_prompt = f"Context:\n{formatted_context}\n\nQuestion: {question}\n\nAnswer:"

        llm_res = await llm_adapter.generate(
            prompt=llm_prompt,
            system_prompt=config.llm.system_prompt,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )

        citations = grounding_engine.extract_citations(llm_res.text, final_chunks)
        is_found = grounding_engine.check_is_found(llm_res.text, final_chunks)

        grounding_duration = (time.perf_counter() - grounding_start) * 1000
        stages.append(
            ExecutionStageTrace(
                stage_name="generation",
                duration_ms=round(grounding_duration, 2),
                details={
                    "citations_count": len(citations),
                    "is_found": is_found,
                    "prompt_tokens": llm_res.prompt_tokens,
                    "completion_tokens": llm_res.completion_tokens,
                    "llm_provider": llm_adapter.provider_name,
                    "llm_model": llm_adapter.model_name,
                },
            )
        )

        total_latency = (time.perf_counter() - start_time) * 1000

        return QueryResult(
            question=question,
            answer=llm_res.text,
            is_grounded=True,
            is_found=is_found,
            citations=citations,
            trace=ExecutionTrace(
                trace_id=trace_id,
                total_latency_ms=round(total_latency, 2),
                config_version=config.version,
                stages=stages,
                prompt_tokens=llm_res.prompt_tokens,
                completion_tokens=llm_res.completion_tokens,
                total_tokens=llm_res.total_tokens,
            ),
            config_version=config.version,
        )
