from fastapi import APIRouter, Depends

from app.api.dependencies import get_metadata_store, get_rag_engine
from app.models.api_schemas import QueryRequest, QueryResponse
from app.models.pipeline_config import PipelineConfig
from app.pipeline.rag_engine import RAGEngine
from app.storage.metadata_store import MetadataStore

router = APIRouter(prefix="/query", tags=["RAG Query"])


@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(
    req: QueryRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    store: MetadataStore = Depends(get_metadata_store),
):
    active_cfg = store.get_active_config()

    # Apply temporary query-level config overrides if provided
    exec_config = active_cfg
    if req.config_overrides:
        cfg_dict = active_cfg.model_dump()
        for k, v in req.config_overrides.items():
            if k in cfg_dict:
                if isinstance(v, dict) and isinstance(cfg_dict[k], dict):
                    cfg_dict[k].update(v)
                else:
                    cfg_dict[k] = v
        exec_config = PipelineConfig.model_validate(cfg_dict)

    res = await rag_engine.execute_query(
        kb_id=req.knowledge_base_id,
        question=req.question,
        filters=req.filters,
        config=exec_config,
    )

    return QueryResponse(
        question=res.question,
        answer=res.answer,
        is_grounded=res.is_grounded,
        is_found=res.is_found,
        citations=res.citations,
        trace=res.trace,
        config_version=res.config_version,
    )
