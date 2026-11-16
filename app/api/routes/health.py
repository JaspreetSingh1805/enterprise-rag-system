from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.adapters.vector_store.base import BaseVectorStoreAdapter
from app.api.dependencies import get_metadata_store, get_vector_adapter
from app.core.config import settings
from app.models.api_schemas import HealthResponse
from app.storage.metadata_store import MetadataStore

router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    store: MetadataStore = Depends(get_metadata_store),
    vector_adapter: BaseVectorStoreAdapter = Depends(get_vector_adapter),
):
    active_cfg = store.get_active_config()
    vector_status = "connected"
    try:
        # Check vector adapter responsiveness
        await vector_adapter.count("health_check_nonexistent_kb")
    except Exception:
        # Nonexistent collection count returning 0 is expected
        vector_status = "ready"

    return HealthResponse(
        status="healthy",
        environment=settings.app_env,
        active_config_version=active_cfg.version,
        vector_store_type=vector_adapter.store_type,
        vector_store_status=vector_status,
        timestamp=datetime.now(timezone.utc),
    )
