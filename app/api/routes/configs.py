from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_config_manager
from app.models.api_schemas import (
    ActivateConfigResponse,
    ConfigListResponse,
    ConfigResponse,
)
from app.models.pipeline_config import PipelineConfig
from app.pipeline.config_manager import DynamicConfigManager

router = APIRouter(prefix="/configs", tags=["Pipeline Configurations"])


@router.get("/", response_model=ConfigListResponse)
async def list_configs(
    config_mgr: DynamicConfigManager = Depends(get_config_manager),
):
    configs = config_mgr.list_configs()
    active = config_mgr.get_active_config()
    return ConfigListResponse(configs=configs, active_version=active.version)


@router.get("/{version}", response_model=ConfigResponse)
async def get_config(
    version: str,
    config_mgr: DynamicConfigManager = Depends(get_config_manager),
):
    cfg = config_mgr.get_config(version)
    if not cfg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline configuration version '{version}' not found",
        )
    return ConfigResponse(config=cfg)


@router.post("/", response_model=ConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_config(
    config: PipelineConfig,
    config_mgr: DynamicConfigManager = Depends(get_config_manager),
):
    saved = config_mgr.save_new_config(config)
    return ConfigResponse(config=saved)


@router.put("/{version}/activate", response_model=ActivateConfigResponse)
async def activate_config(
    version: str,
    config_mgr: DynamicConfigManager = Depends(get_config_manager),
):
    activated = config_mgr.activate_config(version)
    return ActivateConfigResponse(
        version=activated.version,
        is_active=True,
        message=f"Pipeline configuration '{version}' is now active dynamically without restart.",
    )
