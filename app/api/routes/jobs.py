from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_metadata_store
from app.core.exceptions import JobNotFoundError
from app.models.api_schemas import JobListResponse, JobResponse
from app.storage.metadata_store import MetadataStore

router = APIRouter(prefix="/jobs", tags=["Background Jobs"])


@router.get("/", response_model=JobListResponse)
async def list_jobs(
    knowledge_base_id: Optional[str] = Query(default=None),
    store: MetadataStore = Depends(get_metadata_store),
):
    jobs = store.list_jobs(kb_id=knowledge_base_id)
    return JobListResponse(jobs=jobs, total=len(jobs))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    store: MetadataStore = Depends(get_metadata_store),
):
    job = store.get_job(job_id)
    if not job:
        raise JobNotFoundError(job_id)
    return JobResponse(job=job)
