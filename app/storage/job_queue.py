import asyncio
from typing import Any, Callable, Coroutine
from app.core.logging import get_logger
from app.models.domain import Job, JobStatus
from app.storage.metadata_store import MetadataStore

logger = get_logger("job_queue")


class BackgroundJobQueue:
    """
    Asynchronous background job runner updating job states and progress in MetadataStore.
    """

    def __init__(self, metadata_store: MetadataStore):
        self.store = metadata_store
        self._running_tasks: dict[str, asyncio.Task] = {}

    def submit_job(
        self,
        job: Job,
        coro_fn: Callable[[Job], Coroutine[Any, Any, None]],
    ) -> Job:
        self.store.create_job(job)

        async def _wrapper():
            logger.info(f"Starting background job {job.id} (type: {job.job_type})")
            self.store.update_job(job.id, status=JobStatus.PROCESSING, progress_percent=10)
            try:
                await coro_fn(job)
                self.store.update_job(job.id, status=JobStatus.COMPLETED, progress_percent=100)
                logger.info(f"Job {job.id} completed successfully")
            except Exception as e:
                logger.error(f"Job {job.id} failed: {str(e)}", exc_info=True)
                self.store.update_job(
                    job.id,
                    status=JobStatus.FAILED,
                    error_message=str(e),
                )
            finally:
                self._running_tasks.pop(job.id, None)

        task = asyncio.create_task(_wrapper())
        self._running_tasks[job.id] = task
        return job
