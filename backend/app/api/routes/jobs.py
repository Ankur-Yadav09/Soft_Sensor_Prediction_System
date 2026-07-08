from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.jobs.manager import job_manager
from backend.app.jobs.models import JobStatus
from backend.app.schemas.jobs import JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    record = job_manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(
        id=record.id,
        status=record.status.value,
        progress=record.progress,
        error=record.error,
        done=record.status in (JobStatus.DONE, JobStatus.ERROR),
        result=record.result if record.status == JobStatus.DONE else None,
    )
