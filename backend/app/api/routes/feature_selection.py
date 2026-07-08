from __future__ import annotations

from fastapi import APIRouter

from backend.app.jobs.manager import job_manager
from backend.app.schemas.feature_selection import FeatureSelectionRequest, JobIdResponse
from backend.app.services.feature_selection_service import run_feature_selection_job

router = APIRouter(prefix="/feature-selection", tags=["feature-selection"])


@router.post("/jobs", response_model=JobIdResponse, status_code=202)
def submit_feature_selection(body: FeatureSelectionRequest) -> JobIdResponse:
    job_id = job_manager.submit(
        run_feature_selection_job,
        dataset_name=body.dataset_name,
        y_cols=body.y_cols,
        x_cols=body.x_cols,
        top_k=body.top_k,
        enabled_methods=body.enabled_methods,
        corr_threshold=body.corr_threshold,
        vif_threshold=body.vif_threshold,
        per_target=body.per_target,
        progress_mode="message",
    )
    return JobIdResponse(job_id=job_id)
