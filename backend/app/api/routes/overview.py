from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas.overview import ModelPerformance, OverviewResponse
from backend.app.services import overview_service

router = APIRouter(tags=["overview"])


@router.get("/overview", response_model=OverviewResponse)
def get_overview() -> OverviewResponse:
    return overview_service.get_overview()


@router.get("/models/{model_name}/performance", response_model=ModelPerformance)
def get_model_performance(model_name: str) -> ModelPerformance:
    return overview_service.get_model_performance(model_name)
