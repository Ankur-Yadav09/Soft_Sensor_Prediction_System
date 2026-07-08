from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from backend.app.schemas.datasets import DatasetSummary


class SavedModelSummary(BaseModel):
    name: str
    saved_at: str
    input_dim: int
    output_dim: int
    algorithm: Optional[str] = None
    avg_r2: Optional[float] = None
    avg_rmse: Optional[float] = None
    avg_mae: Optional[float] = None


class OverviewResponse(BaseModel):
    datasets: List[DatasetSummary]
    saved_models: List[SavedModelSummary]


class TargetMetric(BaseModel):
    name: str
    r2: float
    mae: float


class ModelPerformance(BaseModel):
    model_name: str
    dataset_name: str
    per_target: List[TargetMetric]
    avg_r2: float
    grade: str
    emoji: str
    x_cols: List[str]
    y_cols: List[str]
