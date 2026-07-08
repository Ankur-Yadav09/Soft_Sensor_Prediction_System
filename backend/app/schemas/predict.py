from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    model_name: str
    source: str  # "project_test" | "dataset"
    project_id: Optional[str] = None
    dataset_name: Optional[str] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None


class PredictResponse(BaseModel):
    x_cols: List[str]
    y_cols: List[str]
    rows: List[Dict[str, Any]]
    has_actuals: bool
    metrics: Optional[List[dict]] = None
