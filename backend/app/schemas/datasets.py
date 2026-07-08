from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    name: str
    uploaded_at: str
    rows: int
    cols: int
    plant: Optional[str] = None
    unit: Optional[str] = None
    status: str = "Ready"


class DatasetListResponse(BaseModel):
    datasets: List[DatasetSummary]


class DatasetPreview(BaseModel):
    name: str
    shape: List[int]
    columns: List[str]
    head: List[dict[str, Any]]
