from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    name: str
    uploaded_at: str
    rows: int
    cols: int


class DatasetListResponse(BaseModel):
    datasets: List[DatasetSummary]


class DatasetPreview(BaseModel):
    name: str
    shape: List[int]
    columns: List[str]
    head: List[dict[str, Any]]
