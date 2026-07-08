from __future__ import annotations

from fastapi import APIRouter, UploadFile

from backend.app.schemas.datasets import DatasetListResponse, DatasetPreview, DatasetSummary
from backend.app.services import dataset_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetSummary, status_code=201)
async def upload_dataset(file: UploadFile) -> DatasetSummary:
    return await dataset_service.upload_dataset(file)


@router.get("", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    return DatasetListResponse(datasets=dataset_service.list_datasets())


@router.get("/{name}/preview", response_model=DatasetPreview)
def get_dataset_preview(name: str) -> DatasetPreview:
    return dataset_service.get_dataset_preview(name)


@router.delete("/{name}", status_code=204)
def delete_dataset(name: str) -> None:
    dataset_service.delete_dataset(name)
