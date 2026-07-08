from __future__ import annotations

from typing import List

from fastapi import APIRouter

from backend.app.schemas.preprocess import (
    ApplyPreprocessingRequest,
    ApplyPreprocessingResponse,
    AutomatedCleaningRequest,
    BasicCleaningRequest,
    CleaningResponse,
    FeatureDetailResponse,
    FeatureStatsResponse,
    ProjectSummary,
)
from backend.app.services import preprocess_service, project_service

router = APIRouter(tags=["preprocess"])


@router.get("/preprocess/{dataset_name}/stats", response_model=FeatureStatsResponse)
def get_stats(dataset_name: str) -> FeatureStatsResponse:
    return FeatureStatsResponse(stats=preprocess_service.get_feature_stats(dataset_name))


@router.get("/preprocess/{dataset_name}/feature-detail", response_model=FeatureDetailResponse)
def get_feature_detail(dataset_name: str, column: str) -> FeatureDetailResponse:
    return FeatureDetailResponse(**preprocess_service.get_feature_detail(dataset_name, column))


@router.post("/preprocess/clean", response_model=CleaningResponse)
def apply_basic_cleaning(body: BasicCleaningRequest) -> CleaningResponse:
    domain_filters = (
        {k: v.model_dump() for k, v in body.domain_filters.items()} if body.domain_filters else None
    )
    result = preprocess_service.apply_basic_cleaning(
        dataset_name=body.dataset_name,
        new_dataset_name=body.new_dataset_name,
        remove_missing_rows=body.remove_missing_rows,
        remove_duplicates=body.remove_duplicates,
        remove_missing_cols=body.remove_missing_cols,
        missing_col_threshold=body.missing_col_threshold,
        remove_constant_cols=body.remove_constant_cols,
        remove_nzv_cols=body.remove_nzv_cols,
        nzv_threshold=body.nzv_threshold,
        impute_method=body.impute_method,
        impute_cols=body.impute_cols,
        custom_fill_value=body.custom_fill_value,
        outlier_method=body.outlier_method,
        outlier_cols=body.outlier_cols,
        zscore_threshold=body.zscore_threshold,
        winsor_lo=body.winsor_lo,
        winsor_hi=body.winsor_hi,
        cap_multiplier=body.cap_multiplier,
        domain_filters=domain_filters,
    )
    return CleaningResponse(**result)


@router.post("/preprocess/automated", response_model=CleaningResponse)
def apply_automated_cleaning(body: AutomatedCleaningRequest) -> CleaningResponse:
    result = preprocess_service.apply_automated_cleaning(body.dataset_name, body.new_dataset_name)
    return CleaningResponse(**result)


@router.post("/preprocess/apply", response_model=ApplyPreprocessingResponse)
def apply_preprocessing(body: ApplyPreprocessingRequest) -> ApplyPreprocessingResponse:
    domain_filters = (
        {k: v.model_dump() for k, v in body.domain_filters.items()}
        if body.domain_filters
        else None
    )
    result = preprocess_service.apply_preprocessing(
        dataset_name=body.dataset_name,
        x_cols=body.x_cols,
        y_cols=body.y_cols,
        imputation_method=body.imputation_method,
        outlier_method=body.outlier_method,
        domain_filters=domain_filters,
        split_method=body.split_method,
        test_size=body.test_size,
        stratify_bins=body.stratify_bins,
    )
    return ApplyPreprocessingResponse(**result)


@router.get("/projects", response_model=List[ProjectSummary])
def list_projects() -> List[ProjectSummary]:
    return [ProjectSummary(**p) for p in project_service.list_projects()]
