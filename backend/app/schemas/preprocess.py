from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FeatureStatsResponse(BaseModel):
    stats: List[dict]


class DomainFilter(BaseModel):
    min: float
    max: float


class ApplyPreprocessingRequest(BaseModel):
    dataset_name: str
    x_cols: List[str]
    y_cols: List[str]
    imputation_method: str = "Mean"
    outlier_method: str = "None"
    domain_filters: Optional[Dict[str, DomainFilter]] = None
    split_method: str = "random"
    test_size: Optional[float] = None
    stratify_bins: int = 0


class ApplyPreprocessingResponse(BaseModel):
    project_id: str
    dataset_name: str
    x_cols: List[str]
    y_cols: List[str]
    n_train: int
    n_test: int


class ProjectSummary(BaseModel):
    project_id: str
    dataset_name: str
    x_cols: List[str]
    y_cols: List[str]
    created_at: str
    n_train: int
    n_test: int
    config: Dict[str, Any]


class FeatureDetailResponse(BaseModel):
    column: str
    empty: bool
    dtype: Optional[str] = None
    distribution_label: Optional[str] = None
    n_total: int
    n_missing: int
    n_unique: Optional[int] = None
    n_duplicate_rows: Optional[int] = None
    n_outliers_iqr: Optional[int] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skew: Optional[float] = None
    kurtosis: Optional[float] = None
    histogram: Optional[Dict[str, List[float]]] = None
    boxplot: Optional[Dict[str, float]] = None


class BasicCleaningRequest(BaseModel):
    dataset_name: str
    new_dataset_name: Optional[str] = None
    remove_missing_rows: bool = False
    remove_duplicates: bool = False
    remove_missing_cols: bool = False
    missing_col_threshold: float = 50.0
    remove_constant_cols: bool = False
    remove_nzv_cols: bool = False
    nzv_threshold: float = 0.01
    impute_method: str = "None"
    impute_cols: Optional[List[str]] = None
    custom_fill_value: float = 0.0
    outlier_method: str = "None"
    outlier_cols: Optional[List[str]] = None
    zscore_threshold: float = 3.0
    winsor_lo: float = 2.5
    winsor_hi: float = 97.5
    cap_multiplier: float = 1.5
    domain_filters: Optional[Dict[str, DomainFilter]] = None


class CleaningResponse(BaseModel):
    new_dataset_name: str
    before_rows: int
    after_rows: int
    before_cols: Optional[int] = None
    after_cols: int
    action_log: Optional[List[str]] = None
    step_log: Optional[List[str]] = None


class AutomatedCleaningRequest(BaseModel):
    dataset_name: str
    new_dataset_name: Optional[str] = None
