"""
backend/app/services/preprocess_service.py
==============================================
Two distinct jobs, matching the real Streamlit app's page split:

1. Whole-dataset cleaning (Data Understanding / Basic Preprocessing /
   Automated Preprocessing) — src/ui/pages/preprocess.py never touches X/Y
   or the train/test split; it only cleans the raw dataframe and keeps the
   result in st.session_state.df. There is no stateless equivalent of "an
   in-memory working copy" in this backend, so "Apply Cleaning" here
   persists the cleaned result as a NEW dataset (via save_dataset_to_db)
   instead — a deliberate improvement (survives a backend restart) rather
   than a bug. The caller picks the new dataset name for downstream steps
   exactly like any other dataset.

2. apply_preprocessing() — the actual impute -> outlier-treat -> filter ->
   split pipeline. In the real app this is Feature Selection's "Final
   Apply" step (after target/feature selection), not Preprocess's. The
   route stays under /api/preprocess for continuity but is now called from
   the Feature Selection page's last step.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from src.data.database import load_dataset_from_db, save_dataset_to_db
from src.data.preprocessing import (
    apply_custom_filters,
    apply_outlier_treatment,
    cast_to_numeric,
    compute_feature_stats,
    impute,
    split_and_scale,
)

from backend.app.services import project_service


def _normalize_extension_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    cast_to_numeric() only recognises plain ``object`` dtype columns, but
    pandas' Parquet round-trip (src.data.database.load_dataset_from_db)
    loads text columns as its newer pandas ``string`` extension dtype,
    which slips past that check uncoerced and later crashes
    compute_feature_stats()'s df.mean() with "Cannot perform reduction
    'mean' with string dtype". Pre-normalizing those columns to plain
    object here — without touching cast_to_numeric itself — lets its
    existing dtype check work as originally intended.
    """
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != object:
            df[col] = df[col].astype(object)
    return df


def _load_and_validate(dataset_name: str, x_cols: List[str], y_cols: List[str]):
    df = load_dataset_from_db(dataset_name)
    if df is None:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset '{dataset_name}' could not be loaded.",
        )
    missing = [c for c in (*x_cols, *y_cols) if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset '{dataset_name}' is missing columns: {missing}",
        )
    return df


def get_feature_stats(dataset_name: str) -> List[dict]:
    df = load_dataset_from_db(dataset_name)
    if df is None:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset '{dataset_name}' could not be loaded.",
        )
    df = cast_to_numeric(_normalize_extension_dtypes(df))  # unchanged
    stats_df = compute_feature_stats(df)  # unchanged
    stats_df = stats_df.reset_index().rename(columns={"index": "Feature"})
    return json.loads(stats_df.to_json(orient="records"))


def apply_preprocessing(
    dataset_name: str,
    x_cols: List[str],
    y_cols: List[str],
    imputation_method: str,
    outlier_method: str,
    domain_filters: Optional[Dict[str, Dict[str, float]]],
    split_method: str,
    test_size: Optional[float],
    stratify_bins: int = 0,
) -> dict:
    df = _load_and_validate(dataset_name, x_cols, y_cols)
    df = cast_to_numeric(_normalize_extension_dtypes(df))  # unchanged

    data_x = df[x_cols].copy()
    data_y = df[y_cols].copy()

    data_x, data_y = impute(data_x, data_y, method=imputation_method)  # unchanged
    data_x, data_y = apply_outlier_treatment(data_x, data_y, method=outlier_method)  # unchanged
    if domain_filters:
        data_x, data_y = apply_custom_filters(data_x, data_y, domain_filters)  # unchanged

    X_train, X_test, y_train, y_test, y_test_raw, scaler_x, scaler_y = split_and_scale(
        data_x, data_y, test_size=test_size, stratify_bins=stratify_bins, split_method=split_method
    )  # unchanged

    project_id = project_service.create_project(
        dataset_name=dataset_name,
        x_cols=x_cols,
        y_cols=y_cols,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_test_raw=y_test_raw,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        config={
            "imputation_method": imputation_method,
            "outlier_method": outlier_method,
            "domain_filters": domain_filters or {},
            "split_method": split_method,
            "test_size": test_size,
        },
    )

    return {
        "project_id": project_id,
        "dataset_name": dataset_name,
        "x_cols": x_cols,
        "y_cols": y_cols,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ---------------------------------------------------------------------------
# Data Understanding — per-feature deep dive (src/ui/pages/preprocess.py's
# _render_data_understanding, ported 1:1 since the real page implements this
# logic itself rather than calling a shared src/ function).
# ---------------------------------------------------------------------------


def _distribution_label(skew: float) -> str:
    if skew > 1.0:
        return "Highly right-skewed"
    if skew > 0.5:
        return "Moderately right-skewed"
    if skew < -1.0:
        return "Highly left-skewed"
    if skew < -0.5:
        return "Moderately left-skewed"
    return "Approximately symmetric"


def _count_iqr_outliers(series: pd.Series) -> int:
    s = series.dropna()
    if len(s) < 4:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def get_feature_detail(dataset_name: str, column: str) -> dict:
    df = load_dataset_from_db(dataset_name)
    if df is None:
        raise HTTPException(status_code=422, detail=f"Dataset '{dataset_name}' could not be loaded.")
    df = cast_to_numeric(_normalize_extension_dtypes(df))
    if column not in df.columns:
        raise HTTPException(status_code=422, detail=f"Column '{column}' not found in '{dataset_name}'.")

    series_raw = df[column]
    series = series_raw.dropna()
    n_total = len(df)
    n_missing = int(series_raw.isnull().sum())
    n_unique = int(series_raw.nunique())
    n_dupes = int(df.duplicated().sum())
    n_outliers = _count_iqr_outliers(series_raw)

    if len(series) == 0:
        return {
            "column": column,
            "n_total": n_total,
            "n_missing": n_missing,
            "empty": True,
        }

    mean_v, median_v, std_v = float(series.mean()), float(series.median()), float(series.std())
    min_v, max_v = float(series.min()), float(series.max())
    skew_v, kurt_v = float(series.skew()), float(series.kurtosis())

    counts, bin_edges = np.histogram(series, bins=40)

    return {
        "column": column,
        "empty": False,
        "dtype": str(series_raw.dtype),
        "distribution_label": _distribution_label(skew_v),
        "n_total": n_total,
        "n_missing": n_missing,
        "n_unique": n_unique,
        "n_duplicate_rows": n_dupes,
        "n_outliers_iqr": n_outliers,
        "mean": mean_v,
        "median": median_v,
        "std": std_v,
        "min": min_v,
        "max": max_v,
        "skew": skew_v,
        "kurtosis": kurt_v,
        "histogram": {"counts": counts.tolist(), "bin_edges": bin_edges.tolist()},
        "boxplot": {
            "min": min_v,
            "q1": float(series.quantile(0.25)),
            "median": median_v,
            "q3": float(series.quantile(0.75)),
            "max": max_v,
        },
    }


# ---------------------------------------------------------------------------
# Basic Preprocessing — ported from src/ui/pages/preprocess.py's inline
# cleaning helpers (that page implements its own outlier/impute logic rather
# than calling src.data.preprocessing, so this mirrors the page, not the
# shared pipeline module).
# ---------------------------------------------------------------------------


def _apply_zscore_cap(working: pd.DataFrame, cols: List[str], thr: float):
    total = 0
    for col in cols:
        if col not in working.columns:
            continue
        s = working[col]
        mu, sigma = s.mean(), s.std()
        if sigma == 0:
            continue
        lo, hi = mu - thr * sigma, mu + thr * sigma
        total += int(((s < lo) | (s > hi)).sum())
        working[col] = s.clip(lo, hi)
    return working, total


def _apply_winsorization(working: pd.DataFrame, cols: List[str], lo_pct: float, hi_pct: float):
    total = 0
    for col in cols:
        if col not in working.columns:
            continue
        s = working[col]
        lo, hi = s.quantile(lo_pct / 100), s.quantile(hi_pct / 100)
        total += int(((s < lo) | (s > hi)).sum())
        working[col] = s.clip(lo, hi)
    return working, total


def _apply_capping_flooring(working: pd.DataFrame, cols: List[str], multiplier: float):
    total = 0
    for col in cols:
        if col not in working.columns:
            continue
        s = working[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        total += int(((s < lo) | (s > hi)).sum())
        working[col] = s.clip(lo, hi)
    return working, total


def _apply_remove_outliers_iqr(working: pd.DataFrame, cols: List[str]):
    n_before = len(working)
    mask = pd.Series(True, index=working.index)
    for col in cols:
        if col not in working.columns:
            continue
        s = working[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= (s >= lo) & (s <= hi)
    result = working[mask].reset_index(drop=True)
    return result, n_before - len(result)


def _apply_remove_outliers_zscore(working: pd.DataFrame, cols: List[str], thr: float):
    n_before = len(working)
    mask = pd.Series(True, index=working.index)
    for col in cols:
        if col not in working.columns:
            continue
        s = working[col]
        if s.std() == 0:
            continue
        z = (s - s.mean()) / s.std()
        mask &= z.abs() <= thr
    result = working[mask].reset_index(drop=True)
    return result, n_before - len(result)


def apply_basic_cleaning(
    dataset_name: str,
    new_dataset_name: Optional[str],
    remove_missing_rows: bool,
    remove_duplicates: bool,
    remove_missing_cols: bool,
    missing_col_threshold: float,
    remove_constant_cols: bool,
    remove_nzv_cols: bool,
    nzv_threshold: float,
    impute_method: str,
    impute_cols: Optional[List[str]],
    custom_fill_value: float,
    outlier_method: str,
    outlier_cols: Optional[List[str]],
    zscore_threshold: float,
    winsor_lo: float,
    winsor_hi: float,
    cap_multiplier: float,
    domain_filters: Optional[Dict[str, Dict[str, float]]],
) -> dict:
    df = load_dataset_from_db(dataset_name)
    if df is None:
        raise HTTPException(status_code=422, detail=f"Dataset '{dataset_name}' could not be loaded.")

    working = cast_to_numeric(_normalize_extension_dtypes(df)).copy()
    before_rows, before_cols = working.shape
    action_log: List[str] = []

    if remove_missing_rows:
        n_before = len(working)
        working = working.dropna().reset_index(drop=True)
        action_log.append(f"Removed {n_before - len(working)} row(s) with missing values.")

    if remove_duplicates:
        n_before = len(working)
        working = working.drop_duplicates().reset_index(drop=True)
        action_log.append(f"Removed {n_before - len(working)} duplicate row(s).")

    if remove_missing_cols:
        num_cols_now = working.select_dtypes(include=[np.number]).columns.tolist()
        drop_cols = [c for c in num_cols_now if working[c].isnull().mean() * 100 >= missing_col_threshold]
        if drop_cols:
            working = working.drop(columns=drop_cols)
        action_log.append(
            f"Removed {len(drop_cols)} column(s) with >= {missing_col_threshold:.0f}% missing"
            + (f": {', '.join(drop_cols)}" if drop_cols else "") + "."
        )

    if remove_constant_cols:
        num_cols_now = working.select_dtypes(include=[np.number]).columns.tolist()
        drop_cols = [c for c in num_cols_now if working[c].std() == 0]
        if drop_cols:
            working = working.drop(columns=drop_cols)
        action_log.append(
            f"Removed {len(drop_cols)} constant column(s)"
            + (f": {', '.join(drop_cols)}" if drop_cols else "") + "."
        )

    if remove_nzv_cols:
        num_cols_now = working.select_dtypes(include=[np.number]).columns.tolist()
        drop_cols = [c for c in num_cols_now if 0 < working[c].std() < nzv_threshold]
        if drop_cols:
            working = working.drop(columns=drop_cols)
        action_log.append(
            f"Removed {len(drop_cols)} near-zero variance column(s) (std < {nzv_threshold})"
            + (f": {', '.join(drop_cols)}" if drop_cols else "") + "."
        )

    if impute_method != "None":
        numeric_cols = working.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = [c for c in (impute_cols or [c for c in numeric_cols if working[c].isnull().any()]) if c in working.columns]
        n_filled = 0
        for col in target_cols:
            n_miss = int(working[col].isnull().sum())
            if n_miss == 0:
                continue
            if impute_method == "Mean":
                working[col] = working[col].fillna(working[col].mean())
            elif impute_method == "Median":
                working[col] = working[col].fillna(working[col].median())
            elif impute_method == "Mode":
                mode_val = working[col].mode()
                working[col] = working[col].fillna(mode_val.iloc[0] if not mode_val.empty else 0)
            elif impute_method == "Forward Fill":
                working[col] = working[col].ffill()
            elif impute_method == "Backward Fill":
                working[col] = working[col].bfill()
            elif impute_method == "Custom Value":
                working[col] = working[col].fillna(custom_fill_value)
            n_filled += n_miss
        action_log.append(f"Imputed {n_filled} missing value(s) using {impute_method} across {len(target_cols)} column(s).")

    resolved_outlier_cols = [c for c in (outlier_cols or []) if c in working.columns]
    if outlier_method != "None" and resolved_outlier_cols:
        if outlier_method == "IQR Capping":
            working, n = _apply_capping_flooring(working, resolved_outlier_cols, 1.5)
            action_log.append(f"IQR capping applied — {n} value(s) capped.")
        elif outlier_method == "Z-Score Capping":
            working, n = _apply_zscore_cap(working, resolved_outlier_cols, zscore_threshold)
            action_log.append(f"Z-Score capping (thr={zscore_threshold}) — {n} value(s) capped.")
        elif outlier_method == "Winsorization":
            working, n = _apply_winsorization(working, resolved_outlier_cols, winsor_lo, winsor_hi)
            action_log.append(f"Winsorization ({winsor_lo}%-{winsor_hi}%) — {n} value(s) capped.")
        elif outlier_method == "Capping/Flooring (custom IQR multiplier)":
            working, n = _apply_capping_flooring(working, resolved_outlier_cols, cap_multiplier)
            action_log.append(f"IQR capping (x{cap_multiplier}) — {n} value(s) capped.")
        elif outlier_method == "Remove Outliers (IQR)":
            working, n = _apply_remove_outliers_iqr(working, resolved_outlier_cols)
            action_log.append(f"Removed {n} outlier row(s) via IQR.")
        elif outlier_method == "Remove Outliers (Z-Score)":
            working, n = _apply_remove_outliers_zscore(working, resolved_outlier_cols, zscore_threshold)
            action_log.append(f"Removed {n} outlier row(s) via Z-Score (thr={zscore_threshold}).")

    if domain_filters:
        for tag, bounds in domain_filters.items():
            if tag in working.columns:
                working[tag] = working[tag].clip(bounds["min"], bounds["max"])
        action_log.append(f"Domain filters applied to {len(domain_filters)} tag(s).")

    after_rows, after_cols = working.shape
    result_name = new_dataset_name or f"{dataset_name}_cleaned"
    save_dataset_to_db(result_name, working)  # unchanged

    return {
        "new_dataset_name": result_name,
        "before_rows": before_rows,
        "after_rows": after_rows,
        "before_cols": before_cols,
        "after_cols": after_cols,
        "action_log": action_log,
    }


def apply_automated_cleaning(dataset_name: str, new_dataset_name: Optional[str]) -> dict:
    df = load_dataset_from_db(dataset_name)
    if df is None:
        raise HTTPException(status_code=422, detail=f"Dataset '{dataset_name}' could not be loaded.")

    working = cast_to_numeric(_normalize_extension_dtypes(df)).copy()
    n_rows_start = len(working)
    step_log: List[str] = []

    n_before = len(working)
    working = working.drop_duplicates().reset_index(drop=True)
    step_log.append(f"Remove duplicates — {n_before - len(working)} duplicate row(s) removed.")

    num_cols = working.select_dtypes(include=[np.number]).columns.tolist()
    miss_drop = [c for c in num_cols if working[c].isnull().mean() >= 0.50]
    if miss_drop:
        working = working.drop(columns=miss_drop)
    step_log.append(f"Drop high-missing columns (>=50%) — {len(miss_drop)} column(s) removed" + (f": {', '.join(miss_drop)}" if miss_drop else "") + ".")

    num_cols = working.select_dtypes(include=[np.number]).columns.tolist()
    const_drop = [c for c in num_cols if working[c].std() == 0]
    if const_drop:
        working = working.drop(columns=const_drop)
    step_log.append(f"Remove constant columns — {len(const_drop)} column(s) removed" + (f": {', '.join(const_drop)}" if const_drop else "") + ".")

    num_cols = working.select_dtypes(include=[np.number]).columns.tolist()
    nzv_drop = [c for c in num_cols if 0 < working[c].std() < 0.01]
    if nzv_drop:
        working = working.drop(columns=nzv_drop)
    step_log.append(f"Remove near-zero variance columns (std<0.01) — {len(nzv_drop)} column(s) removed" + (f": {', '.join(nzv_drop)}" if nzv_drop else "") + ".")

    n_filled = 0
    for col in working.select_dtypes(include=[np.number]).columns:
        n_miss = int(working[col].isnull().sum())
        if n_miss > 0:
            working[col] = working[col].fillna(working[col].median())
            n_filled += n_miss
    step_log.append(f"Median imputation — {n_filled} missing value(s) filled.")

    n_capped = 0
    for col in working.select_dtypes(include=[np.number]).columns:
        s = working[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_capped += int(((s < lo) | (s > hi)).sum())
        working[col] = s.clip(lo, hi)
    step_log.append(f"IQR capping (1.5x) — {n_capped} value(s) capped.")

    result_name = new_dataset_name or f"{dataset_name}_auto_cleaned"
    save_dataset_to_db(result_name, working)  # unchanged

    return {
        "new_dataset_name": result_name,
        "before_rows": n_rows_start,
        "after_rows": len(working),
        "after_cols": working.shape[1],
        "step_log": step_log,
    }
