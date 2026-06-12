"""
src/data/preprocessing.py
==========================
Data preprocessing pipeline for the Soft Sensor Prediction System.

All functions are pure pandas / numpy / sklearn — no Streamlit calls.
The UI layer calls these functions, collects their return values, and
writes results back to ``st.session_state`` itself.

Pipeline steps
--------------
1. cast_to_numeric      — coerce object columns → float (messy strings → NaN)
2. compute_feature_stats— summary statistics before / after processing
3. impute               — fill NaN with mean / median / zero
4. apply_outlier_treatment — IQR capping  or  1 % – 99 % percentile capping
5. apply_custom_filters — per-tag user-defined [min, max] clipping
6. split_and_scale      — train/test split + StandardScaler normalisation

Public API
----------
cast_to_numeric(df)
compute_feature_stats(df)
impute(data_x, data_y, method)
apply_outlier_treatment(data_x, data_y, method)
apply_custom_filters(data_x, data_y, filters)
split_and_scale(data_x, data_y)
"""
from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.settings import RANDOM_STATE, TEST_SIZE

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ImputationMethod = Literal["Mean", "Median", "Zero"]
OutlierMethod = Literal[
    "None",
    "IQR Capping",
    "Min-Max Percentile Capping (1% - 99%)",
]


# ---------------------------------------------------------------------------
# 1. Type coercion
# ---------------------------------------------------------------------------


def cast_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force all object-dtype columns to numeric.

    Values that cannot be coerced become NaN.  This fixes issues where
    sensor data is accidentally parsed as text by openpyxl.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 2. Statistics summary
# ---------------------------------------------------------------------------


def compute_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-column statistics DataFrame suitable for display.

    Columns: Missing Count, Missing %, Min, Mean, Max
    """
    return pd.DataFrame(
        {
            "Missing Count": df.isnull().sum(),
            "Missing %": (df.isnull().sum() / len(df) * 100).round(2),
            "Min": df.min(),
            "Mean": df.mean(),
            "Max": df.max(),
        }
    )


# ---------------------------------------------------------------------------
# 3. Imputation
# ---------------------------------------------------------------------------


def impute(
    data_x: pd.DataFrame,
    data_y: pd.DataFrame,
    method: ImputationMethod = "Mean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill NaN values in feature and target DataFrames."""
    if method == "Mean":
        data_x = data_x.fillna(data_x.mean())
        data_y = data_y.fillna(data_y.mean())
    elif method == "Median":
        data_x = data_x.fillna(data_x.median())
        data_y = data_y.fillna(data_y.median())
    elif method == "Zero":
        data_x = data_x.fillna(0)
        data_y = data_y.fillna(0)
    return data_x, data_y


# ---------------------------------------------------------------------------
# 4. Outlier handling
# ---------------------------------------------------------------------------


def _iqr_cap(series: pd.Series) -> pd.Series:
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = Q3 - Q1
    return np.clip(series, Q1 - 1.5 * iqr, Q3 + 1.5 * iqr)


def _percentile_cap(series: pd.Series) -> pd.Series:
    return np.clip(series, series.quantile(0.01), series.quantile(0.99))


def apply_outlier_treatment(
    data_x: pd.DataFrame,
    data_y: pd.DataFrame,
    method: OutlierMethod = "None",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the chosen outlier treatment to both X and Y DataFrames.

    'None'                                 → pass-through (no changes)
    'IQR Capping'                          → clip at Q1 – 1.5·IQR / Q3 + 1.5·IQR
    'Min-Max Percentile Capping (1% - 99%)' → clip at 1 % / 99 % quantiles
    """
    if method == "None":
        return data_x, data_y

    fn = _iqr_cap if method == "IQR Capping" else _percentile_cap

    data_x = data_x.apply(fn)
    data_y = data_y.apply(fn)
    return data_x, data_y


# ---------------------------------------------------------------------------
# 5. Custom min-max clipping
# ---------------------------------------------------------------------------


def apply_custom_filters(
    data_x: pd.DataFrame,
    data_y: pd.DataFrame,
    custom_filters: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clip individual tags to user-specified [min, max] bounds.

    Parameters
    ----------
    custom_filters : {tag_name: {"min": float, "max": float}}
    """
    for tag, bounds in custom_filters.items():
        lo, hi = bounds["min"], bounds["max"]
        if tag in data_x.columns:
            data_x[tag] = np.clip(data_x[tag], lo, hi)
        if tag in data_y.columns:
            data_y[tag] = np.clip(data_y[tag], lo, hi)
    return data_x, data_y


# ---------------------------------------------------------------------------
# 6. Train / test split + StandardScaler normalisation
# ---------------------------------------------------------------------------


def split_and_scale(
    data_x: pd.DataFrame,
    data_y: pd.DataFrame,
    test_size: float | None = None,
    stratify_bins: int = 0,
    split_method: str = "random",
) -> Tuple[
    np.ndarray,     # X_train  (scaled)
    np.ndarray,     # X_test   (scaled)
    np.ndarray,     # y_train  (scaled)
    np.ndarray,     # y_test   (scaled)
    pd.DataFrame,   # y_test_raw (unscaled, for metric computation)
    StandardScaler, # scaler_x
    StandardScaler, # scaler_y
]:
    """
    Split into train / test partitions and apply StandardScaler.

    Parameters
    ----------
    test_size     : fraction for test set; defaults to ``TEST_SIZE`` in settings
    stratify_bins : >0 → stratified split using quantile-binning of the first
                    Y column into ``stratify_bins`` equal-frequency bins.
                    0 → plain random split (default).
    split_method  : "random" (default) or "sequential" (preserves row order;
                    first N rows → train, remaining → test — no shuffling).

    The scaler is fitted on the training partition only; the test partition
    is transformed with the already-fitted scaler to prevent data leakage.
    """
    effective_test_size = test_size if test_size is not None else TEST_SIZE

    if split_method == "sequential":
        n_total = len(data_x)
        n_train = int(n_total * (1.0 - effective_test_size))
        X_train = data_x.iloc[:n_train]
        X_test  = data_x.iloc[n_train:]
        y_train = data_y.iloc[:n_train]
        y_test  = data_y.iloc[n_train:]
    else:
        stratify_labels = None
        if stratify_bins > 0:
            first_y = data_y.iloc[:, 0]
            try:
                stratify_labels = pd.qcut(first_y, q=stratify_bins, labels=False, duplicates="drop")
            except Exception:
                stratify_labels = None  # fall back to random if binning fails

        X_train, X_test, y_train, y_test = train_test_split(
            data_x, data_y,
            test_size=effective_test_size,
            random_state=RANDOM_STATE,
            stratify=stratify_labels,
        )

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s  = scaler_x.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s  = scaler_y.transform(y_test)

    return X_train_s, X_test_s, y_train_s, y_test_s, y_test, scaler_x, scaler_y
