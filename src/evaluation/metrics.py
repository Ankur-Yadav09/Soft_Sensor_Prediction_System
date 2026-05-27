"""
src/evaluation/metrics.py
==========================
Evaluation metrics for soft sensor model assessment.

All functions are pure Python / NumPy / sklearn — no Streamlit calls.

Public API
----------
compute_metrics(y_true_df, preds_array, y_cols) → pd.DataFrame
grade_r2(r2_value)                               → tuple[str, str]
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import R2_EXCELLENT, R2_GOOD


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true_df: pd.DataFrame,
    preds_array: np.ndarray,
    y_cols: List[str],
) -> pd.DataFrame:
    """
    Compute RMSE, MAE, R², and MAPE for every target column.

    Parameters
    ----------
    y_true_df   : DataFrame of actual values; columns must include all y_cols
    preds_array : numpy array of shape (N, len(y_cols)) — model predictions
    y_cols      : ordered list of target column names

    Returns
    -------
    DataFrame indexed by y_cols with columns
    ['RMSE', 'MAE', 'R2 Score', 'MAPE (%)']
    """
    records: dict = {}

    for i, col in enumerate(y_cols):
        actual    = y_true_df[col].values
        predicted = preds_array[:, i]

        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae  = float(mean_absolute_error(actual, predicted))
        r2   = float(r2_score(actual, predicted))

        nonzero = actual != 0
        mape = (
            float(
                np.mean(
                    np.abs(
                        (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
                    )
                )
                * 100
            )
            if nonzero.sum() > 0
            else 0.0
        )

        records[col] = {
            "RMSE":     rmse,
            "MAE":      mae,
            "R2 Score": r2,
            "MAPE (%)": mape,
        }

    return pd.DataFrame(records).T


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------


def grade_r2(r2_value: float) -> Tuple[str, str]:
    """
    Map an R² value to a (label, emoji) tuple for display.

    Returns
    -------
    (label, emoji)  e.g. ("Excellent", "🟢")
    """
    if r2_value >= R2_EXCELLENT:
        return "Excellent", "🟢"
    elif r2_value >= R2_GOOD:
        return "Good", "🟡"
    return "Needs Improvement", "🔴"


def r2_emoji(r2_value: float) -> str:
    """Return only the traffic-light emoji for a given R² value."""
    _, emoji = grade_r2(r2_value)
    return emoji
