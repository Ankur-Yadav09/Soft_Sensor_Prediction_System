"""
src/simulation/what_if.py
==========================
What-If sensitivity analysis engine.

Sweeps one or more X features across their operating ranges while holding
the rest constant, running model inference at each step, and labelling
the direction of change in each target KPI.

All functions are pure NumPy / PyTorch — no Streamlit calls.

Public API
----------
build_sweep_array(feat_min, feat_max, step_size)   → np.ndarray
run_single_sweep(...)                               → pd.DataFrame
run_multi_sweep(...)                                → list[dict]
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config.settings import MAX_SWEEP_POINTS, TREND_EPSILON


# ---------------------------------------------------------------------------
# Trend labelling
# ---------------------------------------------------------------------------


def _compute_trend(preds: np.ndarray) -> List[str]:
    """
    Label each step as Increasing / Decreasing / Constant.

    The first element is always "-" (no previous step to compare to).
    """
    trends = ["-"]
    for i in range(1, len(preds)):
        diff = preds[i] - preds[i - 1]
        if diff > TREND_EPSILON:
            trends.append("Increasing 📈")
        elif diff < -TREND_EPSILON:
            trends.append("Decreasing 📉")
        else:
            trends.append("Constant ➖")
    return trends


# ---------------------------------------------------------------------------
# Sweep array construction
# ---------------------------------------------------------------------------


def build_sweep_array(
    feat_min: float,
    feat_max: float,
    step_size: float,
) -> np.ndarray:
    """
    Build an evenly-spaced sweep array from feat_min to feat_max.

    If min == max, a ±1 guard band is applied so the array has at least
    two points.  The result is capped at MAX_SWEEP_POINTS entries.
    """
    if feat_min == feat_max:
        feat_min -= 1.0
        feat_max += 1.0

    arr = np.arange(feat_min, feat_max + step_size, step_size)
    if len(arr) > MAX_SWEEP_POINTS:
        arr = arr[:MAX_SWEEP_POINTS]
    return arr


# ---------------------------------------------------------------------------
# Internal: run model inference over a pre-built DataFrame
# ---------------------------------------------------------------------------


def _predict_sweep(
    sim_df: pd.DataFrame,
    x_cols: List[str],
    model,
    scaler_x,
    scaler_y,
) -> np.ndarray:
    """Scale, run inference via the wrapper API, and inverse-transform predictions."""
    input_scaled = scaler_x.transform(sim_df[x_cols])
    pred_scaled  = model.predict_scaled(input_scaled)
    return scaler_y.inverse_transform(pred_scaled)


# ---------------------------------------------------------------------------
# Single-feature sweep
# ---------------------------------------------------------------------------


def run_single_sweep(
    vary_feat: str,
    sweep_vals: np.ndarray,
    constant_features: Dict[str, Dict],
    x_cols: List[str],
    model,
    scaler_x,
    scaler_y,
    target_y_cols: List[str],
    y_cols: List[str],
) -> pd.DataFrame:
    """
    Vary one feature across sweep_vals while holding all others constant.

    Parameters
    ----------
    vary_feat        : name of the feature being swept
    sweep_vals       : array of values for that feature
    constant_features: {feat: {"value": float}} for all non-varying features
    x_cols           : full ordered list of input feature names
    model            : trained IndustrialDAE
    scaler_x / _y   : fitted StandardScalers
    target_y_cols    : KPI columns to observe (subset of y_cols)
    y_cols           : full ordered list of target column names

    Returns
    -------
    DataFrame with columns:
        [vary_feat, "Predicted <ty>", "Trend <ty>", ...]  for each ty
    """
    # Build simulation DataFrame — vary_feat changes, others are constant
    sim_df = pd.DataFrame(
        {
            col: sweep_vals if col == vary_feat else constant_features[col]["value"]
            for col in x_cols
        }
    )

    pred_sim = _predict_sweep(sim_df, x_cols, model, scaler_x, scaler_y)

    result = pd.DataFrame({vary_feat: sweep_vals})
    for ty in target_y_cols:
        y_idx = y_cols.index(ty)
        preds = pred_sim[:, y_idx]
        result[f"Predicted {ty}"] = preds
        result[f"Trend {ty}"]     = _compute_trend(preds)

    return result


# ---------------------------------------------------------------------------
# Multi-feature sweep
# ---------------------------------------------------------------------------


def run_multi_sweep(
    sweep_arrays: Dict[str, np.ndarray],
    constant_features: Dict[str, Dict],
    varying_features: Dict[str, Dict],
    x_cols: List[str],
    df: pd.DataFrame,               # original DataFrame (for column means)
    model,
    scaler_x,
    scaler_y,
    target_y_cols: List[str],
    y_cols: List[str],
) -> List[Dict[str, Any]]:
    """
    For each varying feature, sweep it independently while:
      * constant features stay at their specified values
      * other varying features are held at their data mean

    Returns
    -------
    List of dicts:  {"x": vary_feat, "y": ty, "df": result_df}
    One entry per (varying_feature × target_kpi) combination.
    """
    all_results: List[Dict[str, Any]] = []

    for vary_feat, arr in sweep_arrays.items():
        sim_df = pd.DataFrame(
            {
                col: (
                    arr                                   if col == vary_feat
                    else constant_features[col]["value"]  if col in constant_features
                    else float(df[col].mean())            # other varying feats → mean
                )
                for col in x_cols
            }
        )

        pred_sim = _predict_sweep(sim_df, x_cols, model, scaler_x, scaler_y)

        for ty in target_y_cols:
            y_idx = y_cols.index(ty)
            preds = pred_sim[:, y_idx]

            res_df = pd.DataFrame(
                {
                    vary_feat:          arr,
                    f"Predicted {ty}":  preds,
                    "Trend":            _compute_trend(preds),
                }
            )
            all_results.append({"x": vary_feat, "y": ty, "df": res_df})

    return all_results
