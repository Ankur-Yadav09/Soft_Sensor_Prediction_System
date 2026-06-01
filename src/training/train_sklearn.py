"""
src/training/train_sklearn.py
==============================
Training function for sklearn-compatible models:
    Random Forest, XGBoost, LightGBM.

All three models are wrapped in MultiOutputRegressor (or use native multi-output)
and trained on StandardScaler-scaled targets — the same convention used by the DAE.

Public API
----------
train_sklearn_model(X_train, y_train_scaled,
                    X_test, y_test_scaled, y_test_raw, y_cols, scaler_y,
                    model_type, **hparams)
    → (SklearnWrapper, loss_history_dict)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from src.models.wrappers import SklearnWrapper


def train_sklearn_model(
    X_train: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    y_test_raw,             # pd.DataFrame of unscaled test targets
    y_cols: List[str],
    scaler_y,               # fitted StandardScaler for Y
    model_type: str,
    **hparams,
) -> tuple:
    """
    Train a tree-ensemble model for multi-output soft sensor prediction.

    Parameters
    ----------
    X_train / X_test        : scaled numpy feature arrays
    y_train_scaled          : scaled numpy target array (N, output_dim)
    y_test_scaled           : scaled test targets
    y_test_raw              : unscaled test targets DataFrame
    y_cols                  : ordered target column names
    scaler_y                : fitted StandardScaler
    model_type              : "Random Forest" | "XGBoost" | "LightGBM"
    **hparams               : hyperparameters forwarded to the base estimator

    Returns
    -------
    wrapper      : SklearnWrapper ready for inference
    loss_history : minimal dict (no per-epoch losses for sklearn models)
    """
    # ---- Normalise array shapes ----
    # StandardScaler.fit_transform() always returns 2-D, but session-state arrays
    # may have been squeezed elsewhere.  Ensure 2-D throughout so that
    # RandomForestRegressor (which squeezes single-output predictions to 1-D)
    # and scaler.inverse_transform() both receive the expected shapes.
    X_train        = np.atleast_2d(X_train)
    X_test         = np.atleast_2d(X_test)
    y_train_scaled = np.array(y_train_scaled)
    y_test_scaled  = np.array(y_test_scaled)
    if y_train_scaled.ndim == 1:
        y_train_scaled = y_train_scaled.reshape(-1, 1)
    if y_test_scaled.ndim == 1:
        y_test_scaled = y_test_scaled.reshape(-1, 1)

    # ---- Build base estimator ----
    if model_type == "Random Forest":
        estimator = RandomForestRegressor(**hparams)
        # RandomForestRegressor natively supports multi-output
        model = estimator

    elif model_type == "XGBoost":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(eval_metric="rmse", **hparams)
        model = MultiOutputRegressor(estimator)

    elif model_type == "LightGBM":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(**hparams)
        model = MultiOutputRegressor(estimator)

    else:
        raise ValueError(f"Unknown sklearn model_type: {model_type!r}")

    # ---- Fit ----
    # RandomForestRegressor emits a DataConversionWarning when y is (N, 1);
    # it expects (N,) for single-output.  Squeeze to 1-D for RF only — the
    # predict() output is reshaped back to 2-D below.
    y_fit = (
        y_train_scaled.ravel()
        if model_type == "Random Forest" and y_train_scaled.shape[1] == 1
        else y_train_scaled
    )
    model.fit(X_train, y_fit)

    # ---- Quick validation metrics ----
    pred_scaled = model.predict(X_test)
    # RandomForestRegressor squeezes single-output predictions to 1-D;
    # scaler.inverse_transform() requires 2-D.
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(-1, 1)
    pred_raw = scaler_y.inverse_transform(pred_scaled)

    r2_vals  = [r2_score(y_test_raw[c], pred_raw[:, i])  for i, c in enumerate(y_cols)]
    mae_vals = [mean_absolute_error(y_test_raw[c], pred_raw[:, i]) for i, c in enumerate(y_cols)]

    avg_r2  = float(np.mean(r2_vals))
    avg_mae = float(np.mean(mae_vals))

    # Minimal loss_history so train.py post-training code doesn't need branching
    loss_history: Dict = {
        "epoch_recon_losses": [],
        "epoch_pred_losses":  [],
        "val_recon_losses":   [],
        "val_pred_losses":    [],
        "actual_epochs":      1,
        "early_stopped":      False,
        "final_r2":           avg_r2,
        "final_mae":          avg_mae,
    }

    return SklearnWrapper(model, model_type), loss_history
