"""
src/training/train_kalman.py
==============================
Training function for the Kalman Filter soft sensor model.

The Kalman Filter is applied as a recursive linear estimator — a
Kalman-Filter-based Recursive Least Squares (KF-RLS) regressor. Each target
column gets its own filter whose state is the vector of linear regression
coefficients (plus bias). Every training sample is treated as one step of a
linear-Gaussian state-space system:

    theta_t = theta_(t-1) + w_t,      w_t ~ N(0, Q)   (state transition)
    y_t     = H_t . theta_t + v_t,    v_t ~ N(0, R)   (measurement)

where H_t is the feature row [x_t, 1] (bias appended). Allowing a small
process noise Q (instead of freezing the state as in classic RLS) lets the
filter keep adapting to slow drift in the underlying process — a common
requirement for industrial soft sensors.

Multi-Y is supported the same way MultiOutputRegressor handles it for the
tree ensembles: one independent filter per output column, all driven by the
same feature rows.

Public API
----------
train_kalman_model(X_train, y_train_scaled,
                    X_test, y_test_scaled, y_test_raw, y_cols, scaler_y,
                    **hparams)
    -> (SklearnWrapper, loss_history_dict)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from src.models.wrappers import SklearnWrapper


class _KalmanFilterRegressor:
    """
    Recursive linear regressor fit with a per-output Kalman Filter.

    Exposes the same ``fit(X, y)`` / ``predict(X)`` surface as an
    sklearn estimator so it can be wrapped with the existing
    ``SklearnWrapper`` used by the tree-ensemble models.
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
        initial_covariance: float = 1.0,
        n_epochs: int = 10,
        random_state: int = 42,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_covariance = initial_covariance
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_KalmanFilterRegressor":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        n_states = n_features + 1  # + bias term

        # Augment features with a constant column for the bias state.
        X_aug = np.hstack([X, np.ones((n_samples, 1))])

        Q = np.eye(n_states) * self.process_noise
        R = self.measurement_noise

        theta = np.zeros((n_states, n_outputs))
        P = [np.eye(n_states) * self.initial_covariance for _ in range(n_outputs)]

        rng = np.random.RandomState(self.random_state)
        order = np.arange(n_samples)

        # Multiple shuffled passes let the filter converge on small
        # datasets, since a single online pass rarely sees enough
        # samples to settle the state estimate.
        for _ in range(self.n_epochs):
            rng.shuffle(order)
            for i in order:
                h = X_aug[i]  # feature row, shape (n_states,)
                for j in range(n_outputs):
                    # ---- Predict: random-walk state, inflate covariance ----
                    P_pred = P[j] + Q

                    # ---- Update: fuse the new (x, y) observation ----
                    innovation_cov = float(h @ P_pred @ h.T) + R
                    kalman_gain = (P_pred @ h) / innovation_cov
                    residual = y[i, j] - float(h @ theta[:, j])
                    theta[:, j] = theta[:, j] + kalman_gain * residual
                    P[j] = P_pred - np.outer(kalman_gain, h) @ P_pred

        self.theta_ = theta
        self.n_features_ = n_features
        self.n_outputs_ = n_outputs
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return X_aug @ self.theta_  # shape (n_samples, n_outputs)


def train_kalman_model(
    X_train: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    y_test_raw,             # pd.DataFrame of unscaled test targets
    y_cols: List[str],
    scaler_y,               # fitted StandardScaler for Y
    **hparams,
) -> tuple:
    """
    Train a Kalman-Filter-based recursive linear model for multi-output
    soft sensor prediction.

    Parameters
    ----------
    X_train / X_test   : scaled numpy feature arrays
    y_train_scaled      : scaled numpy target array (N, output_dim)
    y_test_scaled        : scaled test targets
    y_test_raw           : unscaled test targets DataFrame
    y_cols               : ordered target column names
    scaler_y             : fitted StandardScaler
    **hparams            : process_noise, measurement_noise,
                            initial_covariance, n_epochs, random_state

    Returns
    -------
    wrapper      : SklearnWrapper ready for inference
    loss_history : minimal dict matching the format used by the other
                   non-epoch-curve models (tree ensembles)
    """
    # ---- Normalise array shapes ----
    X_train        = np.atleast_2d(X_train)
    X_test         = np.atleast_2d(X_test)
    y_train_scaled = np.array(y_train_scaled)
    y_test_scaled  = np.array(y_test_scaled)
    if y_train_scaled.ndim == 1:
        y_train_scaled = y_train_scaled.reshape(-1, 1)
    if y_test_scaled.ndim == 1:
        y_test_scaled = y_test_scaled.reshape(-1, 1)

    # ---- Fit ----
    model = _KalmanFilterRegressor(**hparams)
    model.fit(X_train, y_train_scaled)

    # ---- Quick validation metrics ----
    pred_scaled = model.predict(X_test)
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
        "actual_epochs":      hparams.get("n_epochs", 10),
        "early_stopped":      False,
        "final_r2":           avg_r2,
        "final_mae":          avg_mae,
    }

    return SklearnWrapper(model, "Kalman Filter"), loss_history
