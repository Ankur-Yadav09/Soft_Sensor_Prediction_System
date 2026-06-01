"""
src/models/wrappers.py
======================
Thin wrapper classes that give every model type a unified inference API.

Every wrapper exposes:
    predict_scaled(X_scaled: np.ndarray) -> np.ndarray  shape (N, output_dim)
        — returns predictions still in SCALED space (caller applies scaler_y.inverse_transform)
    model_type: str
        — one of "DAE" | "Random Forest" | "XGBoost" | "LightGBM" | "LSTM"

This lets predict.py, overview.py, and the What-If engine work with any model
without knowing its internal framework (PyTorch vs sklearn).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DAE wrapper
# ---------------------------------------------------------------------------


class DAEWrapper:
    model_type = "DAE"

    def __init__(self, dae) -> None:
        self.model = dae  # IndustrialDAE instance

    def predict_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        t = torch.tensor(X_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            _, pred = self.model(t)
        return pred.numpy()


# ---------------------------------------------------------------------------
# Sklearn / XGBoost / LightGBM wrapper
# ---------------------------------------------------------------------------


class SklearnWrapper:
    def __init__(self, model, model_type: str) -> None:
        self.model = model          # fitted sklearn-compatible estimator
        self.model_type = model_type

    def predict_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        result = self.model.predict(X_scaled)
        # Ensure 2-D array (N, output_dim) even for single-output models
        if result.ndim == 1:
            result = result[:, np.newaxis]
        return result


# ---------------------------------------------------------------------------
# LSTM architecture
# ---------------------------------------------------------------------------


class LSTMPredictor(nn.Module):
    """
    Sequence-to-value LSTM.

    Processes each input as (batch, window_size, input_size) and predicts
    (batch, output_size) from the last hidden state.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        output_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        return self.head(out[:, -1, :])  # last timestep → (batch, output_size)


# ---------------------------------------------------------------------------
# LSTM wrapper
# ---------------------------------------------------------------------------


class LSTMWrapper:
    model_type = "LSTM"

    def __init__(self, lstm_model: LSTMPredictor, window_size: int = 1) -> None:
        self.model       = lstm_model
        self.window_size = window_size

    def predict_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        # Tile each row window_size times to form a (N, W, features) sequence.
        # This "steady-state" assumption keeps the inference API consistent
        # regardless of whether data is ordered chronologically.
        seq = np.stack([X_scaled] * self.window_size, axis=1)  # (N, W, F)
        t = torch.tensor(seq, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(t)
        return pred.numpy()
