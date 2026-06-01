"""
src/training/train_lstm.py
===========================
Training function for the LSTM-based soft sensor predictor.

Architecture
------------
Each sample is presented as a repeated sequence of shape (window_size, input_dim).
This "steady-state tiling" keeps the inference API identical to other models:
predict_scaled(X_scaled) works on any (N, F) array without temporal ordering.

The training loop mirrors trainer.py:
  * Adam optimiser with ReduceLROnPlateau scheduling
  * Best-model checkpointing (save state_dict at lowest val_pred_loss)
  * Patience-based early stopping

Public API
----------
train_lstm(X_train, y_train_scaled,
           X_test, y_test_scaled, y_test_raw, y_cols, scaler_y,
           *, hidden_size, n_layers, window_size, dropout_rate,
              epochs, lr, batch_size, patience,
              progress_callback, status_callback)
    → (LSTMWrapper, loss_history_dict)
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.models.wrappers import LSTMPredictor, LSTMWrapper


def train_lstm(
    X_train: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    y_test_raw,
    y_cols: List[str],
    scaler_y,
    *,
    hidden_size: int = 64,
    n_layers: int = 2,
    window_size: int = 1,
    dropout_rate: float = 0.2,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    patience: int = 20,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> tuple:
    """
    Train an LSTMPredictor.

    Returns
    -------
    LSTMWrapper  : model in eval mode, best-checkpoint weights restored
    loss_history : dict matching the format returned by trainer.py
    """
    # ---- Normalise array shapes ----
    y_train_scaled = np.array(y_train_scaled)
    y_test_scaled  = np.array(y_test_scaled)
    if y_train_scaled.ndim == 1:
        y_train_scaled = y_train_scaled.reshape(-1, 1)
    if y_test_scaled.ndim == 1:
        y_test_scaled = y_test_scaled.reshape(-1, 1)

    input_size  = X_train.shape[1]
    output_size = y_train_scaled.shape[1]

    # ---- Build tiled sequences (N, W, F) ----
    def _tile(X: np.ndarray) -> np.ndarray:
        return np.stack([X] * window_size, axis=1)

    X_train_seq = _tile(X_train)
    X_test_seq  = _tile(X_test)

    X_train_t = torch.tensor(X_train_seq,   dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_seq,    dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_scaled,  dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    # ---- Model, optimiser, scheduler ----
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        output_size=output_size,
        dropout=dropout_rate,
    )
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    scheduler  = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion  = nn.HuberLoss()

    # ---- Loss history ----
    epoch_pred_losses: List[float] = []
    val_pred_losses:   List[float] = []

    best_val_loss:  float = float("inf")
    best_state:     dict  = None
    no_improve:     int   = 0
    early_stopped:  bool  = False

    # ---- Training loop ----
    for epoch in range(epochs):
        model.train()
        b_pred = 0.0

        for batch_x, batch_y in loader:
            pred_y = model(batch_x)
            loss   = criterion(pred_y, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b_pred += loss.item()

        epoch_pred_losses.append(b_pred / len(loader))

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred      = model(X_test_t)
            avg_val_loss  = criterion(val_pred, y_test_t).item()
            val_pred_losses.append(avg_val_loss)

        # Best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1

        scheduler.step(avg_val_loss)

        if progress_callback:
            progress_callback(epoch + 1, epochs)

        # Patience early stop
        if patience > 0 and no_improve >= patience:
            early_stopped = True
            if status_callback:
                status_callback(
                    f"Early stopped at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)"
                )
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    loss_history: Dict = {
        "epoch_recon_losses": [],
        "epoch_pred_losses":  epoch_pred_losses,
        "val_recon_losses":   [],
        "val_pred_losses":    val_pred_losses,
        "actual_epochs":      len(epoch_pred_losses),
        "early_stopped":      early_stopped,
    }

    return LSTMWrapper(model, window_size), loss_history
