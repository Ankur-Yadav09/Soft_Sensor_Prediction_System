"""
src/training/trainer.py
========================
Training loop for the IndustrialDAE model.

Design goals
------------
* Zero Streamlit imports — this module is a pure Python function.
* Progress / status reporting is handled via optional callbacks so that
  the calling UI layer can wire in st.progress / st.empty without this
  module knowing anything about Streamlit.
* Supports both fixed-epoch training and auto-train mode.

Auto-train mode
---------------
Training continues until:
    avg_R² > AUTO_TRAIN_TARGET_R2  AND  avg_MAE <= best_MAE_so_far
checked every 10 epochs, up to AUTO_TRAIN_MAX_EPOCHS.
Patience early stopping applies in both modes as an additional guard.

Public API
----------
train_model(X_train, y_train, X_test, y_test_scaled,
            y_test_raw, y_cols, scaler_y,
            *, masking_ratio, epochs, lr, latent_dim, dropout_rate,
               weight_to_pred, batch_size, auto_train,
               patience, lr_patience, lr_factor,
               progress_callback, status_callback)
    → (model: IndustrialDAE, loss_history: dict)
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

from config.settings import AUTO_TRAIN_MAX_EPOCHS, AUTO_TRAIN_TARGET_R2
from src.models.architecture import IndustrialDAE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_masking(clean_x: torch.Tensor, masking_ratio: float) -> torch.Tensor:
    """
    Zero-mask a random fraction of features in each sample.

    This is the denoising autoencoder corruption step — the model learns to
    reconstruct clean inputs from partially-zeroed versions, forcing the
    encoder to learn robust latent representations.
    """
    mask = torch.rand(clean_x.shape) < masking_ratio
    noised = clean_x.clone()
    noised[mask] = 0.0
    return noised


# ---------------------------------------------------------------------------
# Public training function
# ---------------------------------------------------------------------------


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    y_test_raw,                   # pd.DataFrame of unscaled test targets
    y_cols: List[str],
    scaler_y,                     # fitted StandardScaler
    *,
    masking_ratio: float,
    epochs: int,
    lr: float,
    latent_dim: int,
    dropout_rate: float,
    weight_to_pred: float,
    batch_size: int,
    auto_train: bool = False,
    patience: int = 20,
    lr_patience: int = 10,
    lr_factor: float = 0.5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> tuple:
    """
    Train an IndustrialDAE from scratch.

    Parameters
    ----------
    X_train        : scaled training features  (N, input_dim)
    y_train        : scaled training targets   (N, output_dim)
    X_test         : scaled test features      (M, input_dim)
    y_test_scaled  : scaled test targets       (M, output_dim)
    y_test_raw     : unscaled test DataFrame   (M, output_dim)
    y_cols         : ordered list of target column names
    scaler_y       : fitted StandardScaler for Y
    masking_ratio  : fraction of features zeroed per sample per batch
    epochs         : number of epochs (ignored when auto_train=True)
    lr             : Adam initial learning rate
    latent_dim     : bottleneck dimension passed to IndustrialDAE
    dropout_rate   : dropout probability passed to IndustrialDAE
    weight_to_pred : scalar weight for the prediction loss term
    batch_size     : DataLoader batch size
    auto_train     : if True, stop early when R²/MAE criteria are met
    patience       : stop if val_pred_loss doesn't improve for this many epochs
                     (0 = disabled); applies to both fixed and auto-train modes
    lr_patience    : halve LR after this many epochs without val improvement
    lr_factor      : LR reduction multiplier (default 0.5 = halve)
    progress_callback : called each epoch as ``callback(current, total)``
    status_callback   : called with a status string (for st.empty().text())

    Returns
    -------
    model        : trained IndustrialDAE in eval mode, weights restored to
                   the best-validation-loss checkpoint
    loss_history : dict with keys
                   epoch_recon_losses, epoch_pred_losses,
                   val_recon_losses,   val_pred_losses,
                   actual_epochs,      early_stopped
    """
    # ---- Tensor conversion ----
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_scaled, dtype=torch.float32)

    # ---- DataLoader ----
    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    # ---- Model, optimizer, scheduler ----
    input_dim  = X_train_t.shape[1]
    output_dim = y_train_t.shape[1]

    model = IndustrialDAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    )
    optimizer       = optim.Adam(model.parameters(), lr=lr)
    scheduler       = ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor,
        patience=lr_patience, min_lr=1e-6,
    )
    criterion_recon = nn.MSELoss()
    criterion_pred  = nn.HuberLoss()

    # ---- Loss history ----
    epoch_recon_losses: List[float] = []
    epoch_pred_losses:  List[float] = []
    val_recon_losses:   List[float] = []
    val_pred_losses:    List[float] = []

    max_epochs = AUTO_TRAIN_MAX_EPOCHS if auto_train else epochs
    best_r2    = -float("inf")
    best_mae   = float("inf")

    # Best-checkpoint state
    best_val_pred_loss: float = float("inf")
    best_state_dict = None
    no_improve_count: int = 0
    early_stopped: bool = False

    # ---- Training loop ----
    for epoch in range(max_epochs):

        # --- Training pass ---
        model.train()
        b_recon = b_pred = 0.0

        for batch_x, batch_y in loader:
            noised_x = _apply_masking(batch_x, masking_ratio)
            recon_x, pred_y = model(noised_x)

            loss_recon = criterion_recon(recon_x, batch_x)
            loss_pred  = criterion_pred(pred_y, batch_y)
            total_loss = loss_recon + weight_to_pred * loss_pred

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            b_recon += loss_recon.item()
            b_pred  += loss_pred.item()

        n_batches = len(loader)
        epoch_recon_losses.append(b_recon / n_batches)
        epoch_pred_losses.append(b_pred  / n_batches)

        # --- Validation pass (clean data, no masking) ---
        model.eval()
        with torch.no_grad():
            val_recon, val_pred = model(X_test_t)
            avg_val_recon_loss = criterion_recon(val_recon, X_test_t).item()
            avg_val_pred_loss  = criterion_pred(val_pred, y_test_t).item()
            val_recon_losses.append(avg_val_recon_loss)
            val_pred_losses.append(avg_val_pred_loss)

        # --- Best-model checkpoint ---
        if avg_val_pred_loss < best_val_pred_loss:
            best_val_pred_loss = avg_val_pred_loss
            best_state_dict    = copy.deepcopy(model.state_dict())
            no_improve_count   = 0
        else:
            no_improve_count += 1

        # --- LR scheduling ---
        scheduler.step(avg_val_pred_loss)

        # --- Progress reporting / auto-stop check ---
        if auto_train:
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    preds = scaler_y.inverse_transform(val_pred.numpy())

                r2_vals  = [r2_score(y_test_raw[c], preds[:, i])
                            for i, c in enumerate(y_cols)]
                mae_vals = [mean_absolute_error(y_test_raw[c], preds[:, i])
                            for i, c in enumerate(y_cols)]
                avg_r2  = float(np.mean(r2_vals))
                avg_mae = float(np.mean(mae_vals))

                if status_callback:
                    status_callback(
                        f"Auto-Training… Epoch {epoch + 1} "
                        f"| Avg R²: {avg_r2:.4f} "
                        f"| Avg MAE: {avg_mae:.4f}"
                    )

                if avg_r2 > AUTO_TRAIN_TARGET_R2 and avg_mae <= best_mae:
                    if status_callback:
                        status_callback(
                            f"Target reached! Stopped at Epoch {epoch + 1} — "
                            f"Avg R² = {avg_r2:.4f}, Avg MAE = {avg_mae:.4f}"
                        )
                    break

                best_r2  = max(best_r2, avg_r2)
                best_mae = min(best_mae, avg_mae)

        else:
            if progress_callback:
                progress_callback(epoch + 1, epochs)

        # --- Patience early stopping (both modes) ---
        if patience > 0 and no_improve_count >= patience:
            early_stopped = True
            if status_callback:
                status_callback(
                    f"Early stopped at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs — best model restored)"
                )
            break

    # ---- Restore best weights ----
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()

    loss_history: Dict = {
        "epoch_recon_losses": epoch_recon_losses,
        "epoch_pred_losses":  epoch_pred_losses,
        "val_recon_losses":   val_recon_losses,
        "val_pred_losses":    val_pred_losses,
        "actual_epochs":      len(epoch_pred_losses),
        "early_stopped":      early_stopped,
    }

    return model, loss_history
