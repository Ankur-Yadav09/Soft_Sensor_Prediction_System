"""
src/ui/pages/train.py
======================
Renders the "Train Model" page.

Features:
  - Load a previously saved model from disk (expander)
  - Configure all hyperparameters via Streamlit widgets
  - Train a new IndustrialDAE via src.training.trainer.train_model
  - Auto-train mode (early stopping on R² / MAE criteria)
  - Auto-save the trained model to disk
  - Post-training metrics table
  - Loss curves (reconstruction + prediction)
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch

from config.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_LATENT_DIM,
    DEFAULT_LR,
    DEFAULT_MASKING_RATIO,
    DEFAULT_WEIGHT_TO_PRED,
)
from src.evaluation.metrics import compute_metrics
from src.persistence.model_store import (
    list_saved_models,
    load_model_from_disk,
    save_model_to_disk,
)
from src.training.trainer import train_model
from src.ui.components import render_loss_curves


def render() -> None:
    st.title("Train Model (Industrial DAE)")

    # ------------------------------------------------------------------ #
    # Load a previously saved model
    # ------------------------------------------------------------------ #
    saved_models = list_saved_models()
    if saved_models:
        with st.expander("📂 Load a Previously Saved Model", expanded=False):
            model_meta_df = (
                pd.DataFrame(saved_models)[
                    ["name", "saved_at", "input_dim", "output_dim"]
                ].rename(
                    columns={
                        "name":       "Model Name",
                        "saved_at":   "Saved At",
                        "input_dim":  "Input Features",
                        "output_dim": "Output Targets",
                    }
                )
            )
            st.dataframe(model_meta_df, use_container_width=True)

            sel_model_name = st.selectbox(
                "Select Model to Load", [m["name"] for m in saved_models]
            )
            if st.button("Load Selected Model"):
                (
                    loaded_model,
                    loaded_sx, loaded_sy,
                    loaded_x,  loaded_y,
                ) = load_model_from_disk(sel_model_name)

                st.session_state.model         = loaded_model
                st.session_state.scaler_x      = loaded_sx
                st.session_state.scaler_y      = loaded_sy
                st.session_state.x_cols        = loaded_x
                st.session_state.y_cols        = loaded_y
                st.session_state.model_trained = True

                st.success(
                    f"✅ Model **{sel_model_name}** loaded! "
                    "You can now use Predict and What-If tabs."
                )
                st.rerun()

        st.markdown("---")

    # ------------------------------------------------------------------ #
    # Guard: need preprocessed data
    # ------------------------------------------------------------------ #
    if st.session_state.X_train is None:
        st.warning("Please preprocess data first in the 'Preprocess' tab.")
        return

    # ------------------------------------------------------------------ #
    # Hyperparameter widgets
    # ------------------------------------------------------------------ #
    st.subheader("Hyperparameters")
    col1, col2 = st.columns(2)

    with col1:
        masking_ratio = st.slider(
            "Masking Ratio (Corruption)", 0.0, 0.5, DEFAULT_MASKING_RATIO
        )
        epochs = st.number_input("Epochs", 10, 1000, DEFAULT_EPOCHS)
        lr = st.number_input(
            "Learning Rate",
            0.0001, 0.1, DEFAULT_LR,
            format="%.4f",
        )
        auto_train = st.checkbox(
            "Auto-Train (Until R² > 0.85 & MAE lower)", value=False
        )

    with col2:
        latent_dim = st.slider(
            "Latent Dimension",
            2,
            max(2, len(st.session_state.x_cols)),
            DEFAULT_LATENT_DIM,
        )
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, DEFAULT_DROPOUT_RATE)
        weight_to_pred = st.number_input(
            "Weight to Predictor Loss",
            0.1, 10.0, DEFAULT_WEIGHT_TO_PRED,
        )
        batch_size = st.selectbox(
            "Batch Size", [16, 32, 64, 128, 256],
            index=[16, 32, 64, 128, 256].index(DEFAULT_BATCH_SIZE),
        )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    if st.button("Train"):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def _progress(current: int, total: int) -> None:
            progress_bar.progress(current / total)

        def _status(msg: str) -> None:
            status_text.text(msg)

        trained_model, loss_history = train_model(
            X_train       = st.session_state.X_train,
            y_train       = st.session_state.y_train,
            X_test        = st.session_state.X_test,
            y_test_scaled = st.session_state.y_test,
            y_test_raw    = st.session_state.y_test_raw,
            y_cols        = st.session_state.y_cols,
            scaler_y      = st.session_state.scaler_y,
            masking_ratio  = masking_ratio,
            epochs         = epochs,
            lr             = lr,
            latent_dim     = latent_dim,
            dropout_rate   = dropout_rate,
            weight_to_pred = weight_to_pred,
            batch_size     = batch_size,
            auto_train     = auto_train,
            progress_callback = _progress,
            status_callback   = _status,
        )

        if not auto_train:
            status_text.text("Training Complete!")

        # ---- Store trained model in session state ----
        st.session_state.model         = trained_model
        st.session_state.model_trained = True

        # ---- Final evaluation on test set ----
        X_test_t = torch.tensor(
            st.session_state.X_test, dtype=torch.float32
        )
        trained_model.eval()
        with torch.no_grad():
            _, val_pred = trained_model(X_test_t)
            preds_test = st.session_state.scaler_y.inverse_transform(
                val_pred.numpy()
            )

        metrics_df = compute_metrics(
            st.session_state.y_test_raw, preds_test, st.session_state.y_cols
        )
        avg_rmse = float(metrics_df["RMSE"].mean())

        # ---- Append to training history ----
        run_id = len(st.session_state.history) + 1
        st.session_state.history.append(
            {
                "Run ID":        run_id,
                "Masking":       masking_ratio,
                "Latent Dim":    latent_dim,
                "Epochs":        loss_history["actual_epochs"],
                "Avg Test RMSE": avg_rmse,
                "Model":         trained_model,
            }
        )

        # ---- Auto-save to disk ----
        model_name = (
            f"DAE_Run{run_id}_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        save_model_to_disk(
            trained_model,
            st.session_state.scaler_x,
            st.session_state.scaler_y,
            st.session_state.x_cols,
            st.session_state.y_cols,
            model_name,
        )

        st.success(
            f"✅ Model trained, saved as **{model_name}**, and added to History! "
            f"(Epochs: {loss_history['actual_epochs']})"
        )

        # ---- Post-training metrics ----
        st.subheader("Training Post-Evaluation Metrics")
        st.dataframe(metrics_df)

        # ---- Loss curves ----
        render_loss_curves(
            loss_history["epoch_recon_losses"],
            loss_history["epoch_pred_losses"],
            loss_history["val_recon_losses"],
            loss_history["val_pred_losses"],
        )
