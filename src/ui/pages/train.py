"""
src/ui/pages/train.py
======================
Renders the "Train Model" page.

Supported models
----------------
DAE           — Industrial Denoising Autoencoder (PyTorch)
Random Forest — sklearn RandomForestRegressor (multi-output)
XGBoost       — xgboost XGBRegressor via MultiOutputRegressor
LightGBM      — lightgbm LGBMRegressor via MultiOutputRegressor
LSTM          — PyTorch LSTMPredictor with tiled-window sequences

Features
--------
  - Load a previously saved model from disk (expander)
  - Model type selector with per-model hyperparameter widgets
  - Train button dispatches to the appropriate training function
  - Best-model checkpointing + LR scheduling + patience (DAE & LSTM)
  - Auto-train mode (DAE only)
  - Post-training metrics table and loss curves
  - Auto-save to disk and append to training history
"""
from __future__ import annotations

import datetime
import os

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_LATENT_DIM,
    DEFAULT_LR,
    DEFAULT_MASKING_RATIO,
    DEFAULT_WEIGHT_TO_PRED,
)
from src.data.database import save_model_to_registry
from src.evaluation.metrics import compute_metrics
from src.models.wrappers import DAEWrapper
from src.persistence.model_store import (
    list_saved_models,
    load_model_from_disk,
    save_model_to_disk,
)
from src.training.train_lstm import train_lstm
from src.training.train_sklearn import train_sklearn_model
from src.training.trainer import train_model
from src.ui.components import render_loss_curves

_MODEL_OPTIONS = ["DAE", "Random Forest", "XGBoost", "LightGBM", "LSTM"]


def render() -> None:
    st.title("Train Model")

    # ------------------------------------------------------------------ #
    # Load a previously saved model
    # ------------------------------------------------------------------ #
    saved_models = list_saved_models()
    if saved_models:
        with st.expander("📂 Load a Previously Saved Model", expanded=False):
            model_meta_df = (
                pd.DataFrame(saved_models)[
                    [c for c in
                     ["name", "model_type", "saved_at", "input_dim", "output_dim"]
                     if c in pd.DataFrame(saved_models).columns]
                ].rename(columns={
                    "name":       "Model Name",
                    "model_type": "Type",
                    "saved_at":   "Saved At",
                    "input_dim":  "Input Features",
                    "output_dim": "Output Targets",
                })
            )
            st.dataframe(model_meta_df, use_container_width=True)

            sel_model_name = st.selectbox(
                "Select Model to Load", [m["name"] for m in saved_models]
            )
            if st.button("Load Selected Model"):
                (
                    loaded_wrapper,
                    loaded_sx, loaded_sy,
                    loaded_x,  loaded_y,
                ) = load_model_from_disk(sel_model_name)

                st.session_state.model         = loaded_wrapper
                st.session_state.scaler_x      = loaded_sx
                st.session_state.scaler_y      = loaded_sy
                st.session_state.x_cols        = loaded_x
                st.session_state.y_cols        = loaded_y
                st.session_state.model_trained = True

                st.success(
                    f"✅ Model **{sel_model_name}** "
                    f"({loaded_wrapper.model_type}) loaded! "
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
    # Model type selector
    # ------------------------------------------------------------------ #
    st.subheader("Select Model Architecture")
    model_type = st.radio(
        "Model",
        _MODEL_OPTIONS,
        horizontal=True,
        help="DAE & LSTM are deep-learning models with per-epoch loss curves. "
             "RF / XGBoost / LightGBM are tree ensembles that train instantly.",
    )

    st.markdown("---")
    st.subheader("Hyperparameters")

    # ------------------------------------------------------------------ #
    # Per-model hyperparameter widgets
    # ------------------------------------------------------------------ #

    # --- DAE -------------------------------------------------------
    if model_type == "DAE":
        col1, col2 = st.columns(2)
        with col1:
            masking_ratio = st.slider(
                "Masking Ratio (Corruption)", 0.0, 0.5, DEFAULT_MASKING_RATIO
            )
            epochs = st.number_input("Epochs", 10, 1000, DEFAULT_EPOCHS)
            lr = st.number_input(
                "Learning Rate", 0.0001, 0.1, DEFAULT_LR, format="%.4f"
            )
            auto_train = st.checkbox(
                "Auto-Train (Until R² > 0.85 & MAE lower)", value=False
            )
        with col2:
            latent_dim = st.slider(
                "Latent Dimension",
                2, max(2, len(st.session_state.x_cols)), DEFAULT_LATENT_DIM,
            )
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, DEFAULT_DROPOUT_RATE)
            weight_to_pred = st.number_input(
                "Weight to Predictor Loss", 0.1, 10.0, DEFAULT_WEIGHT_TO_PRED
            )
            batch_size = st.selectbox(
                "Batch Size", [16, 32, 64, 128, 256],
                index=[16, 32, 64, 128, 256].index(DEFAULT_BATCH_SIZE),
            )
            patience = st.slider(
                "Early Stop Patience (epochs)", 0, 100,
                DEFAULT_EARLY_STOP_PATIENCE, step=5,
                help="Stop if validation loss doesn't improve. 0 = disabled.",
            )

    # --- Random Forest -----------------------------------------------
    elif model_type == "Random Forest":
        col1, col2 = st.columns(2)
        with col1:
            rf_n_estimators = st.slider("N Estimators", 50, 1000, 300, step=50)
            rf_max_depth    = st.slider(
                "Max Depth (0 = unlimited)", 0, 50, 0,
                help="Depth of each tree. 0 means no limit.",
            )
        with col2:
            rf_min_samples = st.slider("Min Samples Split", 2, 20, 2)
            rf_max_features = st.selectbox(
                "Max Features per Split", ["sqrt", "log2", "None"], index=0
            )
        st.caption(
            "Random Forest trains quickly and natively supports multi-output regression."
        )

    # --- XGBoost -----------------------------------------------------
    elif model_type == "XGBoost":
        col1, col2 = st.columns(2)
        with col1:
            xgb_n_estimators = st.slider("N Estimators", 50, 500, 300, step=50)
            xgb_max_depth    = st.slider("Max Depth", 3, 15, 6)
            xgb_lr           = st.number_input(
                "Learning Rate", 0.01, 0.3, 0.1, format="%.3f"
            )
        with col2:
            xgb_subsample    = st.slider("Subsample", 0.5, 1.0, 0.8, step=0.05)
            xgb_colsample    = st.slider("ColSample by Tree", 0.5, 1.0, 0.8, step=0.05)
        st.caption(
            "XGBoost is wrapped in MultiOutputRegressor — one model per target column."
        )

    # --- LightGBM ----------------------------------------------------
    elif model_type == "LightGBM":
        col1, col2 = st.columns(2)
        with col1:
            lgbm_n_estimators = st.slider("N Estimators", 50, 500, 300, step=50)
            lgbm_max_depth    = st.slider("Max Depth", 3, 15, 6)
            lgbm_lr           = st.number_input(
                "Learning Rate", 0.01, 0.3, 0.05, format="%.3f"
            )
        with col2:
            lgbm_num_leaves = st.slider("Num Leaves", 15, 127, 31, step=4)
            lgbm_subsample  = st.slider("Subsample", 0.5, 1.0, 0.8, step=0.05)
        st.caption(
            "LightGBM is wrapped in MultiOutputRegressor — one model per target column."
        )

    # --- LSTM --------------------------------------------------------
    elif model_type == "LSTM":
        col1, col2 = st.columns(2)
        with col1:
            lstm_hidden = st.selectbox(
                "Hidden Size", [32, 64, 128, 256], index=1
            )
            lstm_layers  = st.slider("LSTM Layers", 1, 4, 2)
            lstm_window  = st.slider(
                "Window Size", 1, 30, 1,
                help="How many timesteps to tile per sample. "
                     "1 = single-step (recommended unless data is time-ordered).",
            )
            lstm_dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
        with col2:
            lstm_epochs  = st.number_input("Epochs", 10, 500, 100)
            lstm_lr      = st.number_input(
                "Learning Rate", 0.0001, 0.01, 0.001, format="%.4f"
            )
            lstm_bs      = st.selectbox(
                "Batch Size", [16, 32, 64, 128], index=2
            )
            lstm_patience = st.slider(
                "Early Stop Patience", 0, 100, DEFAULT_EARLY_STOP_PATIENCE, step=5
            )

    # ------------------------------------------------------------------ #
    # Train button
    # ------------------------------------------------------------------ #
    if st.button("🚀 Train", type="primary"):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def _progress(current: int, total: int) -> None:
            progress_bar.progress(current / total)

        def _status(msg: str) -> None:
            status_text.text(msg)

        # ---- Dispatch ----
        with st.spinner("Training…"):

            if model_type == "DAE":
                raw_model, loss_history = train_model(
                    X_train        = st.session_state.X_train,
                    y_train        = st.session_state.y_train,
                    X_test         = st.session_state.X_test,
                    y_test_scaled  = st.session_state.y_test,
                    y_test_raw     = st.session_state.y_test_raw,
                    y_cols         = st.session_state.y_cols,
                    scaler_y       = st.session_state.scaler_y,
                    masking_ratio  = masking_ratio,
                    epochs         = epochs,
                    lr             = lr,
                    latent_dim     = latent_dim,
                    dropout_rate   = dropout_rate,
                    weight_to_pred = weight_to_pred,
                    batch_size     = batch_size,
                    auto_train     = auto_train,
                    patience       = patience,
                    progress_callback = _progress,
                    status_callback   = _status,
                )
                wrapper = DAEWrapper(raw_model)

            elif model_type == "LSTM":
                wrapper, loss_history = train_lstm(
                    X_train       = st.session_state.X_train,
                    y_train_scaled = st.session_state.y_train,
                    X_test         = st.session_state.X_test,
                    y_test_scaled  = st.session_state.y_test,
                    y_test_raw     = st.session_state.y_test_raw,
                    y_cols         = st.session_state.y_cols,
                    scaler_y       = st.session_state.scaler_y,
                    hidden_size    = lstm_hidden,
                    n_layers       = lstm_layers,
                    window_size    = lstm_window,
                    dropout_rate   = lstm_dropout,
                    epochs         = lstm_epochs,
                    lr             = lstm_lr,
                    batch_size     = lstm_bs,
                    patience       = lstm_patience,
                    progress_callback = _progress,
                    status_callback   = _status,
                )

            elif model_type == "Random Forest":
                hparams = {
                    "n_estimators":     rf_n_estimators,
                    "max_depth":        None if rf_max_depth == 0 else rf_max_depth,
                    "min_samples_split": rf_min_samples,
                    "max_features":     None if rf_max_features == "None" else rf_max_features,
                    "n_jobs":           -1,
                    "random_state":     42,
                }
                wrapper, loss_history = train_sklearn_model(
                    X_train        = st.session_state.X_train,
                    y_train_scaled = st.session_state.y_train,
                    X_test         = st.session_state.X_test,
                    y_test_scaled  = st.session_state.y_test,
                    y_test_raw     = st.session_state.y_test_raw,
                    y_cols         = st.session_state.y_cols,
                    scaler_y       = st.session_state.scaler_y,
                    model_type     = model_type,
                    **hparams,
                )

            elif model_type == "XGBoost":
                hparams = {
                    "n_estimators":   xgb_n_estimators,
                    "max_depth":      xgb_max_depth,
                    "learning_rate":  xgb_lr,
                    "subsample":      xgb_subsample,
                    "colsample_bytree": xgb_colsample,
                    "n_jobs":         -1,
                    "random_state":   42,
                    "verbosity":      0,
                }
                wrapper, loss_history = train_sklearn_model(
                    X_train        = st.session_state.X_train,
                    y_train_scaled = st.session_state.y_train,
                    X_test         = st.session_state.X_test,
                    y_test_scaled  = st.session_state.y_test,
                    y_test_raw     = st.session_state.y_test_raw,
                    y_cols         = st.session_state.y_cols,
                    scaler_y       = st.session_state.scaler_y,
                    model_type     = model_type,
                    **hparams,
                )

            elif model_type == "LightGBM":
                hparams = {
                    "n_estimators":  lgbm_n_estimators,
                    "max_depth":     lgbm_max_depth,
                    "learning_rate": lgbm_lr,
                    "num_leaves":    lgbm_num_leaves,
                    "subsample":     lgbm_subsample,
                    "n_jobs":        -1,
                    "random_state":  42,
                    "verbose":       -1,
                }
                wrapper, loss_history = train_sklearn_model(
                    X_train        = st.session_state.X_train,
                    y_train_scaled = st.session_state.y_train,
                    X_test         = st.session_state.X_test,
                    y_test_scaled  = st.session_state.y_test,
                    y_test_raw     = st.session_state.y_test_raw,
                    y_cols         = st.session_state.y_cols,
                    scaler_y       = st.session_state.scaler_y,
                    model_type     = model_type,
                    **hparams,
                )

        progress_bar.empty()

        # ---- Status messages ----
        if model_type == "DAE":
            if not auto_train and not loss_history.get("early_stopped"):
                status_text.text("Training Complete!")
            if loss_history.get("early_stopped"):
                st.info(
                    f"Training stopped early at epoch "
                    f"**{loss_history['actual_epochs']}** "
                    f"(patience = {patience}). Best-checkpoint weights restored."
                )
        elif model_type == "LSTM":
            if not loss_history.get("early_stopped"):
                status_text.text("LSTM Training Complete!")
            else:
                st.info(
                    f"Training stopped early at epoch "
                    f"**{loss_history['actual_epochs']}** "
                    f"(patience = {lstm_patience}). Best-checkpoint weights restored."
                )
        else:
            status_text.text(f"{model_type} training complete!")

        # ---- Store wrapper in session state ----
        st.session_state.model         = wrapper
        st.session_state.model_trained = True

        # ---- Final evaluation ----
        preds_test = st.session_state.scaler_y.inverse_transform(
            wrapper.predict_scaled(st.session_state.X_test)
        )
        metrics_df = compute_metrics(
            st.session_state.y_test_raw, preds_test, st.session_state.y_cols
        )
        avg_rmse = float(metrics_df["RMSE"].mean())
        avg_r2   = float(metrics_df["R2 Score"].mean())
        avg_mae  = float(metrics_df["MAE"].mean())

        # ---- Append to training history ----
        run_id = len(st.session_state.history) + 1
        st.session_state.history.append(
            {
                "Run ID":        run_id,
                "Model Type":    model_type,
                "Epochs":        loss_history["actual_epochs"],
                "Avg Test RMSE": avg_rmse,
                "Avg Test R2":   avg_r2,
                "Avg Test MAE":  avg_mae,
                "Model":         wrapper,
            }
        )

        # ---- Auto-save to disk ----
        model_name = (
            f"{model_type.replace(' ', '_')}_Run{run_id}_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        save_model_to_disk(
            wrapper,
            st.session_state.scaler_x,
            st.session_state.scaler_y,
            st.session_state.x_cols,
            st.session_state.y_cols,
            model_name,
        )

        dataset_name = next(iter(st.session_state.data_history), "Unknown")
        try:
            save_model_to_registry(
                model_name=model_name,
                algorithm=model_type,
                dataset_name=dataset_name,
                x_cols=st.session_state.x_cols,
                y_cols=st.session_state.y_cols,
                avg_r2=avg_r2,
                avg_rmse=avg_rmse,
                avg_mae=avg_mae,
                file_path=os.path.join("saved_models", model_name),
            )
        except Exception:
            pass  # registry write failure must not block training success

        st.success(
            f"✅ **{model_type}** trained and saved as **{model_name}** | "
            f"Avg R² = `{avg_r2:.4f}` | Avg RMSE = `{avg_rmse:.4f}`"
        )

        # ---- Post-training metrics ----
        st.subheader("Training Post-Evaluation Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        # ---- Loss curves (deep-learning models only) ----
        if model_type in ("DAE", "LSTM") and loss_history.get("epoch_pred_losses"):
            render_loss_curves(
                loss_history["epoch_recon_losses"],
                loss_history["epoch_pred_losses"],
                loss_history["val_recon_losses"],
                loss_history["val_pred_losses"],
                model_type=model_type,
            )
