"""
src/ui/pages/overview.py
========================
Renders the "Overview" page of the Industrial DAE Dashboard.

Displays:
  - 4-column system status strip
  - Live model performance KPI cards (if a model is trained)
  - Feature lists (X inputs / Y targets)
  - Dataset and saved-model inventory tables
  - Step-by-step workflow guide
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data.database import list_datasets_from_db
from src.evaluation.metrics import compute_metrics, grade_r2
from src.persistence.model_store import list_saved_models
from src.ui.components import (
    render_kpi_cards,
    render_system_status,
    render_workflow_guide,
)


def render() -> None:
    st.title("🏭 Industrial DAE — Multi X-Y Dashboard")
    st.caption(
        "End-to-end Denoising Autoencoder for Sensor Reconstruction & KPI Prediction"
    )

    # ------------------------------------------------------------------ #
    # System Status
    # ------------------------------------------------------------------ #
    st.markdown("### 🔄 System Status")
    saved_models_list = list_saved_models()
    render_system_status(
        df=st.session_state.df,
        x_cols=st.session_state.x_cols,
        y_cols=st.session_state.y_cols,
        model_trained=st.session_state.model_trained,
        saved_count=len(saved_models_list),
    )
    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Current Model Performance
    # ------------------------------------------------------------------ #
    if st.session_state.model_trained and st.session_state.X_test is not None:
        st.markdown("### 📊 Current Model Performance")

        model    = st.session_state.model
        scaler_y = st.session_state.scaler_y
        y_test   = st.session_state.y_test_raw
        y_cols   = st.session_state.y_cols

        X_test_t = torch.tensor(st.session_state.X_test, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            _, preds_scaled = model(X_test_t)
            preds = scaler_y.inverse_transform(preds_scaled.numpy())

        metrics_df = compute_metrics(y_test, preds, y_cols)
        render_kpi_cards(metrics_df, y_cols)

        avg_r2          = float(metrics_df["R2 Score"].mean())
        grade, emoji    = grade_r2(avg_r2)
        st.markdown(
            f"**Overall Average R²:** `{avg_r2:.4f}` — **{grade} {emoji}**"
        )
        st.markdown("---")

        # Feature info columns
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("**Input Features (X)**")
            for c in st.session_state.x_cols:
                st.markdown(f"- `{c}`")
        with col_info2:
            st.markdown("**Target Features (Y)**")
            for c in y_cols:
                st.markdown(f"- `{c}`")

    else:
        st.info(
            "Upload data, preprocess, and train a model to see "
            "performance metrics here."
        )

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Dataset & Saved-Model Inventory
    # ------------------------------------------------------------------ #
    db_col1, db_col2 = st.columns(2)

    with db_col1:
        st.markdown("### 📦 Datasets in Database")
        db_ds = list_datasets_from_db()
        if db_ds:
            st.dataframe(
                pd.DataFrame(db_ds, columns=["Name", "Uploaded", "Rows", "Cols"]),
                use_container_width=True,
            )
        else:
            st.caption("No datasets stored yet.")

    with db_col2:
        st.markdown("### 💾 Saved Models")
        if saved_models_list:
            model_df = (
                pd.DataFrame(saved_models_list)[
                    ["name", "saved_at", "input_dim", "output_dim"]
                ].rename(
                    columns={
                        "name":       "Name",
                        "saved_at":   "Saved At",
                        "input_dim":  "X Features",
                        "output_dim": "Y Targets",
                    }
                )
            )
            st.dataframe(model_df, use_container_width=True)
        else:
            st.caption("No models saved yet.")

    st.markdown("---")
    render_workflow_guide()
