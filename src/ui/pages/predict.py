"""
src/ui/pages/predict.py
========================
Renders the "Predict & Evaluate" page.

Displays:
  - Test-set metrics table (RMSE, MAE, R², MAPE)
  - Traffic-light KPI metric cards
  - Actual vs Predicted line charts (first 100 samples)
  - Scatter plots with 45° ideal reference line
  - Residual error distribution histograms
  - CSV download of predictions and errors
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.evaluation.metrics import compute_metrics
from src.ui.components import (
    render_actual_vs_predicted_lines,
    render_kpi_cards,
    render_residual_histograms,
    render_scatter_plots,
)


def render() -> None:
    st.title("Predict & Evaluate")

    # ------------------------------------------------------------------ #
    # Guard: model must be trained
    # ------------------------------------------------------------------ #
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Train Model' tab.")
        return

    # ------------------------------------------------------------------ #
    # Run inference on the test set
    # ------------------------------------------------------------------ #
    model    = st.session_state.model
    scaler_y = st.session_state.scaler_y
    y_test   = st.session_state.y_test_raw
    y_cols   = st.session_state.y_cols

    X_test_t = torch.tensor(st.session_state.X_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        _, val_pred = model(X_test_t)
        preds_test = scaler_y.inverse_transform(val_pred.numpy())

    # ------------------------------------------------------------------ #
    # Metrics table
    # ------------------------------------------------------------------ #
    st.subheader("Test Set Metrics")
    metrics_df = compute_metrics(y_test, preds_test, y_cols)
    st.dataframe(metrics_df, use_container_width=True)

    # ------------------------------------------------------------------ #
    # KPI summary cards
    # ------------------------------------------------------------------ #
    st.subheader("📊 Model Performance Summary")
    render_kpi_cards(metrics_df, y_cols)

    # ------------------------------------------------------------------ #
    # Actual vs Predicted line charts
    # ------------------------------------------------------------------ #
    st.subheader("📈 Actual vs Predicted (All Y Features)")
    render_actual_vs_predicted_lines(y_test, preds_test, y_cols, metrics_df)

    # ------------------------------------------------------------------ #
    # Scatter plots
    # ------------------------------------------------------------------ #
    st.subheader("🎯 Scatter Plot: Actual vs Predicted")
    render_scatter_plots(y_test, preds_test, y_cols, metrics_df)

    # ------------------------------------------------------------------ #
    # Residual histograms
    # ------------------------------------------------------------------ #
    st.subheader("📉 Residual Analysis (Error Distribution)")
    render_residual_histograms(y_test, preds_test, y_cols)

    # ------------------------------------------------------------------ #
    # CSV export
    # ------------------------------------------------------------------ #
    st.subheader("📥 Export Predictions")
    export_df = y_test.copy().reset_index(drop=True)
    for i, col in enumerate(y_cols):
        export_df[f"Predicted_{col}"] = preds_test[:, i]
        export_df[f"Error_{col}"]     = y_test[col].values - preds_test[:, i]

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Full Predictions with Errors (CSV)",
        data=csv_bytes,
        file_name="Predictions_with_Errors.csv",
        mime="text/csv",
    )
