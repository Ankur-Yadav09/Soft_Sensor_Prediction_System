"""
src/ui/components.py
=====================
Reusable Streamlit UI components shared across multiple pages.

Every function here renders something to the Streamlit app.  These are
stateless widgets — they only read data passed as arguments and never
write to ``st.session_state`` directly.

Components
----------
render_system_status    — 4-column status banner (Overview)
render_kpi_cards        — per-target R²/MAE metric cards (Overview, Predict)
render_loss_curves      — dual matplotlib loss plots (Train)
render_actual_vs_predicted_lines  — Plotly line charts  (Predict)
render_scatter_plots    — Plotly scatter with 45° line  (Predict)
render_residual_histograms        — Plotly histograms   (Predict)
render_workflow_guide   — markdown workflow table       (Overview)
"""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluation.metrics import r2_emoji


# ---------------------------------------------------------------------------
# System status strip
# ---------------------------------------------------------------------------


def render_system_status(
    df: pd.DataFrame | None,
    x_cols: List[str],
    y_cols: List[str],
    model_trained: bool,
    saved_count: int,
) -> None:
    """
    Four-column status banner showing data / preprocess / model / disk status.
    """
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        if df is not None:
            st.success(
                f"✅ Data Loaded\n\n**{df.shape[0]}** rows × **{df.shape[1]}** cols"
            )
        else:
            st.warning("⚠️ No Data\n\nUpload data to begin")

    with s2:
        if len(x_cols) > 0:
            st.success(
                f"✅ Preprocessed\n\n**{len(x_cols)}** X  |  **{len(y_cols)}** Y"
            )
        else:
            st.warning("⚠️ Not Preprocessed")

    with s3:
        if model_trained:
            st.success("✅ Model Trained\n\nReady for Prediction")
        else:
            st.warning("⚠️ No Model\n\nTrain a model first")

    with s4:
        st.info(f"💾 Saved Models\n\n**{saved_count}** model(s) on disk")


# ---------------------------------------------------------------------------
# KPI metric cards
# ---------------------------------------------------------------------------


def render_kpi_cards(
    metrics_df: pd.DataFrame,
    y_cols: List[str],
) -> None:
    """
    Traffic-light metric cards — one card per target variable.

    Each card shows the R² value and MAE as delta.
    """
    cols = st.columns(len(y_cols))
    for i, col in enumerate(y_cols):
        r2_val  = float(metrics_df.loc[col, "R2 Score"])
        mae_val = float(metrics_df.loc[col, "MAE"])
        with cols[i]:
            st.metric(
                label=f"{r2_emoji(r2_val)} {col}",
                value=f"R² = {r2_val:.4f}",
                delta=f"MAE = {mae_val:.4f}",
            )


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------


def render_loss_curves(
    epoch_recon_losses: List[float],
    epoch_pred_losses:  List[float],
    val_recon_losses:   List[float],
    val_pred_losses:    List[float],
    model_type: str = "DAE",
) -> None:
    """
    Loss curve plots. DAE shows two charts (reconstruction + predictor).
    LSTM shows one chart (training loss vs validation loss).
    """
    if model_type == "LSTM":
        st.subheader("LSTM Training Loss")
        fig, ax = plt.subplots()
        ax.plot(epoch_pred_losses, color="orange", label="Train Loss")
        ax.plot(val_pred_losses,   color="red",    label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("DAE Reconstruction Loss (MSE)")
        fig, ax = plt.subplots()
        ax.plot(epoch_recon_losses, color="blue",  label="Train Loss")
        ax.plot(val_recon_losses,   color="cyan",  label="Validation Loss")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Predictor Loss (Huber)")
        fig, ax = plt.subplots()
        ax.plot(epoch_pred_losses, color="orange", label="Train Loss")
        ax.plot(val_pred_losses,   color="red",    label="Validation Loss")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Actual vs Predicted line charts
# ---------------------------------------------------------------------------


def render_actual_vs_predicted_lines(
    y_test: pd.DataFrame,
    preds_test: np.ndarray,
    y_cols: List[str],
    metrics_df: pd.DataFrame,
    n_samples: int = 100,
) -> None:
    """
    One Plotly line chart per target variable showing the first n_samples.
    """
    pts = min(n_samples, len(y_test))

    for i, col in enumerate(y_cols):
        r2_val = float(metrics_df.loc[col, "R2 Score"])
        chart_df = pd.DataFrame(
            {
                "Sample Index": range(pts),
                "Actual":       y_test[col].values[:pts],
                "Predicted":    preds_test[:pts, i],
            }
        )
        melted = chart_df.melt(
            id_vars=["Sample Index"],
            value_vars=["Actual", "Predicted"],
            var_name="Type",
            value_name="Value",
        )
        fig = px.line(
            melted,
            x="Sample Index",
            y="Value",
            color="Type",
            title=f"{col}  |  R² = {r2_val:.4f}",
        )
        fig.update_layout(yaxis=dict(autorange=True))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Scatter plots with 45° reference line
# ---------------------------------------------------------------------------


def render_scatter_plots(
    y_test: pd.DataFrame,
    preds_test: np.ndarray,
    y_cols: List[str],
    metrics_df: pd.DataFrame,
    max_cols: int = 3,
) -> None:
    """
    Scatter plots (Actual vs Predicted) with an ideal 45° dashed line.
    """
    cols = st.columns(min(len(y_cols), max_cols))

    for i, col in enumerate(y_cols):
        actual    = y_test[col].values
        predicted = preds_test[:, i]
        r2_val    = float(metrics_df.loc[col, "R2 Score"])

        with cols[i % max_cols]:
            fig = px.scatter(
                x=actual,
                y=predicted,
                labels={"x": "Actual", "y": "Predicted"},
                title=f"{col} | R² = {r2_val:.4f}",
                opacity=0.5,
            )
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            fig.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash", width=2),
            )
            fig.update_layout(yaxis=dict(autorange=True), height=400)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Residual histograms
# ---------------------------------------------------------------------------


def render_residual_histograms(
    y_test: pd.DataFrame,
    preds_test: np.ndarray,
    y_cols: List[str],
    max_cols: int = 3,
) -> None:
    """
    Histogram of residuals (Actual – Predicted) for each target variable.
    """
    cols = st.columns(min(len(y_cols), max_cols))

    for i, col in enumerate(y_cols):
        residuals = y_test[col].values - preds_test[:, i]
        with cols[i % max_cols]:
            fig = px.histogram(
                residuals,
                nbins=30,
                title=f"Residuals: {col}",
                labels={
                    "value": "Error (Actual - Predicted)",
                    "count": "Frequency",
                },
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Workflow guide
# ---------------------------------------------------------------------------


def render_workflow_guide() -> None:
    """Render the step-by-step workflow markdown table."""
    st.markdown("### 🗺️ Workflow Guide")
    st.markdown(
        """
        | Step | Tab | Action |
        |------|-----|--------|
        | 1 | **Upload Data** | Upload Excel dataset or load from database |
        | 2 | **Preprocess** | Select X/Y features, impute missing data, handle outliers |
        | 3 | **Train Model** | Configure hyperparameters, train DAE, or load a saved model |
        | 4 | **Predict** | Evaluate on test data — metrics, scatter plots, residual analysis |
        | 5 | **What-If** | Sensitivity analysis with step changes & trend detection |
        | 6 | **History** | Review all training runs |
        | 7 | **Comparison** | Compare metrics across different model runs |
        """
    )
