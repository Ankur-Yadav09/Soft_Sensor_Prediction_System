"""
src/ui/pages/comparison.py
===========================
Renders the "Model Comparison" page.

Shows a summary table and three grouped Plotly bar charts (Avg R², RMSE, MAE)
across all in-session training runs so the user can judge which hyperparameter
configuration produced the best model.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render() -> None:
    st.title("Model Comparison")

    if len(st.session_state.history) < 2:
        st.info(
            "Need at least 2 training runs to compare. "
            "Go to 'Train Model' and try different hyperparameters."
        )
        return

    history_df = pd.DataFrame(st.session_state.history).drop(
        columns=["Model"], errors="ignore"
    )

    # ------------------------------------------------------------------ #
    # Best-run callout
    # ------------------------------------------------------------------ #
    if "Avg Test R2" in history_df.columns:
        best_idx  = history_df["Avg Test R2"].idxmax()
        best_run  = history_df.loc[best_idx, "Run ID"]
        best_r2   = history_df.loc[best_idx, "Avg Test R2"]
        best_rmse = history_df.loc[best_idx, "Avg Test RMSE"]
        st.success(
            f"Best run: **Run {best_run}** — "
            f"Avg R² = `{best_r2:.4f}`, Avg RMSE = `{best_rmse:.4f}`"
        )
    else:
        best_idx = history_df["Avg Test RMSE"].idxmin()
        best_run = history_df.loc[best_idx, "Run ID"]
        st.success(f"Best run by RMSE: **Run {best_run}**")

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    st.subheader("Summary Table")
    display_cols = [c for c in [
        "Run ID", "Latent Dim", "Masking", "Epochs",
        "Avg Test R2", "Avg Test RMSE", "Avg Test MAE",
    ] if c in history_df.columns]
    st.dataframe(
        history_df[display_cols].style.highlight_max(
            subset=[c for c in ["Avg Test R2"] if c in history_df.columns],
            color="rgba(16,185,129,0.3)",
        ).highlight_min(
            subset=[c for c in ["Avg Test RMSE", "Avg Test MAE"]
                    if c in history_df.columns],
            color="rgba(16,185,129,0.3)",
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Multi-metric bar charts
    # ------------------------------------------------------------------ #
    run_labels = ["Run " + str(r) for r in history_df["Run ID"]]

    has_r2  = "Avg Test R2"   in history_df.columns
    has_mae = "Avg Test MAE"  in history_df.columns

    n_plots = 1 + int(has_r2) + int(has_mae)
    subplot_titles = []
    if has_r2:
        subplot_titles.append("Avg R² (higher is better)")
    subplot_titles.append("Avg RMSE (lower is better)")
    if has_mae:
        subplot_titles.append("Avg MAE (lower is better)")

    fig = make_subplots(
        rows=1, cols=n_plots,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
    )

    # Colour: highlight best run in accent green, others in steel blue
    best_run_label = "Run " + str(best_run)
    colors = [
        "#10b981" if lbl == best_run_label else "#4da6ff"
        for lbl in run_labels
    ]

    col_idx = 1

    if has_r2:
        fig.add_trace(
            go.Bar(
                x=run_labels,
                y=history_df["Avg Test R2"],
                marker_color=colors,
                name="Avg R²",
                text=[f"{v:.3f}" for v in history_df["Avg Test R2"]],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=col_idx,
        )
        col_idx += 1

    fig.add_trace(
        go.Bar(
            x=run_labels,
            y=history_df["Avg Test RMSE"],
            marker_color=colors,
            name="Avg RMSE",
            text=[f"{v:.4f}" for v in history_df["Avg Test RMSE"]],
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=col_idx,
    )
    col_idx += 1

    if has_mae:
        fig.add_trace(
            go.Bar(
                x=run_labels,
                y=history_df["Avg Test MAE"],
                marker_color=colors,
                name="Avg MAE",
                text=[f"{v:.4f}" for v in history_df["Avg Test MAE"]],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

    st.subheader("Metric Comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Green bar = best run. "
        "R² closer to 1.0 is better; RMSE and MAE closer to 0 is better."
    )
