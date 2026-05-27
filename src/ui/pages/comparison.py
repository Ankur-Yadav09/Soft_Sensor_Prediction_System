"""
src/ui/pages/comparison.py
===========================
Renders the "Model Comparison" page.

Displays a bar chart of average test RMSE across all training runs,
allowing the user to visually compare the effect of different
hyperparameter configurations.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def render() -> None:
    st.title("Model Comparison")

    if len(st.session_state.history) < 2:
        st.info(
            "Need at least 2 training runs to compare. "
            "Go to 'Train Model' and try different hyperparameters."
        )
        return

    history_df = pd.DataFrame(st.session_state.history)

    st.subheader("Average Test Metric Comparison")
    metric_col = (
        "Avg Test RMSE"
        if "Avg Test RMSE" in history_df.columns
        else "Avg Test MSE"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        history_df["Run ID"].astype(str),
        history_df[metric_col],
        color="skyblue",
    )
    ax.set_xlabel("Run ID")
    ax.set_ylabel(metric_col)
    st.pyplot(fig)
    plt.close(fig)
