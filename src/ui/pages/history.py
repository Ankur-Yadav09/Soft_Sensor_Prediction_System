"""
src/ui/pages/history.py
========================
Renders the "Training History" page.

Lists all training runs from the current session and allows the user to
reload any run's model as the active model.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render() -> None:
    st.title("Training History")

    if not st.session_state.history:
        st.info("No training history available. Train a model first.")
        return

    # Drop the raw model object before displaying (not serialisable)
    history_df = pd.DataFrame(st.session_state.history).drop(
        columns=["Model"], errors="ignore"
    )
    # Show Model Type first if it exists
    cols = ["Run ID", "Model Type"] + [
        c for c in history_df.columns if c not in ("Run ID", "Model Type")
    ]
    history_df = history_df[[c for c in cols if c in history_df.columns]]
    st.dataframe(history_df, use_container_width=True)

    # ------------------------------------------------------------------ #
    # Reload a past run's model
    # ------------------------------------------------------------------ #
    load_run = st.selectbox(
        "Select a Run ID to load as active model",
        history_df["Run ID"].tolist(),
    )
    if st.button("Load Model"):
        run_data = next(
            item
            for item in st.session_state.history
            if item["Run ID"] == load_run
        )
        st.session_state.model         = run_data["Model"]
        st.session_state.model_trained = True
        st.success(f"Model from Run {load_run} loaded successfully!")
