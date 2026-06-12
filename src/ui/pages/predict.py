"""
src/ui/pages/predict.py
========================
Renders the "Predict & Evaluate" page.

Features
--------
- Model Source: current session model or load from Model Registry
- Prediction Data: Test Set / Full Dataset / Selected Row Range
- Timestamp column auto-detection and inclusion in exports
- Test-set metrics table (RMSE, MAE, R², MAPE)
- Traffic-light KPI metric cards
- Actual vs Predicted line charts
- Scatter plots with 45° reference line
- Residual error distribution histograms
- CSV download with timestamp and errors
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.data.database import list_models_from_registry
from src.data.preprocessing import cast_to_numeric
from src.evaluation.metrics import compute_metrics
from src.persistence.model_store import load_model_from_disk
from src.ui.components import (
    render_actual_vs_predicted_lines,
    render_kpi_cards,
    render_residual_histograms,
    render_scatter_plots,
)

_MUTED = "#94a3b8"
_PRIMARY = "#4da6ff"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    """Return the first datetime or timestamp-named column, or None."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    ts_keywords = ("time", "date", "timestamp", "datetime", "ts")
    for col in df.columns:
        if any(kw in col.lower() for kw in ts_keywords):
            return col
    return None


def _prepare_x_for_prediction(
    df_rows: pd.DataFrame,
    x_cols: list,
    scaler_x,
) -> np.ndarray:
    """Fill NaN with column mean and apply the fitted scaler."""
    X = df_rows[x_cols].copy()
    X = X.fillna(X.mean())
    return scaler_x.transform(X)


def _render_model_registry_loader() -> bool:
    """
    Show the Model Registry table and a load button.
    Returns True if a model was loaded this run (triggers rerun).
    """
    registry = list_models_from_registry()
    if not registry:
        st.info(
            "No models in the registry yet. "
            "Train a model first — it will be saved automatically."
        )
        return False

    reg_df = pd.DataFrame([{
        "Model Name":   r["model_name"],
        "Algorithm":    r["algorithm"],
        "Target(s)":    ", ".join(r["y_cols"]),
        "Features":     len(r["x_cols"]),
        "Avg R²":       r["avg_r2"],
        "Avg RMSE":     r["avg_rmse"],
        "Avg MAE":      r["avg_mae"],
        "Dataset":      r["dataset_name"],
        "Trained On":   r["created_at"],
    } for r in registry])

    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    model_names = [r["model_name"] for r in registry]
    sel_name = st.selectbox("Select model to load", model_names, key="pred_reg_sel")
    sel_meta = next(r for r in registry if r["model_name"] == sel_name)

    col_load, col_info = st.columns([1, 3])
    with col_load:
        if st.button("Load Model", key="pred_reg_load", use_container_width=True):
            try:
                loaded_wrapper, sx, sy, x_cols, y_cols = load_model_from_disk(sel_meta["model_name"])
                st.session_state.model         = loaded_wrapper
                st.session_state.scaler_x      = sx
                st.session_state.scaler_y      = sy
                st.session_state.x_cols        = x_cols
                st.session_state.y_cols        = y_cols
                st.session_state.model_trained = True
                # Clear test arrays — they belong to a previous session split
                st.session_state.X_test     = None
                st.session_state.y_test     = None
                st.session_state.y_test_raw = None
                st.session_state["_pred_loaded_from_registry"] = sel_meta["model_name"]
                st.success(f"Loaded **{sel_meta['model_name']}** ({sel_meta['algorithm']})")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")
    with col_info:
        st.markdown(
            f"<p style='color:{_MUTED};font-size:0.84rem;padding-top:0.6rem'>"
            f"Targets: <b style='color:#f8fafc'>{', '.join(sel_meta['y_cols'])}</b> &nbsp;|&nbsp; "
            f"Features: <b style='color:#f8fafc'>{len(sel_meta['x_cols'])}</b> &nbsp;|&nbsp; "
            f"Dataset: <b style='color:#f8fafc'>{sel_meta['dataset_name']}</b>"
            "</p>",
            unsafe_allow_html=True,
        )
    return False


def _build_row_range_selector(df: pd.DataFrame, ts_col: Optional[str]) -> pd.DataFrame:
    """Render row / timestamp range UI and return the selected subset."""
    n_total = len(df)

    if ts_col and not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        try:
            df = df.copy()
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        except Exception:
            pass

    use_ts = (
        ts_col is not None
        and pd.api.types.is_datetime64_any_dtype(df[ts_col])
        and df[ts_col].notna().any()
    )

    if use_ts:
        tab_row, tab_ts = st.tabs(["Row Range", "Timestamp Range"])
    else:
        tab_row = st.container()

    with tab_row:
        c1, c2 = st.columns(2)
        with c1:
            start_row = st.number_input(
                "Start Row (1-indexed)", min_value=1, max_value=n_total, value=1, key="pred_start_row"
            )
        with c2:
            end_row = st.number_input(
                "End Row (1-indexed)", min_value=1, max_value=n_total, value=n_total, key="pred_end_row"
            )
        if start_row > end_row:
            st.warning("Start row must be ≤ end row.")
            end_row = start_row
        row_subset = df.iloc[int(start_row) - 1: int(end_row)]
        st.caption(f"Selected {len(row_subset)} row(s) — rows {int(start_row)}–{int(end_row)}.")
        selected_by_row = row_subset

    if use_ts:
        with tab_ts:
            ts_sorted = df[ts_col].dropna().sort_values()
            ts_min = ts_sorted.iloc[0].to_pydatetime()
            ts_max = ts_sorted.iloc[-1].to_pydatetime()
            c1, c2 = st.columns(2)
            with c1:
                start_ts = st.date_input("Start Date", value=ts_min.date(), key="pred_start_ts")
            with c2:
                end_ts = st.date_input("End Date", value=ts_max.date(), key="pred_end_ts")
            ts_mask = (
                df[ts_col].notna()
                & (df[ts_col].dt.date >= start_ts)
                & (df[ts_col].dt.date <= end_ts)
            )
            ts_subset = df[ts_mask]
            st.caption(f"Selected {len(ts_subset)} row(s) between {start_ts} and {end_ts}.")
        # Return whichever tab was last interacted with — use row selection as default
        return selected_by_row

    return selected_by_row


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("Predict & Evaluate")

    # ------------------------------------------------------------------ #
    # Model Source
    # ------------------------------------------------------------------ #
    st.subheader("Model Source")
    model_source = st.radio(
        "Select model source",
        ["Use Current Session Model", "Load Previously Trained Model"],
        horizontal=True,
        key="pred_model_source",
        label_visibility="collapsed",
    )

    if model_source == "Load Previously Trained Model":
        st.markdown("#### Model Registry")
        _render_model_registry_loader()
        loaded_name = st.session_state.get("_pred_loaded_from_registry")
        if loaded_name:
            st.info(
                f"Active model: **{loaded_name}** (loaded from registry). "
                "Test Set option is unavailable — use Full Dataset or Selected Range below."
            )
        st.markdown("---")

    # ------------------------------------------------------------------ #
    # Guard: model must be available
    # ------------------------------------------------------------------ #
    if not st.session_state.model_trained:
        st.warning("Please train or load a model first.")
        return

    model    = st.session_state.model
    scaler_x = st.session_state.scaler_x
    scaler_y = st.session_state.scaler_y
    x_cols   = st.session_state.x_cols
    y_cols   = st.session_state.y_cols

    from_registry = bool(st.session_state.get("_pred_loaded_from_registry"))
    has_test_split = st.session_state.X_test is not None

    # ------------------------------------------------------------------ #
    # Prediction Data Source
    # ------------------------------------------------------------------ #
    st.subheader("Prediction Data")

    data_options = ["Full Dataset", "Selected Data Range"]
    if has_test_split and not from_registry:
        data_options = ["Test Set (Default)"] + data_options

    pred_data_opt = st.radio(
        "Predict on",
        data_options,
        horizontal=True,
        key="pred_data_opt",
    )

    df_full = st.session_state.df
    ts_col: Optional[str] = None
    y_actual: Optional[pd.DataFrame] = None
    X_pred: Optional[np.ndarray] = None
    ts_series: Optional[pd.Series] = None

    if df_full is not None:
        df_full = cast_to_numeric(df_full)
        ts_col = _find_timestamp_col(st.session_state.df)

    if pred_data_opt == "Test Set (Default)":
        X_pred   = st.session_state.X_test
        y_actual = st.session_state.y_test_raw
        if df_full is not None and ts_col and y_actual is not None:
            try:
                ts_series = st.session_state.df[ts_col].iloc[y_actual.index]
            except Exception:
                ts_series = None

    elif pred_data_opt == "Full Dataset":
        if df_full is None:
            st.warning("No dataset loaded. Upload and preprocess data first.")
            return
        missing_x = [c for c in x_cols if c not in df_full.columns]
        if missing_x:
            st.error(f"Loaded model expects features not found in the active dataset: {missing_x}")
            return
        X_pred   = _prepare_x_for_prediction(df_full, x_cols, scaler_x)
        y_actual = df_full[y_cols] if all(c in df_full.columns for c in y_cols) else None
        if ts_col and ts_col in st.session_state.df.columns:
            ts_series = st.session_state.df[ts_col].reset_index(drop=True)

    elif pred_data_opt == "Selected Data Range":
        if df_full is None:
            st.warning("No dataset loaded. Upload and preprocess data first.")
            return
        missing_x = [c for c in x_cols if c not in df_full.columns]
        if missing_x:
            st.error(f"Loaded model expects features not found in the active dataset: {missing_x}")
            return
        df_subset = _build_row_range_selector(df_full, ts_col)
        if len(df_subset) == 0:
            st.warning("Empty selection — adjust the row or timestamp range.")
            return
        X_pred   = _prepare_x_for_prediction(df_subset, x_cols, scaler_x)
        y_actual = df_subset[y_cols].reset_index(drop=True) if all(c in df_subset.columns for c in y_cols) else None
        if ts_col and ts_col in df_subset.columns:
            ts_series = df_subset[ts_col].reset_index(drop=True)

    if X_pred is None:
        st.error("Could not prepare prediction input. Check that X features match the loaded model.")
        return

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Run inference
    # ------------------------------------------------------------------ #
    preds = scaler_y.inverse_transform(model.predict_scaled(X_pred))

    # ------------------------------------------------------------------ #
    # Metrics (only when we have actuals with matching shape)
    # ------------------------------------------------------------------ #
    show_metrics = (
        y_actual is not None
        and len(y_actual) == len(preds)
        and all(c in y_actual.columns for c in y_cols)
    )

    if show_metrics:
        y_actual_aligned = y_actual[y_cols].reset_index(drop=True)
        metrics_df = compute_metrics(y_actual_aligned, preds, y_cols)

        st.subheader("Test Set Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("📊 Model Performance Summary")
        render_kpi_cards(metrics_df, y_cols)

        st.subheader("📈 Actual vs Predicted")
        render_actual_vs_predicted_lines(y_actual_aligned, preds, y_cols, metrics_df)

        st.subheader("🎯 Scatter Plot: Actual vs Predicted")
        render_scatter_plots(y_actual_aligned, preds, y_cols, metrics_df)

        st.subheader("📉 Residual Analysis")
        render_residual_histograms(y_actual_aligned, preds, y_cols)
    else:
        st.subheader("📈 Predicted Values")
        pred_display = pd.DataFrame(preds, columns=[f"Predicted_{c}" for c in y_cols])
        if ts_series is not None:
            pred_display.insert(0, ts_col, ts_series.values)
        st.dataframe(pred_display.head(200), use_container_width=True)
        if len(preds) > 200:
            st.caption(f"Showing first 200 of {len(preds)} rows.")

    # ------------------------------------------------------------------ #
    # CSV export (with timestamp)
    # ------------------------------------------------------------------ #
    st.subheader("📥 Export Predictions")

    export_rows = []
    for i in range(len(preds)):
        row: dict = {}
        if ts_series is not None:
            try:
                row[ts_col] = ts_series.iloc[i]
            except Exception:
                pass
        if show_metrics:
            for j, col in enumerate(y_cols):
                row[f"Actual_{col}"]    = float(y_actual_aligned[col].iloc[i])
                row[f"Predicted_{col}"] = float(preds[i, j])
                row[f"Error_{col}"]     = float(y_actual_aligned[col].iloc[i]) - float(preds[i, j])
        else:
            for j, col in enumerate(y_cols):
                row[f"Predicted_{col}"] = float(preds[i, j])
        export_rows.append(row)

    export_df = pd.DataFrame(export_rows)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    model_label = st.session_state.get("_pred_loaded_from_registry") or "session"
    filename = f"Predictions_{pred_data_opt.replace(' ', '_')}_{model_label}.csv"

    st.download_button(
        label="📥 Download Predictions (CSV)",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )

    if ts_series is not None:
        st.caption(f"Timestamp column **{ts_col}** included in export.")
    if show_metrics:
        st.caption("Export includes: Timestamp (if detected), Actual, Predicted, and Error columns per target.")
    else:
        st.caption("Export includes: Timestamp (if detected) and Predicted columns per target.")
