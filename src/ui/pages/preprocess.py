"""
src/ui/pages/preprocess.py
===========================
Renders the "Preprocess" page.

Page sections (in order)
-------------------------
1.  Dataset switcher        — load any stored dataset as the active one
2.  Auto Feature Selection  — select Y first, then auto-rank X features by:
                              Mutual Information / RF Importance / Correlation
                              with top-k (3 / 5 / 7 / 10) options and full reasoning
3.  Manual Variable Selection — X and Y checkboxes (pre-filled by auto selection)
4.  Missing Data & Outliers — imputation method + outlier treatment
5.  Custom Min-Max Filter   — per-tag clipping
6.  Apply Preprocessing     — runs the full pipeline and writes session state
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data.database import list_datasets_from_db, load_dataset_from_db
from src.data.preprocessing import (
    apply_custom_filters,
    apply_outlier_treatment,
    cast_to_numeric,
    compute_feature_stats,
    impute,
    split_and_scale,
)
from src.feature_selection.selector import run_feature_selection

# ---------------------------------------------------------------------------
# Strength badge colours (displayed as colour-coded text)
# ---------------------------------------------------------------------------
_STRENGTH_STYLE = {
    "Strong":   "color:#10b981; font-weight:700",   # green
    "Moderate": "color:#f59e0b; font-weight:700",   # amber
    "Weak":     "color:#ef4444; font-weight:700",   # red
}

_TOP_K_OPTIONS = [3, 5, 7, 10]
_METHODS = [
    "Mutual Information",
    "Random Forest Importance",
    "Correlation Filtering",
]
_METHOD_DESCRIPTIONS = {
    "Mutual Information": (
        "Quantifies *any* statistical dependency (linear or non-linear) "
        "between each X sensor and the target KPIs using k-NN entropy estimation. "
        "Best for process data with complex non-linear sensor interactions."
    ),
    "Random Forest Importance": (
        "Trains a Random Forest per target KPI and averages the Mean Decrease in "
        "Impurity (MDI) across all targets. Behaves similarly to tree-SHAP values "
        "in ranking order. Robust to feature scale, outliers, and multi-collinearity."
    ),
    "Correlation Filtering": (
        "Ranks by average absolute Pearson correlation with all Y targets, then "
        "applies a collinearity filter: if two features are too similar to each "
        "other (|r| > threshold), the lower-ranked one is dropped. Produces a "
        "diverse, non-redundant feature set."
    ),
}


# ---------------------------------------------------------------------------
# Section 2: Auto Feature Selection (expander)
# ---------------------------------------------------------------------------

def _render_auto_selection(df: pd.DataFrame, numeric_cols: list) -> None:
    """
    Render the Auto Feature Selection expander.

    Writes ``st.session_state.x_cols`` and ``st.session_state.y_cols`` on
    "Apply" and triggers a rerun so the manual checkboxes are pre-populated.
    """
    with st.expander("🤖 Auto Feature Selection", expanded=False):
        st.markdown(
            "Select your **target Y** features first, then let the algorithm "
            "rank the best **X input** features automatically."
        )
        st.markdown("---")

        # ---- Step 1: Y target selector ----
        st.markdown("#### Step 1 — Select Target Y Features")
        auto_y_cols = st.multiselect(
            "Target KPI columns (Y)",
            options=numeric_cols,
            default=st.session_state.y_cols if st.session_state.y_cols else [],
            key="auto_y_selector",
        )

        if not auto_y_cols:
            st.info("Select at least one Y target to enable feature ranking.")
            return

        candidate_x = [c for c in numeric_cols if c not in auto_y_cols]
        if not candidate_x:
            st.warning("No candidate X features remain after selecting Y.")
            return

        # ---- Step 2: Method + k ----
        st.markdown("#### Step 2 — Choose Method & Number of Features")
        col_m, col_k, col_ct = st.columns([3, 1, 1])

        with col_m:
            method = st.selectbox(
                "Feature Selection Method",
                _METHODS,
                key="auto_method",
            )
        with col_k:
            top_k = st.selectbox(
                "Top-K Features",
                _TOP_K_OPTIONS,
                index=1,   # default = 5
                key="auto_k",
            )
        with col_ct:
            corr_threshold = st.number_input(
                "Collinearity threshold",
                min_value=0.50,
                max_value=0.99,
                value=0.85,
                step=0.05,
                format="%.2f",
                key="auto_corr_thresh",
                help="Only used by the Correlation Filtering method. "
                     "Features more correlated than this threshold with an "
                     "already-selected feature are dropped.",
            )

        # Method description card
        st.info(f"**{method}** — {_METHOD_DESCRIPTIONS[method]}")

        # ---- Step 3: Run analysis ----
        if st.button("🔍 Analyze Features", key="run_auto_select"):
            X_candidate = cast_to_numeric(df)[candidate_x]
            y_target    = cast_to_numeric(df)[auto_y_cols]

            with st.spinner(f"Running {method} on {len(candidate_x)} candidate features…"):
                results = run_feature_selection(
                    X_df=X_candidate,
                    y_df=y_target,
                    method=method,
                    k=top_k,
                    corr_threshold=corr_threshold,
                )

            if not results:
                st.warning("Feature selection returned no results. Check data quality.")
                return

            # Store results in session state for display persistence
            st.session_state["_auto_results"]  = results
            st.session_state["_auto_method"]   = method
            st.session_state["_auto_y_cols"]   = auto_y_cols
            st.session_state["_auto_k"]        = top_k

        # ---- Step 4: Display results (if available) ----
        if "_auto_results" not in st.session_state:
            return

        results   = st.session_state["_auto_results"]
        res_method = st.session_state.get("_auto_method", method)
        res_y      = st.session_state.get("_auto_y_cols", auto_y_cols)
        res_k      = st.session_state.get("_auto_k", top_k)

        st.markdown(f"---")
        st.markdown(
            f"#### Step 3 — Results: Top **{res_k}** Features "
            f"by **{res_method}** for Y = `{', '.join(res_y)}`"
        )

        # Summary table (clean, without Reason column)
        summary_rows = [
            {
                "Rank":       r["Rank"],
                "Feature":    r["Feature"],
                r["Score Label"]: r["Score"],
                "Strength":   r["Strength"],
            }
            for r in results
        ]
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        # Feature cards with full reasoning
        st.markdown("#### 📋 Feature-by-Feature Reasoning")
        for r in results:
            strength_css = _STRENGTH_STYLE.get(r["Strength"], "")
            with st.container():
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown(
                        f"**#{r['Rank']} — {r['Feature']}**\n\n"
                        f"{r['Score Label']}: `{r['Score']}`\n\n"
                        f"<span style='{strength_css}'>{r['Strength']}</span>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(r["Reason"])
                st.markdown("---")

        # ---- Step 4: Apply button ----
        selected_x_features = [r["Feature"] for r in results]
        st.markdown(
            f"**Selected X features:** `{', '.join(selected_x_features)}`\n\n"
            f"**Selected Y targets:**  `{', '.join(res_y)}`"
        )

        col_apply, col_clear = st.columns(2)
        with col_apply:
            if st.button(
                f"✅ Apply These {res_k} X Features + Y Targets",
                key="apply_auto_select",
            ):
                st.session_state.x_cols = selected_x_features
                st.session_state.y_cols = res_y
                st.success(
                    f"Applied! X = {selected_x_features}, Y = {res_y}. "
                    "The checkboxes below are now pre-filled."
                )
                st.rerun()

        with col_clear:
            if st.button("🗑️ Clear Auto-Selection Results", key="clear_auto"):
                for key in ["_auto_results", "_auto_method", "_auto_y_cols", "_auto_k"]:
                    st.session_state.pop(key, None)
                st.rerun()


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("Preprocess Data")

    # ------------------------------------------------------------------ #
    # Section 1: Dataset switcher
    # ------------------------------------------------------------------ #
    db_datasets = list_datasets_from_db()
    if db_datasets:
        col1, col2 = st.columns([3, 1])
        with col1:
            history_file_prep = st.selectbox(
                "Select Active Dataset",
                [r[0] for r in db_datasets],
                key="prep_dataset",
            )
        with col2:
            st.write("")
            st.write("")
            if st.button("Load Dataset", key="load_prep"):
                loaded_df = load_dataset_from_db(history_file_prep)
                if loaded_df is not None:
                    st.session_state.df = loaded_df
                    st.session_state.data_history[history_file_prep] = loaded_df
                    st.success(f"Dataset switched to {history_file_prep}")
                    st.rerun()
        st.markdown("---")

    # Guard: need data loaded
    if st.session_state.df is None:
        st.warning("Please upload data first in the 'Upload Data' tab.")
        return

    df = cast_to_numeric(st.session_state.df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------ #
    # Section 2: Auto Feature Selection
    # ------------------------------------------------------------------ #
    _render_auto_selection(df, numeric_cols)
    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Section 3: Manual Variable Selection
    # ------------------------------------------------------------------ #
    st.subheader("Variable Selection")
    st.caption(
        "Checkboxes below are pre-filled when you apply Auto Feature Selection above. "
        "You can always adjust them manually."
    )

    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown("**Select Input Features (X)**")
        select_all_x = st.checkbox(
            "Select All X",
            value=len(st.session_state.x_cols) == len(numeric_cols),
            key="sel_all_x",
        )
        x_cols = []
        for col in numeric_cols:
            default_checked = (
                True if select_all_x else col in st.session_state.x_cols
            )
            if st.checkbox(col, value=default_checked, key=f"x_{col}"):
                x_cols.append(col)

    with col_y:
        st.markdown("**Select Target Variables (Y)**")
        y_options = [c for c in numeric_cols if c not in x_cols]
        select_all_y = st.checkbox(
            "Select All Y",
            value=(
                len(st.session_state.y_cols) == len(y_options)
                and len(y_options) > 0
            ),
            key="sel_all_y",
        )
        y_cols = []
        for col in y_options:
            default_checked = (
                True if select_all_y else col in st.session_state.y_cols
            )
            if st.checkbox(col, value=default_checked, key=f"y_{col}"):
                y_cols.append(col)

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Section 4: Imputation & Outlier options
    # ------------------------------------------------------------------ #
    st.subheader("Missing Data Imputation & Outliers")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        imputation_method = st.selectbox(
            "Missing Value Imputation Method", ["Mean", "Median", "Zero"]
        )
    with col_f2:
        outlier_method = st.radio(
            "Select Outlier Treatment Method",
            [
                "None",
                "IQR Capping",
                "Min-Max Percentile Capping (1% - 99%)",
            ],
        )

    # ------------------------------------------------------------------ #
    # Section 5: Custom per-tag min-max filter
    # ------------------------------------------------------------------ #
    st.subheader("🔧 Custom Min-Max Filter (Per Tag)")
    st.caption(
        "Select specific features and set custom min/max bounds. "
        "Data outside these limits will be clipped."
    )

    all_selected       = x_cols + y_cols
    custom_filter_tags = st.multiselect(
        "Select Tags to Apply Custom Min-Max Filter",
        all_selected,
        default=[],
        key="custom_filter_tags",
    )

    custom_filters: dict = {}
    if custom_filter_tags:
        filter_cols = st.columns(3)
        for idx, tag in enumerate(custom_filter_tags):
            tag_min = float(df[tag].min())
            tag_max = float(df[tag].max())
            with filter_cols[idx % 3]:
                st.markdown(f"**{tag}**")
                st.caption(f"Data Range: {tag_min:.4f} — {tag_max:.4f}")
                c1, c2 = st.columns(2)
                with c1:
                    user_min = st.number_input(
                        "Min", value=tag_min, format="%.4f", key=f"fmin_{tag}"
                    )
                with c2:
                    user_max = st.number_input(
                        "Max", value=tag_max, format="%.4f", key=f"fmax_{tag}"
                    )
                custom_filters[tag] = {"min": user_min, "max": user_max}

    # ------------------------------------------------------------------ #
    # Section 6: Apply preprocessing
    # ------------------------------------------------------------------ #
    if st.button("Apply Preprocessing"):
        if len(x_cols) == 0 or len(y_cols) == 0:
            st.error("Please select at least one X and one Y variable.")
            return

        st.session_state.x_cols = x_cols
        st.session_state.y_cols = y_cols

        data_x = df[x_cols].copy()
        data_y = df[y_cols].copy()

        # Before stats
        st.markdown("### Feature-wise Statistics (Before Imputation)")
        st.dataframe(compute_feature_stats(data_x))

        # Pipeline
        data_x, data_y = impute(data_x, data_y, imputation_method)
        data_x, data_y = apply_outlier_treatment(data_x, data_y, outlier_method)
        data_x, data_y = apply_custom_filters(data_x, data_y, custom_filters)

        # After stats
        st.markdown("### Feature-wise Statistics (After Preprocessing)")
        st.dataframe(compute_feature_stats(data_x))

        # Split and scale
        (
            X_train_s, X_test_s,
            y_train_s, y_test_s,
            y_test_raw,
            scaler_x, scaler_y,
        ) = split_and_scale(data_x, data_y)

        # Write results to session state
        st.session_state.X_train    = X_train_s
        st.session_state.X_test     = X_test_s
        st.session_state.y_train    = y_train_s
        st.session_state.y_test     = y_test_s
        st.session_state.y_test_raw = y_test_raw
        st.session_state.scaler_x   = scaler_x
        st.session_state.scaler_y   = scaler_y

        st.success(
            f"Preprocessing complete! Applied **{outlier_method}**. "
            "Train/Test split created and features scaled. "
            f"X = {len(x_cols)} features, Y = {len(y_cols)} targets."
        )
