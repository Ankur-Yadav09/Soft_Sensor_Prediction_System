"""
src/ui/pages/preprocess.py
===========================
Renders the "Preprocess" page.

Page sections
--------------
1. Dataset switcher
2. Data Understanding   ← feature-level analysis panel
3. Basic Preprocessing  ← remove rows, impute, outliers, domain filters
4. Intelligent Auto Feature Selection  ← 12-method engine (unchanged)
5. Manual Variable Selection + Final Apply
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
from src.feature_selection.auto_selector import (
    ALL_METHOD_IDS,
    METHOD_CATEGORIES,
    METHOD_LABELS,
    AutoSelectionResult,
    run_auto_feature_selection,
)

# ---------------------------------------------------------------------------
# Theme constants (consistent with app-wide dark theme)
# ---------------------------------------------------------------------------
_CARD = "background:rgba(30,41,59,0.7);border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:0.8rem"
_PRIMARY  = "#4da6ff"
_ACCENT   = "#10b981"
_WARN     = "#f59e0b"
_DANGER   = "#ef4444"
_MUTED    = "#94a3b8"

_REC_COLORS = {
    "Highly Recommended": _ACCENT,
    "Recommended":        _PRIMARY,
    "Optional":           _WARN,
    "Remove":             _DANGER,
}
_REC_ICONS = {
    "Highly Recommended": "🟢",
    "Recommended":        "🔵",
    "Optional":           "🟡",
    "Remove":             "🔴",
}


# ---------------------------------------------------------------------------
# Section header helper
# ---------------------------------------------------------------------------

def _sync_checkboxes(x_list: List[str], y_list: List[str], all_cols: List[str]) -> None:
    """Write individual checkbox widget states so the manual selection reflects the choice."""
    for col in all_cols:
        st.session_state[f"x_{col}"] = col in x_list
    for col in all_cols:
        if col not in x_list:
            st.session_state[f"y_{col}"] = col in y_list


def _section_header(icon: str, title: str, subtitle: str = "") -> None:
    sub_html = f"<p style='margin:0.25rem 0 0;color:{_MUTED};font-size:0.88rem'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div style='{_CARD}'>"
        f"<h3 style='margin:0;color:{_PRIMARY};font-family:Outfit,sans-serif'>{icon} {title}</h3>"
        f"{sub_html}</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# SECTION 2 — DATA UNDERSTANDING
# ===========================================================================

def _count_iqr_outliers(series: pd.Series) -> int:
    s = series.dropna()
    if len(s) < 4:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def _distribution_label(skew: float) -> str:
    if skew > 1.0:
        return "Highly right-skewed"
    if skew > 0.5:
        return "Moderately right-skewed"
    if skew < -1.0:
        return "Highly left-skewed"
    if skew < -0.5:
        return "Moderately left-skewed"
    return "Approximately symmetric"


def _render_data_understanding(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    _section_header("🔍", "Data Understanding",
                    "Explore any feature in detail — statistics, distribution and outlier profile.")

    with st.container():
        selected_col = st.selectbox(
            "Select a feature to analyse",
            numeric_cols,
            key="du_feature_select",
        )
        if not selected_col:
            return

        series_raw = df[selected_col]
        series     = series_raw.dropna()
        n_total    = len(df)
        n_missing  = int(series_raw.isnull().sum())
        n_unique   = int(series_raw.nunique())
        n_dupes    = int(df.duplicated().sum())
        n_outliers = _count_iqr_outliers(series_raw)

        if len(series) == 0:
            st.warning("All values are missing for this feature.")
            return

        mean_v   = float(series.mean())
        median_v = float(series.median())
        std_v    = float(series.std())
        min_v    = float(series.min())
        max_v    = float(series.max())
        skew_v   = float(series.skew())
        kurt_v   = float(series.kurtosis())

        # ---- Row 1: core identity
        st.markdown(f"<p style='color:{_MUTED};font-size:0.82rem;margin-bottom:0.4rem'>"
                    f"Data Type: <b style='color:#f8fafc'>{series_raw.dtype}</b> &nbsp;|&nbsp; "
                    f"Distribution: <b style='color:#f8fafc'>{_distribution_label(skew_v)}</b>"
                    f"</p>", unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Records",  n_total)
        c2.metric("Missing",        f"{n_missing} ({n_missing/n_total*100:.1f}%)")
        c3.metric("Unique Values",  n_unique)
        c4.metric("Duplicate Rows", n_dupes)
        c5.metric("Outliers (IQR)", n_outliers)
        c6.metric("Skewness",       f"{skew_v:.3f}")

        c7, c8, c9, c10, c11, c12 = st.columns(6)
        c7.metric("Min",      f"{min_v:.4g}")
        c8.metric("Max",      f"{max_v:.4g}")
        c9.metric("Mean",     f"{mean_v:.4g}")
        c10.metric("Median",  f"{median_v:.4g}")
        c11.metric("Std Dev", f"{std_v:.4g}")
        c12.metric("Kurtosis", f"{kurt_v:.3f}")

        # ---- Distribution summary text
        flags = []
        if n_missing > 0:
            flags.append(f"**{n_missing} missing values** ({n_missing/n_total*100:.1f}%)")
        if n_outliers > 0:
            flags.append(f"**{n_outliers} potential outliers** (IQR method)")
        if abs(skew_v) > 1:
            flags.append(f"**high skewness** ({skew_v:+.2f}) — consider transformation")

        flag_text = (", ".join(flags) + ".") if flags else "No data quality issues detected."
        st.info(
            f"**{selected_col}** — {_distribution_label(skew_v).lower()} distribution "
            f"(skew = {skew_v:+.3f}, kurtosis = {kurt_v:.3f}). "
            f"Range: {min_v:.4g} → {max_v:.4g}, mean ± σ = {mean_v:.4g} ± {std_v:.4g}. "
            + flag_text
        )

        # ---- Charts
        ch1, ch2 = st.columns(2)

        with ch1:
            fig_hist = px.histogram(
                series, nbins=40,
                title=f"Distribution — {selected_col}",
                labels={"value": selected_col, "count": "Frequency"},
                color_discrete_sequence=[_PRIMARY],
            )
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc", height=300,
                margin=dict(l=5, r=5, t=40, b=5),
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with ch2:
            fig_box = go.Figure(go.Box(
                y=series, name=selected_col,
                marker_color=_PRIMARY,
                line_color=_PRIMARY,
                boxmean="sd",
            ))
            fig_box.update_layout(
                title=f"Box Plot — {selected_col}",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc", height=300,
                margin=dict(l=5, r=5, t=40, b=5),
            )
            st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")


# ===========================================================================
# SECTION 3 — BASIC PREPROCESSING
# ===========================================================================

# --- Outlier helpers -------------------------------------------------------

def _zscore_outlier_count(series: pd.Series, thr: float) -> int:
    s = series.dropna()
    if s.std() == 0:
        return 0
    return int(((s - s.mean()).abs() / s.std() > thr).sum())


def _apply_zscore_cap(
    working: pd.DataFrame, cols: List[str], thr: float
) -> Tuple[pd.DataFrame, int]:
    """Cap values where |z| > thr to thr × σ boundaries."""
    total = 0
    for col in cols:
        if col not in working.columns:
            continue
        s = working[col]
        mu, sigma = s.mean(), s.std()
        if sigma == 0:
            continue
        lo, hi = mu - thr * sigma, mu + thr * sigma
        n = int(((s < lo) | (s > hi)).sum())
        working[col] = s.clip(lo, hi)
        total += n
    return working, total


def _apply_winsorization(
    working: pd.DataFrame, cols: List[str], lo_pct: float, hi_pct: float
) -> Tuple[pd.DataFrame, int]:
    """Cap at user-specified percentiles."""
    total = 0
    for col in cols:
        if col not in working.columns:
            continue
        s = working[col]
        lo, hi = s.quantile(lo_pct / 100), s.quantile(hi_pct / 100)
        n = int(((s < lo) | (s > hi)).sum())
        working[col] = s.clip(lo, hi)
        total += n
    return working, total


def _apply_capping_flooring(
    working: pd.DataFrame, cols: List[str], multiplier: float
) -> Tuple[pd.DataFrame, int]:
    """IQR-based capping with a user-specified multiplier."""
    total = 0
    for col in cols:
        if col not in working.columns:
            continue
        s = working[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        n = int(((s < lo) | (s > hi)).sum())
        working[col] = s.clip(lo, hi)
        total += n
    return working, total


def _apply_remove_outliers_iqr(
    working: pd.DataFrame, cols: List[str]
) -> Tuple[pd.DataFrame, int]:
    """Drop rows with IQR outliers in any of the selected columns."""
    n_before = len(working)
    mask = pd.Series(True, index=working.index)
    for col in cols:
        if col not in working.columns:
            continue
        s = working[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= (s >= lo) & (s <= hi)
    result = working[mask].reset_index(drop=True)
    return result, n_before - len(result)


def _apply_remove_outliers_zscore(
    working: pd.DataFrame, cols: List[str], thr: float
) -> Tuple[pd.DataFrame, int]:
    """Drop rows where |z-score| > thr in any selected column."""
    n_before = len(working)
    mask = pd.Series(True, index=working.index)
    for col in cols:
        if col not in working.columns:
            continue
        s = working[col]
        if s.std() == 0:
            continue
        z = (s - s.mean()) / s.std()
        mask &= z.abs() <= thr
    result = working[mask].reset_index(drop=True)
    return result, n_before - len(result)


# --- Render basic preprocessing -------------------------------------------

def _render_basic_preprocessing(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    _section_header(
        "⚙️", "Basic Preprocessing",
        "Remove records, impute missing values, handle outliers, and apply domain filters. "
        "Click 'Apply Cleaning' to update the active dataset.",
    )

    tab_remove, tab_impute, tab_outlier, tab_filter = st.tabs([
        "🗑️ Remove Records",
        "🔧 Missing Values",
        "📊 Outlier Treatment",
        "📐 Domain Filters",
    ])

    # ---- Tab 1: Remove Records -----------------------------------------
    with tab_remove:
        st.markdown("#### Remove Records")
        c1, c2 = st.columns(2)
        with c1:
            remove_missing = st.checkbox(
                "Remove rows with **any** missing values",
                key="bp_rm_missing",
            )
            if remove_missing:
                n_missing_rows = int(df.isnull().any(axis=1).sum())
                st.caption(f"Will remove **{n_missing_rows}** row(s) with at least one NaN.")
        with c2:
            remove_dupes = st.checkbox(
                "Remove **duplicate** records",
                key="bp_rm_dupes",
            )
            if remove_dupes:
                n_dupes = int(df.duplicated().sum())
                st.caption(f"Will remove **{n_dupes}** duplicate row(s).")

    # ---- Tab 2: Missing Values -----------------------------------------
    with tab_impute:
        st.markdown("#### Handle Missing Values")
        impute_cols = st.multiselect(
            "Apply to columns",
            numeric_cols,
            default=[],
            key="bp_impute_cols",
            help="Leave empty to apply to all numeric columns that have missing values.",
        )
        if not impute_cols:
            impute_target = [c for c in numeric_cols if df[c].isnull().any()]
            st.caption(
                f"No columns selected — will apply to all {len(impute_target)} column(s) "
                "with missing values when triggered."
            )
        else:
            impute_target = impute_cols

        im_col1, im_col2 = st.columns([2, 1])
        with im_col1:
            impute_method = st.selectbox(
                "Imputation Method",
                ["None", "Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Custom Value"],
                key="bp_impute_method",
            )
        with im_col2:
            custom_fill_val = 0.0
            if impute_method == "Custom Value":
                custom_fill_val = st.number_input(
                    "Fill Value", value=0.0, format="%.4f", key="bp_custom_val"
                )

        if impute_method != "None":
            total_missing = sum(df[c].isnull().sum() for c in impute_target)
            st.caption(
                f"**{impute_method}** imputation will fill **{total_missing}** missing value(s) "
                f"across {len(impute_target)} column(s)."
            )

    # ---- Tab 3: Outlier Treatment --------------------------------------
    with tab_outlier:
        st.markdown("#### Handle Outliers")

        out_col1, out_col2 = st.columns([2, 1])
        with out_col1:
            outlier_method = st.selectbox(
                "Outlier Treatment Method",
                [
                    "None",
                    "IQR Capping (Q1-1.5×IQR, Q3+1.5×IQR)",
                    "Z-Score Capping",
                    "Winsorization",
                    "Capping/Flooring (custom IQR multiplier)",
                    "Remove Outliers (IQR)",
                    "Remove Outliers (Z-Score)",
                ],
                key="bp_outlier_method",
            )
        with out_col2:
            zscore_thr  = 3.0
            wins_lo     = 2.5
            wins_hi     = 97.5
            cap_mult    = 1.5

            if "Z-Score" in outlier_method:
                zscore_thr = st.number_input(
                    "Z-Score threshold", min_value=1.0, max_value=10.0,
                    value=3.0, step=0.5, key="bp_zscore_thr",
                )
            elif "Winsorization" in outlier_method:
                wins_lo = st.number_input(
                    "Lower %", min_value=0.1, max_value=10.0,
                    value=2.5, step=0.5, key="bp_wins_lo",
                )
                wins_hi = st.number_input(
                    "Upper %", min_value=90.0, max_value=99.9,
                    value=97.5, step=0.5, key="bp_wins_hi",
                )
            elif "custom IQR" in outlier_method:
                cap_mult = st.number_input(
                    "IQR multiplier", min_value=0.5, max_value=5.0,
                    value=1.5, step=0.5, key="bp_cap_mult",
                )

        outlier_cols = st.multiselect(
            "Apply to columns",
            numeric_cols,
            default=numeric_cols,
            key="bp_outlier_cols",
        )

        # Show live outlier counts
        if outlier_method != "None" and outlier_cols:
            preview_rows = []
            for col in outlier_cols[:15]:
                n_iqr = _count_iqr_outliers(df[col])
                n_z   = _zscore_outlier_count(df[col], zscore_thr)
                if n_iqr > 0 or n_z > 0:
                    preview_rows.append({
                        "Column": col,
                        "IQR Outliers": n_iqr,
                        f"Z-Score Outliers (>{zscore_thr})": n_z,
                    })
            if preview_rows:
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)
            else:
                st.success("No outliers detected in selected columns.")

    # ---- Tab 4: Domain Filters -----------------------------------------
    with tab_filter:
        st.markdown("#### Domain-Based Min / Max Filtering")
        st.caption("Clip feature values to physically meaningful bounds.")

        filter_tags = st.multiselect(
            "Select features to filter",
            numeric_cols,
            default=[],
            key="bp_filter_tags",
        )

        domain_filters: Dict[str, Dict[str, float]] = {}
        if filter_tags:
            n_fcols = min(3, len(filter_tags))
            f_cols = st.columns(n_fcols)
            for idx, tag in enumerate(filter_tags):
                tag_min = float(df[tag].min())
                tag_max = float(df[tag].max())
                with f_cols[idx % n_fcols]:
                    st.markdown(
                        f"<p style='color:{_PRIMARY};font-weight:700;margin-bottom:0.2rem'>{tag}</p>"
                        f"<p style='color:{_MUTED};font-size:0.8rem;margin-top:0'>Data range: "
                        f"{tag_min:.4g} — {tag_max:.4g}</p>",
                        unsafe_allow_html=True,
                    )
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        umin = st.number_input(
                            "Min", value=tag_min, format="%.4f", key=f"bp_fmin_{tag}"
                        )
                    with fc2:
                        umax = st.number_input(
                            "Max", value=tag_max, format="%.4f", key=f"bp_fmax_{tag}"
                        )
                    domain_filters[tag] = {"min": umin, "max": umax}

    # ---- Apply Cleaning button -----------------------------------------
    st.markdown("---")

    # Config summary
    active_steps = []
    if st.session_state.get("bp_rm_missing"):
        active_steps.append("Remove missing rows")
    if st.session_state.get("bp_rm_dupes"):
        active_steps.append("Remove duplicates")
    if st.session_state.get("bp_impute_method", "None") != "None":
        active_steps.append(f"Impute: {st.session_state.get('bp_impute_method')}")
    if st.session_state.get("bp_outlier_method", "None") != "None":
        active_steps.append(f"Outliers: {st.session_state.get('bp_outlier_method','').split(' (')[0]}")
    if domain_filters:
        active_steps.append(f"Domain filters: {len(domain_filters)} tag(s)")

    if active_steps:
        st.markdown(
            "<p style='color:" + _MUTED + ";font-size:0.85rem'>"
            "Configured steps: "
            + " → ".join(f"<b style='color:#f8fafc'>{s}</b>" for s in active_steps)
            + "</p>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No preprocessing steps configured — all tabs are set to 'None'.")

    apply_col, _ = st.columns([1, 3])
    with apply_col:
        apply_btn = st.button("✅ Apply Cleaning", key="bp_apply", use_container_width=True)

    if apply_btn:
        working = cast_to_numeric(st.session_state.df).copy()
        before_rows = len(working)
        action_log: List[str] = []

        # Step 1: Remove missing rows
        if st.session_state.get("bp_rm_missing"):
            n_before = len(working)
            working = working.dropna().reset_index(drop=True)
            n_removed = n_before - len(working)
            action_log.append(f"Removed **{n_removed}** row(s) with missing values.")

        # Step 2: Remove duplicates
        if st.session_state.get("bp_rm_dupes"):
            n_before = len(working)
            working = working.drop_duplicates().reset_index(drop=True)
            n_removed = n_before - len(working)
            action_log.append(f"Removed **{n_removed}** duplicate row(s).")

        # Step 3: Impute missing values
        _imp_method = st.session_state.get("bp_impute_method", "None")
        if _imp_method != "None":
            _imp_cols = [c for c in (
                st.session_state.get("bp_impute_cols") or
                [c for c in numeric_cols if working[c].isnull().any()]
            ) if c in working.columns]
            n_filled = 0
            for col in _imp_cols:
                n_missing = working[col].isnull().sum()
                if n_missing == 0:
                    continue
                if _imp_method == "Mean":
                    working[col] = working[col].fillna(working[col].mean())
                elif _imp_method == "Median":
                    working[col] = working[col].fillna(working[col].median())
                elif _imp_method == "Mode":
                    mode_val = working[col].mode()
                    working[col] = working[col].fillna(mode_val.iloc[0] if not mode_val.empty else 0)
                elif _imp_method == "Forward Fill":
                    working[col] = working[col].ffill()
                elif _imp_method == "Backward Fill":
                    working[col] = working[col].bfill()
                elif _imp_method == "Custom Value":
                    working[col] = working[col].fillna(custom_fill_val)
                n_filled += n_missing
            action_log.append(
                f"Imputed **{n_filled}** missing value(s) using **{_imp_method}** "
                f"across {len(_imp_cols)} column(s)."
            )

        # Step 4: Outlier treatment
        _out_method = st.session_state.get("bp_outlier_method", "None")
        _out_cols   = [c for c in (st.session_state.get("bp_outlier_cols") or []) if c in working.columns]
        if _out_method != "None" and _out_cols:
            if _out_method == "IQR Capping (Q1-1.5×IQR, Q3+1.5×IQR)":
                working, n_affected = _apply_capping_flooring(working, _out_cols, 1.5)
                action_log.append(f"IQR capping applied — **{n_affected}** value(s) capped.")
            elif "Z-Score Capping" in _out_method:
                working, n_affected = _apply_zscore_cap(working, _out_cols, zscore_thr)
                action_log.append(
                    f"Z-Score capping (thr={zscore_thr}) — **{n_affected}** value(s) capped."
                )
            elif "Winsorization" in _out_method:
                working, n_affected = _apply_winsorization(working, _out_cols, wins_lo, wins_hi)
                action_log.append(
                    f"Winsorization ({wins_lo:.1f}%–{wins_hi:.1f}%) — **{n_affected}** value(s) capped."
                )
            elif "custom IQR" in _out_method:
                working, n_affected = _apply_capping_flooring(working, _out_cols, cap_mult)
                action_log.append(
                    f"IQR capping (×{cap_mult}) — **{n_affected}** value(s) capped."
                )
            elif _out_method == "Remove Outliers (IQR)":
                working, n_removed = _apply_remove_outliers_iqr(working, _out_cols)
                action_log.append(f"Removed **{n_removed}** outlier row(s) via IQR.")
            elif "Remove Outliers (Z-Score)" in _out_method:
                working, n_removed = _apply_remove_outliers_zscore(working, _out_cols, zscore_thr)
                action_log.append(
                    f"Removed **{n_removed}** outlier row(s) via Z-Score (thr={zscore_thr})."
                )

        # Step 5: Domain filters
        if domain_filters:
            n_before = len(working)
            for tag, bounds in domain_filters.items():
                if tag in working.columns:
                    working[tag] = working[tag].clip(bounds["min"], bounds["max"])
            action_log.append(f"Domain filters applied to **{len(domain_filters)}** tag(s).")

        after_rows = len(working)

        # Update session state
        st.session_state.df = working

        # Show summary
        st.success(
            f"Cleaning complete. Records: **{before_rows}** → **{after_rows}** "
            f"({before_rows - after_rows} row(s) removed)."
        )
        for msg in action_log:
            st.markdown(f"- {msg}")

        # Before / After stats table
        if action_log:
            ba1, ba2 = st.columns(2)
            with ba1:
                st.metric("Before", before_rows, delta=None)
            with ba2:
                delta_val = after_rows - before_rows
                st.metric("After", after_rows, delta=f"{delta_val:+d} rows", delta_color="inverse")
        st.rerun()

    st.markdown("---")


# ===========================================================================
# SECTION 4 — INTELLIGENT AUTO FEATURE SELECTION  (completely unchanged)
# ===========================================================================

def _rec_badge(rec: str) -> str:
    color = _REC_COLORS.get(rec, "#94a3b8")
    icon  = _REC_ICONS.get(rec, "⚪")
    return (
        f"<span style='background:{color};color:#fff;"
        f"padding:2px 10px;border-radius:12px;"
        f"font-size:0.78rem;font-weight:700'>"
        f"{icon} {rec}</span>"
    )


def _method_checkboxes(
    available_methods: List[str],
    default_enabled: List[str],
) -> List[str]:
    cat_map: Dict[str, List[str]] = {}
    for mid in available_methods:
        cat = METHOD_CATEGORIES[mid]
        cat_map.setdefault(cat, []).append(mid)

    selected: List[str] = []
    cols = st.columns(3)
    col_idx = 0
    for cat, mids in cat_map.items():
        with cols[col_idx % 3]:
            st.markdown(f"**{cat}**")
            for mid in mids:
                label = METHOD_LABELS[mid]
                checked = mid in default_enabled
                if st.checkbox(label, value=checked, key=f"chk_{mid}"):
                    selected.append(mid)
        col_idx += 1
    return selected


def _plot_consensus_bar(consensus_df: pd.DataFrame) -> go.Figure:
    df = consensus_df.reset_index().rename(columns={"Rank": "Rank_col"}) \
        if "Rank" not in consensus_df.columns else consensus_df.reset_index()
    df = df.sort_values("ConfidenceScore", ascending=True).tail(30)

    colors = [_REC_COLORS.get(r, "#94a3b8") for r in df["Recommendation"]]
    fig = go.Figure(go.Bar(
        x=df["ConfidenceScore"],
        y=df["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in df["ConfidenceScore"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Feature Confidence Scores",
        xaxis_title="Confidence Score (%)",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        height=max(400, len(df) * 24),
        margin=dict(l=10, r=60, t=40, b=20),
        xaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig


def _plot_correlation_heatmap(
    corr_matrix: pd.DataFrame, title: str = "Feature Correlation Matrix"
) -> go.Figure:
    cols = corr_matrix.columns.tolist()
    if len(cols) > 40:
        cols = cols[:40]
    data = corr_matrix.loc[cols, cols]
    fig = px.imshow(
        data,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=title,
        labels=dict(color="Pearson r"),
        aspect="auto",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        height=max(400, len(cols) * 18 + 100),
        margin=dict(l=5, r=5, t=50, b=5),
    )
    return fig


def _plot_vif_chart(vif_df: pd.DataFrame) -> go.Figure:
    df = vif_df.dropna(subset=["VIF"]).copy()
    if df.empty:
        return go.Figure()
    df = df.sort_values("VIF", ascending=True).tail(30)
    colors = [
        _REC_COLORS["Remove"]      if v > 10
        else _REC_COLORS["Optional"]   if v > 5
        else _REC_COLORS["Recommended"]
        for v in df["VIF"]
    ]
    fig = go.Figure(go.Bar(
        x=df["VIF"], y=df["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in df["VIF"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>VIF: %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=5,  line_dash="dash", line_color=_REC_COLORS["Optional"],
                  annotation_text="Moderate (5)",  annotation_position="top right")
    fig.add_vline(x=10, line_dash="dash", line_color=_REC_COLORS["Remove"],
                  annotation_text="High (10)", annotation_position="top right")
    fig.update_layout(
        title="Variance Inflation Factor (VIF)",
        xaxis_title="VIF",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        height=max(350, len(df) * 22 + 80),
        margin=dict(l=10, r=80, t=40, b=20),
    )
    return fig


def _plot_target_corr_heatmap(corr_with_target: pd.DataFrame) -> go.Figure:
    if corr_with_target.empty:
        return go.Figure()
    data = corr_with_target.head(40)
    fig = px.imshow(
        data,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature–Target Correlation",
        labels=dict(color="Pearson r"),
        aspect="auto",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        height=max(350, len(data) * 22 + 100),
        margin=dict(l=5, r=5, t=50, b=5),
    )
    return fig


def _plot_method_summary(method_results: list) -> go.Figure:
    cat_colors = {
        "Supervised":             _PRIMARY,
        "Feature Importance":     _ACCENT,
        "Intrinsic":              "#8b5cf6",
        "Wrapper":                _WARN,
        "Dimensionality Reduction": "#ec4899",
    }
    names, counts, colors_list = [], [], []
    for r in method_results:
        names.append(r.name)
        counts.append(len(r.selected_features) if r.success else 0)
        colors_list.append(cat_colors.get(r.category, "#94a3b8") if r.success else "#4b5563")

    fig = go.Figure(go.Bar(
        x=names, y=counts,
        marker_color=colors_list,
        text=counts,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Features Selected: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="Features Selected per Method",
        xaxis_tickangle=-35,
        yaxis_title="# Features",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        height=380,
        margin=dict(l=5, r=5, t=50, b=100),
    )
    return fig


def _render_intelligent_feature_selection(df: pd.DataFrame, numeric_cols: list) -> None:
    _section_header(
        "🤖", "Intelligent Auto Feature Selection",
        "Select target Y variables, choose which methods to run, and get consensus-ranked "
        "feature recommendations with full explainability.",
    )

    with st.container():
        # ---- Step 1: Y target selector -----------------------------------
        st.markdown("##### Step 1 — Select Target Variable(s) Y")
        auto_y_cols = st.multiselect(
            "Target KPI column(s)",
            options=numeric_cols,
            default=st.session_state.get("y_cols", []),
            key="ias_y_selector",
        )
        if not auto_y_cols:
            st.info("Select at least one Y target to proceed.")
            return

        candidate_x = [c for c in numeric_cols if c not in auto_y_cols]
        if not candidate_x:
            st.warning("No candidate X features remain after selecting Y.")
            return

        st.markdown(
            f"<p style='color:{_MUTED};font-size:0.85rem'>"
            f"<b style='color:#f8fafc'>{len(candidate_x)}</b> candidate X features &nbsp;|&nbsp; "
            f"<b style='color:#f8fafc'>{len(auto_y_cols)}</b> target(s): "
            f"<code>{'</code>, <code>'.join(auto_y_cols)}</code></p>",
            unsafe_allow_html=True,
        )

        # ---- Step 2: Configuration ----------------------------------------
        st.markdown("##### Step 2 — Configure Analysis")
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            top_k = st.slider(
                "Top-K features per method",
                min_value=3, max_value=min(25, len(candidate_x)),
                value=min(10, len(candidate_x)),
                key="ias_top_k",
            )
        with c2:
            corr_thresh = st.number_input(
                "X–X collinearity flag threshold",
                min_value=0.50, max_value=0.99, value=0.85, step=0.05,
                format="%.2f", key="ias_corr_thresh",
                help="Feature pairs with |r| > threshold are flagged as redundant.",
            )
        with c3:
            vif_thresh = st.number_input(
                "VIF threshold",
                min_value=2.0, max_value=50.0, value=10.0, step=1.0,
                format="%.1f", key="ias_vif_thresh",
                help="VIF above this value flags high multicollinearity.",
            )

        with st.expander("⚙️ Method Selection (defaults auto-tuned for your dataset)", expanded=False):
            n_feat = len(candidate_x)
            from src.feature_selection.auto_selector import (
                _MAX_FEATURES_SFS, _MAX_FEATURES_SFS_BK,
                _XGBOOST_AVAILABLE, _LIGHTGBM_AVAILABLE, _SFS_AVAILABLE,
            )
            default_enabled = [
                "target_correlation", "f_test", "mutual_information",
                "rf_importance", "lasso", "elasticnet", "rfe", "pca_analysis",
            ]
            if n_feat <= _MAX_FEATURES_SFS:
                default_enabled.append("sfs_forward")
            if n_feat <= _MAX_FEATURES_SFS_BK:
                default_enabled.append("sfs_backward")
            if _XGBOOST_AVAILABLE:
                default_enabled.append("xgboost_importance")
            if _LIGHTGBM_AVAILABLE:
                default_enabled.append("lightgbm_importance")

            avail_methods = []
            for mid in ALL_METHOD_IDS:
                if mid == "xgboost_importance" and not _XGBOOST_AVAILABLE:
                    st.caption(f"⚠️ {METHOD_LABELS[mid]} — install `xgboost` to enable")
                    continue
                if mid == "lightgbm_importance" and not _LIGHTGBM_AVAILABLE:
                    st.caption(f"⚠️ {METHOD_LABELS[mid]} — install `lightgbm` to enable")
                    continue
                if mid in ("sfs_forward", "sfs_backward") and not _SFS_AVAILABLE:
                    st.caption(f"⚠️ {METHOD_LABELS[mid]} — upgrade scikit-learn to enable")
                    continue
                avail_methods.append(mid)

            enabled_methods = _method_checkboxes(avail_methods, default_enabled)

        if not enabled_methods:
            st.warning("Select at least one method to run the analysis.")
            return

        st.markdown(f"<p style='color:{_MUTED};font-size:0.85rem'><b style='color:#f8fafc'>{len(enabled_methods)}</b> method(s) selected.</p>", unsafe_allow_html=True)

        # ---- Step 3: Run -------------------------------------------------
        run_btn = st.button("🔍 Run Intelligent Feature Analysis", key="ias_run")

        if run_btn:
            for k in ["_ias_result", "_ias_y_cols", "_ias_top_k", "_ias_methods"]:
                st.session_state.pop(k, None)

            df_num = cast_to_numeric(df)
            X_cand = df_num[candidate_x]
            y_targ = df_num[auto_y_cols]

            progress_placeholder = st.empty()
            progress_bar = st.progress(0)
            steps: List[str] = []

            def progress_cb(msg: str) -> None:
                steps.append(msg)
                progress_placeholder.caption(f"⏳ {msg}")
                progress_bar.progress(min(len(steps) / (len(enabled_methods) + 5), 0.95))

            with st.spinner("Analysing features — this may take 20–60 seconds…"):
                try:
                    result: AutoSelectionResult = run_auto_feature_selection(
                        X_df=X_cand, y_df=y_targ,
                        top_k=top_k,
                        enabled_methods=enabled_methods,
                        corr_threshold=corr_thresh,
                        vif_threshold=vif_thresh,
                        progress_callback=progress_cb,
                    )
                    st.session_state["_ias_result"]  = result
                    st.session_state["_ias_y_cols"]  = auto_y_cols
                    st.session_state["_ias_top_k"]   = top_k
                    st.session_state["_ias_methods"] = enabled_methods
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    progress_bar.empty()
                    progress_placeholder.empty()
                    return

            progress_bar.progress(1.0)
            progress_placeholder.empty()
            progress_bar.empty()
            st.success(f"Analysis complete — {len(enabled_methods)} methods ran on {len(candidate_x)} features.")
            st.rerun()

        # ---- Step 4: Display results (persisted) -------------------------
        if "_ias_result" not in st.session_state:
            return

        result: AutoSelectionResult = st.session_state["_ias_result"]
        res_y:  list                = st.session_state.get("_ias_y_cols", auto_y_cols)
        res_k:  int                 = st.session_state.get("_ias_top_k", top_k)
        cdf  = result.consensus_df
        info = result.dataset_info

        st.markdown(f"---")
        st.markdown(
            f"##### Analysis Results &nbsp; Top-{res_k} features &nbsp;|&nbsp; "
            f"Y = `{', '.join(res_y)}`"
        )

        n_methods_ran = sum(1 for r in result.method_results if r.success)
        n_highly = sum(1 for r in cdf["Recommendation"] if r == "Highly Recommended")
        n_rec    = sum(1 for r in cdf["Recommendation"] if r == "Recommended")
        n_opt    = sum(1 for r in cdf["Recommendation"] if r == "Optional")
        n_rem    = sum(1 for r in cdf["Recommendation"] if r == "Remove")

        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        kc1.metric("Methods Run",          n_methods_ran)
        kc2.metric("🟢 Highly Recommended", n_highly)
        kc3.metric("🔵 Recommended",        n_rec)
        kc4.metric("🟡 Optional",           n_opt)
        kc5.metric("🔴 Remove",             n_rem)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "🏆 Consensus Rankings",
            "📈 Visualizations",
            "🎯 Recommendations",
            "🔬 Method Details",
        ])

        with tab1:
            st.markdown("#### Dataset Quality Report")
            qc1, qc2, qc3, qc4 = st.columns(4)
            qc1.metric("Total Rows",     info.get("n_rows", "?"))
            qc2.metric("Clean Features", info.get("n_clean_features", "?"))
            qc3.metric("Missing % (X)",  f"{info.get('missing_pct_x', 0):.1f}%")
            qc4.metric("Missing % (Y)",  f"{info.get('missing_pct_y', 0):.1f}%")

            if info.get("constant_features"):
                st.warning(
                    f"**{len(info['constant_features'])} constant feature(s) removed** "
                    f"(zero variance): `{', '.join(info['constant_features'])}`"
                )

            st.markdown("#### Method Execution Summary")
            method_rows = [
                {
                    "Method":   r.name,
                    "Category": r.category,
                    "Status":   "✅ Success" if r.success else "❌ Failed",
                    "Selected": len(r.selected_features),
                    "Notes":    r.notes,
                }
                for r in result.method_results
            ]
            st.dataframe(pd.DataFrame(method_rows), use_container_width=True)

            st.markdown("#### Highly Correlated Feature Pairs (|r| > threshold)")
            corr_m = result.correlation_matrix
            pairs = []
            cols_m = corr_m.columns.tolist()
            for i in range(len(cols_m)):
                for j in range(i + 1, len(cols_m)):
                    r_val = corr_m.iloc[i, j]
                    if abs(r_val) > corr_thresh:
                        pairs.append({
                            "Feature A":    cols_m[i],
                            "Feature B":    cols_m[j],
                            "|Pearson r|":  round(abs(r_val), 4),
                        })
            if pairs:
                pairs_df = pd.DataFrame(pairs).sort_values("|Pearson r|", ascending=False)
                st.dataframe(pairs_df, use_container_width=True)
                st.caption(f"{len(pairs)} redundant pair(s) detected.")
            else:
                st.success("No highly correlated feature pairs detected at this threshold.")

        with tab2:
            st.markdown("#### Feature Ranking by Consensus Score")
            st.caption(
                "Confidence Score = 60% × Selection Frequency + 40% × Avg Normalized Score."
            )

            def _style_rec(val: str) -> str:
                color = _REC_COLORS.get(val, "")
                return f"color: {color}; font-weight: bold" if color else ""

            display_cols = [
                "Feature", "SelectionCount", "TotalMethods", "SelectionFreq",
                "ConfidenceScore", "CorrWithTarget", "VIF", "PValue",
                "LassoSelected", "ElasticNetSelected", "Recommendation",
            ]
            disp_df = cdf.reset_index()[display_cols] if "Rank" not in cdf.columns else cdf[display_cols]
            styled = disp_df.style.map(_style_rec, subset=["Recommendation"])
            st.dataframe(styled, use_container_width=True, height=450)

            st.markdown("#### Confidence Score Distribution")
            st.plotly_chart(_plot_consensus_bar(cdf), use_container_width=True)

        with tab3:
            viz1, viz2 = st.columns(2)
            with viz1:
                st.plotly_chart(_plot_correlation_heatmap(result.correlation_matrix), use_container_width=True)
            with viz2:
                st.plotly_chart(_plot_target_corr_heatmap(result.corr_with_target), use_container_width=True)
            viz3, viz4 = st.columns(2)
            with viz3:
                st.plotly_chart(_plot_vif_chart(result.vif_df), use_container_width=True)
            with viz4:
                st.plotly_chart(_plot_method_summary(result.method_results), use_container_width=True)

        with tab4:
            st.markdown("#### Feature Recommendation Cards")
            for rec_cat in ["Highly Recommended", "Recommended", "Optional", "Remove"]:
                feats_in_cat = cdf[cdf["Recommendation"] == rec_cat]
                if feats_in_cat.empty:
                    continue
                color = _REC_COLORS[rec_cat]
                icon  = _REC_ICONS[rec_cat]
                st.markdown(
                    f"<h4 style='color:{color}'>{icon} {rec_cat} ({len(feats_in_cat)} feature(s))</h4>",
                    unsafe_allow_html=True,
                )
                for _, row in feats_in_cat.iterrows():
                    feat     = row["Feature"]
                    conf     = row["ConfidenceScore"]
                    n_sel    = int(row["SelectionCount"])
                    n_tot    = int(row["TotalMethods"])
                    corr_val = row.get("CorrWithTarget")
                    vif_val  = row.get("VIF")
                    with st.expander(
                        f"**{feat}** — Confidence: {conf:.0f}%  ({n_sel}/{n_tot} methods)",
                        expanded=(rec_cat == "Highly Recommended"),
                    ):
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Confidence",        f"{conf:.0f}%")
                        mc2.metric("Methods",           f"{n_sel}/{n_tot}")
                        if corr_val is not None:
                            mc3.metric("Avg |r| w/ Target", f"{corr_val:.3f}")
                        if vif_val is not None:
                            mc4.metric("VIF",               f"{vif_val:.1f}")
                        st.markdown(result.per_feature_reasoning.get(feat, ""))
                        st.markdown("---")

        with tab5:
            st.markdown("#### Per-Method Feature Rankings")
            for r in result.method_results:
                status = "✅" if r.success else "❌"
                with st.expander(
                    f"{status} **{r.name}** — {r.category}  |  "
                    f"{len(r.selected_features)} features  ({r.notes})"
                ):
                    if not r.success:
                        st.error(r.notes)
                        continue
                    rows = [
                        {
                            "Rank":       rank,
                            "Feature":    feat,
                            "Raw Score":  round(r.raw_scores.get(feat, 0), 5),
                            "Norm Score": round(r.all_scores.get(feat, 0), 4),
                        }
                        for rank, feat in enumerate(r.selected_features, 1)
                    ]
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Apply / clear buttons
        st.markdown("---")
        all_keep = result.recommended_features + result.optional_features
        st.markdown(
            f"**Recommended X ({len(result.recommended_features)}):** "
            f"`{', '.join(result.recommended_features) or 'None'}`  \n"
            f"**Optional X ({len(result.optional_features)}):** "
            f"`{', '.join(result.optional_features) or 'None'}`  \n"
            f"**Remove ({len(result.features_to_remove)}):** "
            f"`{', '.join(result.features_to_remove) or 'None'}`"
        )

        ba1, ba2, ba3, ba4 = st.columns(4)
        with ba1:
            if st.button(f"✅ Apply Recommended ({len(result.recommended_features)})", key="ias_apply_rec"):
                st.session_state.x_cols = result.recommended_features
                st.session_state.y_cols = res_y
                _sync_checkboxes(result.recommended_features, res_y, numeric_cols)
                st.success("Applied Highly Recommended + Recommended features as X.")
                st.rerun()
        with ba2:
            if st.button(f"⭐ Apply Rec + Optional ({len(all_keep)})", key="ias_apply_all_keep"):
                st.session_state.x_cols = all_keep
                st.session_state.y_cols = res_y
                _sync_checkboxes(all_keep, res_y, numeric_cols)
                st.success("Applied Recommended + Optional features as X.")
                st.rerun()
        with ba3:
            custom_sel = st.multiselect(
                "Custom X selection",
                options=candidate_x,
                default=result.recommended_features,
                key="ias_custom_x",
            )
            if st.button("Apply Custom Selection", key="ias_apply_custom"):
                st.session_state.x_cols = custom_sel
                st.session_state.y_cols = res_y
                _sync_checkboxes(custom_sel, res_y, numeric_cols)
                st.success(f"Applied {len(custom_sel)} custom X features.")
                st.rerun()
        with ba4:
            if st.button("🗑️ Clear Results", key="ias_clear"):
                for k in ["_ias_result", "_ias_y_cols", "_ias_top_k", "_ias_methods"]:
                    st.session_state.pop(k, None)
                st.rerun()

    st.markdown("---")


# ===========================================================================
# MAIN PAGE RENDERER
# ===========================================================================

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

    if st.session_state.df is None:
        st.warning("Please upload data first in the 'Upload Data' tab.")
        return

    df = cast_to_numeric(st.session_state.df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found in the dataset.")
        return

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Section 2: Data Understanding
    # ------------------------------------------------------------------ #
    _render_data_understanding(df, numeric_cols)

    # ------------------------------------------------------------------ #
    # Section 3: Basic Preprocessing
    # ------------------------------------------------------------------ #
    _render_basic_preprocessing(df, numeric_cols)

    # Refresh df after any cleaning
    df = cast_to_numeric(st.session_state.df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------ #
    # Section 4: Intelligent Auto Feature Selection  (UNCHANGED)
    # ------------------------------------------------------------------ #
    _render_intelligent_feature_selection(df, numeric_cols)

    # ------------------------------------------------------------------ #
    # Section 5: Manual Variable Selection + Final Apply
    # ------------------------------------------------------------------ #
    _section_header(
        "🎛️", "Manual Variable Selection",
        "Confirm or adjust X input features and Y target variables, then finalise preprocessing.",
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
            default_checked = True if select_all_x else col in st.session_state.x_cols
            if st.checkbox(col, value=default_checked, key=f"x_{col}"):
                x_cols.append(col)

    with col_y:
        st.markdown("**Select Target Variables (Y)**")
        y_options = [c for c in numeric_cols if c not in x_cols]
        select_all_y = st.checkbox(
            "Select All Y",
            value=(
                len(st.session_state.y_cols) == len(y_options) and len(y_options) > 0
            ),
            key="sel_all_y",
        )
        y_cols = []
        for col in y_options:
            default_checked = True if select_all_y else col in st.session_state.y_cols
            if st.checkbox(col, value=default_checked, key=f"y_{col}"):
                y_cols.append(col)

    st.markdown("---")

    # Final imputation fallback (in case any NaN remain after cleaning)
    st.markdown("**Fallback Imputation** (for any remaining missing values after cleaning)")
    fi_col1, _ = st.columns([2, 3])
    with fi_col1:
        imputation_method = st.selectbox(
            "Method for remaining NaN",
            ["Mean", "Median", "Zero"],
            help="Applied only to any NaN values that remain after the Basic Preprocessing step.",
        )

    # Apply Preprocessing (split + scale)
    if st.button("🚀 Apply Preprocessing & Split Dataset", use_container_width=False):
        if len(x_cols) == 0 or len(y_cols) == 0:
            st.error("Please select at least one X and one Y variable.")
            return

        st.session_state.x_cols = x_cols
        st.session_state.y_cols = y_cols

        data_x = df[x_cols].copy()
        data_y = df[y_cols].copy()

        st.markdown("#### Feature-wise Statistics (Before Final Scaling)")
        st.dataframe(compute_feature_stats(data_x), use_container_width=True)

        # Fallback imputation for any remaining NaN
        data_x, data_y = impute(data_x, data_y, imputation_method)

        (
            X_train_s, X_test_s,
            y_train_s, y_test_s,
            y_test_raw,
            scaler_x, scaler_y,
        ) = split_and_scale(data_x, data_y)

        st.session_state.X_train    = X_train_s
        st.session_state.X_test     = X_test_s
        st.session_state.y_train    = y_train_s
        st.session_state.y_test     = y_test_s
        st.session_state.y_test_raw = y_test_raw
        st.session_state.scaler_x   = scaler_x
        st.session_state.scaler_y   = scaler_y

        st.success(
            f"Preprocessing complete — **{len(x_cols)}** X features, "
            f"**{len(y_cols)}** Y target(s). "
            "Train/Test split created and StandardScaler applied. "
            "Proceed to the **Train Model** tab."
        )

        st.markdown("#### Feature-wise Statistics (After Scaling — Train Set)")
        train_stats_df = pd.DataFrame(
            X_train_s,
            columns=x_cols,
        )
        st.dataframe(compute_feature_stats(train_stats_df), use_container_width=True)
