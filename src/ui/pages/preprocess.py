"""
src/ui/pages/preprocess.py
===========================
Renders the "Preprocessing" page.

Page flow
---------
Step 1 : Load Dataset          — dataset switcher
Step 2 : Configure Preprocessing
    2a : Data Understanding    — per-feature stats, distribution, outlier profile
    2b : Basic Preprocessing   — remove rows, impute, outliers, domain filters
    2c : Automated Preprocessing — one-click best-defaults pipeline

Output : Preview & Download processed dataset
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
    cast_to_numeric,
    compute_feature_stats,
)

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
_CARD     = "background:rgba(30,41,59,0.7);border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:0.8rem"
_PRIMARY  = "#4da6ff"
_ACCENT   = "#10b981"
_WARN     = "#f59e0b"
_DANGER   = "#ef4444"
_MUTED    = "#94a3b8"


def _section_header(icon: str, title: str, subtitle: str = "") -> None:
    sub = f"<p style='margin:0.25rem 0 0;color:{_MUTED};font-size:0.88rem'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div style='{_CARD}'>"
        f"<h3 style='margin:0;color:{_PRIMARY};font-family:Outfit,sans-serif'>{icon} {title}</h3>"
        f"{sub}</div>",
        unsafe_allow_html=True,
    )


def _step_badge(n: int, label: str) -> None:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:0.75rem;margin:1.2rem 0 0.6rem'>"
        f"<span style='background:{_PRIMARY};color:#0f172a;font-weight:800;font-size:0.9rem;"
        f"border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center'>{n}</span>"
        f"<span style='color:#f8fafc;font-family:Outfit,sans-serif;font-weight:700;font-size:1.05rem'>{label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# HELPERS
# ===========================================================================

def _count_iqr_outliers(series: pd.Series) -> int:
    s = series.dropna()
    if len(s) < 4:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def _zscore_outlier_count(series: pd.Series, thr: float) -> int:
    s = series.dropna()
    if s.std() == 0:
        return 0
    return int(((s - s.mean()).abs() / s.std() > thr).sum())


def _distribution_label(skew: float) -> str:
    if skew > 1.0:   return "Highly right-skewed"
    if skew > 0.5:   return "Moderately right-skewed"
    if skew < -1.0:  return "Highly left-skewed"
    if skew < -0.5:  return "Moderately left-skewed"
    return "Approximately symmetric"


def _apply_zscore_cap(working: pd.DataFrame, cols: List[str], thr: float) -> Tuple[pd.DataFrame, int]:
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


def _apply_winsorization(working: pd.DataFrame, cols: List[str], lo_pct: float, hi_pct: float) -> Tuple[pd.DataFrame, int]:
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


def _apply_capping_flooring(working: pd.DataFrame, cols: List[str], multiplier: float) -> Tuple[pd.DataFrame, int]:
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


def _apply_remove_outliers_iqr(working: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, int]:
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


def _apply_remove_outliers_zscore(working: pd.DataFrame, cols: List[str], thr: float) -> Tuple[pd.DataFrame, int]:
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


# ===========================================================================
# SECTION: DATA UNDERSTANDING
# ===========================================================================

def _render_data_understanding(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    _section_header(
        "🔍", "Data Understanding",
        "Explore any feature in detail — statistics, distribution and outlier profile.",
    )
    with st.container():
        selected_col = st.selectbox("Select a feature to analyse", numeric_cols, key="du_feature_select")
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

        st.markdown(
            f"<p style='color:{_MUTED};font-size:0.82rem;margin-bottom:0.4rem'>"
            f"Data Type: <b style='color:#f8fafc'>{series_raw.dtype}</b> &nbsp;|&nbsp; "
            f"Distribution: <b style='color:#f8fafc'>{_distribution_label(skew_v)}</b></p>",
            unsafe_allow_html=True,
        )

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
                margin=dict(l=5, r=5, t=40, b=5), showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with ch2:
            fig_box = go.Figure(go.Box(
                y=series, name=selected_col,
                marker_color=_PRIMARY, line_color=_PRIMARY, boxmean="sd",
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
# SECTION: BASIC PREPROCESSING
# ===========================================================================

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

    # ---- Tab 1: Remove Records
    with tab_remove:
        st.markdown("#### Remove Rows")
        c1, c2 = st.columns(2)
        with c1:
            remove_missing = st.checkbox("Remove rows with **any** missing values", key="bp_rm_missing")
            if remove_missing:
                st.caption(f"Will remove **{int(df.isnull().any(axis=1).sum())}** row(s) with at least one NaN.")
        with c2:
            remove_dupes = st.checkbox("Remove **duplicate** records", key="bp_rm_dupes")
            if remove_dupes:
                st.caption(f"Will remove **{int(df.duplicated().sum())}** duplicate row(s).")

        st.markdown("---")
        st.markdown("#### Remove Columns")

        # --- Missing columns ---
        _miss_pct = df[numeric_cols].isnull().mean() * 100  # % missing per column
        cm1, cm2 = st.columns([1, 2])
        with cm1:
            remove_miss_cols = st.checkbox(
                "Remove columns with **missing values**",
                key="bp_rm_miss_cols",
                help="Drops columns whose missing % exceeds the threshold below.",
            )
        with cm2:
            miss_col_thr = st.number_input(
                "Missing % threshold (drop if ≥ this %)",
                min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                format="%.1f", key="bp_miss_col_thr",
                help="0% = drop any column with even one missing value. 50% = only drop columns where more than half are missing.",
                disabled=not st.session_state.get("bp_rm_miss_cols", False),
            )
        if remove_miss_cols:
            _miss_drop_cols = [c for c in numeric_cols if _miss_pct[c] >= miss_col_thr]
            if _miss_drop_cols:
                st.caption(
                    f"Will remove **{len(_miss_drop_cols)}** column(s) with ≥ {miss_col_thr:.0f}% missing: "
                    f"`{', '.join(_miss_drop_cols)}`"
                )
            else:
                st.caption(f"No columns with ≥ {miss_col_thr:.0f}% missing found.")

        st.markdown("")
        _const_cols = [c for c in numeric_cols if df[c].std() == 0]

        c3, _ = st.columns(2)
        with c3:
            remove_const_cols = st.checkbox(
                "Remove **constant** columns (std = 0)",
                key="bp_rm_const_cols",
                help="Drops columns where every row has the same value, including all-zero columns — these carry no predictive information.",
            )
            if remove_const_cols:
                if _const_cols:
                    st.caption(
                        f"Will remove **{len(_const_cols)}** column(s): `{', '.join(_const_cols)}`"
                    )
                else:
                    st.caption("No constant columns found in the dataset.")

        st.markdown("")
        c5, c6 = st.columns([1, 2])
        with c5:
            remove_nzv_cols = st.checkbox(
                "Remove **near-zero variance** columns",
                key="bp_rm_nzv_cols",
                help="Drops columns whose std is above 0 but below the threshold — barely varying sensors add noise without signal.",
            )
        with c6:
            nzv_threshold = st.number_input(
                "Std threshold",
                min_value=0.0001, max_value=1.0, value=0.01, step=0.001,
                format="%.4f", key="bp_nzv_thr",
                help="Columns with std < this value are considered near-zero variance.",
                disabled=not st.session_state.get("bp_rm_nzv_cols", False),
            )
        if remove_nzv_cols:
            _nzv_cols = [
                c for c in numeric_cols
                if 0 < df[c].std() < nzv_threshold
            ]
            if _nzv_cols:
                st.caption(
                    f"Will remove **{len(_nzv_cols)}** near-zero variance column(s) "
                    f"(std < {nzv_threshold}): `{', '.join(_nzv_cols)}`"
                )
            else:
                st.caption(f"No columns with 0 < std < {nzv_threshold} found.")

    # ---- Tab 2: Missing Values
    with tab_impute:
        st.markdown("#### Handle Missing Values")
        impute_cols = st.multiselect(
            "Apply to columns", numeric_cols, default=[], key="bp_impute_cols",
            help="Leave empty to apply to all numeric columns with missing values.",
        )
        impute_target = impute_cols if impute_cols else [c for c in numeric_cols if df[c].isnull().any()]
        if not impute_cols:
            st.caption(f"No columns selected — will apply to all {len(impute_target)} column(s) with missing values.")

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
                custom_fill_val = st.number_input("Fill Value", value=0.0, format="%.4f", key="bp_custom_val")

        if impute_method != "None":
            total_missing = sum(df[c].isnull().sum() for c in impute_target)
            st.caption(
                f"**{impute_method}** will fill **{total_missing}** missing value(s) across {len(impute_target)} column(s)."
            )

    # ---- Tab 3: Outlier Treatment
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
            zscore_thr = 3.0
            wins_lo    = 2.5
            wins_hi    = 97.5
            cap_mult   = 1.5
            if "Z-Score" in outlier_method:
                zscore_thr = st.number_input("Z-Score threshold", min_value=1.0, max_value=10.0, value=3.0, step=0.5, key="bp_zscore_thr")
            elif "Winsorization" in outlier_method:
                wins_lo = st.number_input("Lower %", min_value=0.1, max_value=10.0, value=2.5, step=0.5, key="bp_wins_lo")
                wins_hi = st.number_input("Upper %", min_value=90.0, max_value=99.9, value=97.5, step=0.5, key="bp_wins_hi")
            elif "custom IQR" in outlier_method:
                cap_mult = st.number_input("IQR multiplier", min_value=0.5, max_value=5.0, value=1.5, step=0.5, key="bp_cap_mult")

        outlier_cols = st.multiselect("Apply to columns", numeric_cols, default=numeric_cols, key="bp_outlier_cols")

        if outlier_method != "None" and outlier_cols:
            preview_rows = []
            for col in outlier_cols[:15]:
                n_iqr = _count_iqr_outliers(df[col])
                n_z   = _zscore_outlier_count(df[col], zscore_thr)
                if n_iqr > 0 or n_z > 0:
                    preview_rows.append({"Column": col, "IQR Outliers": n_iqr, f"Z-Score (>{zscore_thr})": n_z})
            if preview_rows:
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)
            else:
                st.success("No outliers detected in selected columns.")

    # ---- Tab 4: Domain Filters
    with tab_filter:
        st.markdown("#### Domain-Based Min / Max Filtering")
        st.caption("Clip feature values to physically meaningful bounds.")
        filter_tags = st.multiselect("Select features to filter", numeric_cols, default=[], key="bp_filter_tags")

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
                        f"<p style='color:{_MUTED};font-size:0.8rem;margin-top:0'>Data range: {tag_min:.4g} — {tag_max:.4g}</p>",
                        unsafe_allow_html=True,
                    )
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        umin = st.number_input("Min", value=tag_min, format="%.4f", key=f"bp_fmin_{tag}")
                    with fc2:
                        umax = st.number_input("Max", value=tag_max, format="%.4f", key=f"bp_fmax_{tag}")
                    domain_filters[tag] = {"min": umin, "max": umax}

    # ---- Apply Cleaning button
    st.markdown("---")
    active_steps = []
    if st.session_state.get("bp_rm_missing"):
        active_steps.append("Remove missing rows")
    if st.session_state.get("bp_rm_dupes"):
        active_steps.append("Remove duplicates")
    if st.session_state.get("bp_rm_miss_cols"):
        active_steps.append(f"Remove columns (missing ≥ {st.session_state.get('bp_miss_col_thr', 0.0):.0f}%)")
    if st.session_state.get("bp_rm_const_cols"):
        active_steps.append("Remove constant columns")
    if st.session_state.get("bp_rm_nzv_cols"):
        active_steps.append(f"Remove NZV columns (std < {st.session_state.get('bp_nzv_thr', 0.01)})")
    if st.session_state.get("bp_impute_method", "None") != "None":
        active_steps.append(f"Impute: {st.session_state.get('bp_impute_method')}")
    if st.session_state.get("bp_outlier_method", "None") != "None":
        active_steps.append(f"Outliers: {st.session_state.get('bp_outlier_method','').split(' (')[0]}")
    if domain_filters:
        active_steps.append(f"Domain filters: {len(domain_filters)} tag(s)")

    if active_steps:
        st.markdown(
            "<p style='color:" + _MUTED + ";font-size:0.85rem'>Configured steps: "
            + " → ".join(f"<b style='color:#f8fafc'>{s}</b>" for s in active_steps) + "</p>",
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

        if st.session_state.get("bp_rm_missing"):
            n_before = len(working)
            working = working.dropna().reset_index(drop=True)
            action_log.append(f"Removed **{n_before - len(working)}** row(s) with missing values.")

        if st.session_state.get("bp_rm_dupes"):
            n_before = len(working)
            working = working.drop_duplicates().reset_index(drop=True)
            action_log.append(f"Removed **{n_before - len(working)}** duplicate row(s).")

        if st.session_state.get("bp_rm_miss_cols"):
            _miss_thr = float(st.session_state.get("bp_miss_col_thr", 0.0))
            num_cols_now = working.select_dtypes(include=[np.number]).columns.tolist()
            miss_drop = [
                c for c in num_cols_now
                if working[c].isnull().mean() * 100 >= _miss_thr
            ]
            if miss_drop:
                working = working.drop(columns=miss_drop)
                action_log.append(
                    f"Removed **{len(miss_drop)}** column(s) with ≥ {_miss_thr:.0f}% missing values: "
                    f"`{', '.join(miss_drop)}`."
                )

        if st.session_state.get("bp_rm_const_cols"):
            num_cols_now = working.select_dtypes(include=[np.number]).columns.tolist()
            const_drop = [c for c in num_cols_now if working[c].std() == 0]
            if const_drop:
                working = working.drop(columns=const_drop)
                action_log.append(
                    f"Removed **{len(const_drop)}** constant column(s) (std = 0): `{', '.join(const_drop)}`."
                )

        if st.session_state.get("bp_rm_nzv_cols"):
            _nzv_thr = float(st.session_state.get("bp_nzv_thr", 0.01))
            nzv_drop = [
                c for c in working.select_dtypes(include=[np.number]).columns
                if 0 < working[c].std() < _nzv_thr
            ]
            if nzv_drop:
                working = working.drop(columns=nzv_drop)
                action_log.append(
                    f"Removed **{len(nzv_drop)}** near-zero variance column(s) "
                    f"(std < {_nzv_thr}): `{', '.join(nzv_drop)}`."
                )

        _imp_method = st.session_state.get("bp_impute_method", "None")
        if _imp_method != "None":
            _imp_cols = [c for c in (
                st.session_state.get("bp_impute_cols") or
                [c for c in numeric_cols if working[c].isnull().any()]
            ) if c in working.columns]
            n_filled = 0
            for col in _imp_cols:
                n_miss = working[col].isnull().sum()
                if n_miss == 0:
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
                n_filled += n_miss
            action_log.append(f"Imputed **{n_filled}** missing value(s) using **{_imp_method}** across {len(_imp_cols)} column(s).")

        _out_method = st.session_state.get("bp_outlier_method", "None")
        _out_cols   = [c for c in (st.session_state.get("bp_outlier_cols") or []) if c in working.columns]
        if _out_method != "None" and _out_cols:
            if _out_method == "IQR Capping (Q1-1.5×IQR, Q3+1.5×IQR)":
                working, n = _apply_capping_flooring(working, _out_cols, 1.5)
                action_log.append(f"IQR capping applied — **{n}** value(s) capped.")
            elif "Z-Score Capping" in _out_method:
                working, n = _apply_zscore_cap(working, _out_cols, zscore_thr)
                action_log.append(f"Z-Score capping (thr={zscore_thr}) — **{n}** value(s) capped.")
            elif "Winsorization" in _out_method:
                working, n = _apply_winsorization(working, _out_cols, wins_lo, wins_hi)
                action_log.append(f"Winsorization ({wins_lo:.1f}%–{wins_hi:.1f}%) — **{n}** value(s) capped.")
            elif "custom IQR" in _out_method:
                working, n = _apply_capping_flooring(working, _out_cols, cap_mult)
                action_log.append(f"IQR capping (×{cap_mult}) — **{n}** value(s) capped.")
            elif _out_method == "Remove Outliers (IQR)":
                working, n = _apply_remove_outliers_iqr(working, _out_cols)
                action_log.append(f"Removed **{n}** outlier row(s) via IQR.")
            elif "Remove Outliers (Z-Score)" in _out_method:
                working, n = _apply_remove_outliers_zscore(working, _out_cols, zscore_thr)
                action_log.append(f"Removed **{n}** outlier row(s) via Z-Score (thr={zscore_thr}).")

        if domain_filters:
            for tag, bounds in domain_filters.items():
                if tag in working.columns:
                    working[tag] = working[tag].clip(bounds["min"], bounds["max"])
            action_log.append(f"Domain filters applied to **{len(domain_filters)}** tag(s).")

        after_rows = len(working)
        st.session_state.df = working
        st.success(f"Cleaning complete. Records: **{before_rows}** → **{after_rows}** ({before_rows - after_rows} row(s) removed).")
        for msg in action_log:
            st.markdown(f"- {msg}")
        ba1, ba2 = st.columns(2)
        ba1.metric("Before", before_rows)
        ba2.metric("After", after_rows, delta=f"{after_rows - before_rows:+d} rows", delta_color="inverse")
        st.rerun()

    st.markdown("---")


# ===========================================================================
# SECTION: AUTOMATED PREPROCESSING
# ===========================================================================

def _render_automated_preprocessing(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    _section_header(
        "🤖", "Automated Preprocessing",
        "One click — runs the optimal cleaning pipeline with best-default settings and shows each step applied.",
    )

    st.markdown(
        f"<p style='color:{_MUTED};font-size:0.88rem'>"
        "Best-default pipeline: <b style='color:#f8fafc'>Cast to numeric</b> → "
        "<b style='color:#f8fafc'>Remove duplicates</b> → "
        "<b style='color:#f8fafc'>Drop high-missing columns (≥50%)</b> → "
        "<b style='color:#f8fafc'>Remove constant columns</b> → "
        "<b style='color:#f8fafc'>Remove near-zero variance columns (std&lt;0.01)</b> → "
        "<b style='color:#f8fafc'>Median imputation</b> → "
        "<b style='color:#f8fafc'>IQR capping (1.5×)</b>"
        "</p>",
        unsafe_allow_html=True,
    )

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_auto = st.button("▶ Run Preprocessing", key="auto_prep_run", use_container_width=True)

    if run_auto:
        log_placeholder = st.empty()
        progress_bar    = st.progress(0)
        steps_log: List[str] = []

        def _log(msg: str, pct: float) -> None:
            steps_log.append(msg)
            log_placeholder.markdown("\n".join(f"- {m}" for m in steps_log))
            progress_bar.progress(pct)

        working = cast_to_numeric(st.session_state.df).copy()
        n_rows_start = len(working)
        _log(f"✅ **Cast to numeric** — {working.shape[1]} columns processed.", 0.14)

        n_before = len(working)
        working = working.drop_duplicates().reset_index(drop=True)
        n_dupes = n_before - len(working)
        _log(f"✅ **Remove duplicates** — {n_dupes} duplicate row(s) removed.", 0.28)

        auto_num_cols = working.select_dtypes(include=[np.number]).columns.tolist()
        miss_auto = [c for c in auto_num_cols if working[c].isnull().mean() >= 0.50]
        if miss_auto:
            working = working.drop(columns=miss_auto)
        _log(
            f"✅ **Drop high-missing columns (≥50%)** — {len(miss_auto)} column(s) removed"
            + (f": `{', '.join(miss_auto)}`" if miss_auto else " (none found)") + ".",
            0.42,
        )

        auto_num_cols = working.select_dtypes(include=[np.number]).columns.tolist()
        const_auto = [c for c in auto_num_cols if working[c].std() == 0]
        if const_auto:
            working = working.drop(columns=const_auto)
        _log(
            f"✅ **Remove constant columns** — {len(const_auto)} column(s) removed"
            + (f": `{', '.join(const_auto)}`" if const_auto else " (none found)") + ".",
            0.56,
        )

        auto_num_cols = working.select_dtypes(include=[np.number]).columns.tolist()
        nzv_auto = [c for c in auto_num_cols if 0 < working[c].std() < 0.01]
        if nzv_auto:
            working = working.drop(columns=nzv_auto)
        _log(
            f"✅ **Remove near-zero variance columns (std < 0.01)** — {len(nzv_auto)} column(s) removed"
            + (f": `{', '.join(nzv_auto)}`" if nzv_auto else " (none found)") + ".",
            0.70,
        )

        n_filled = 0
        for col in working.select_dtypes(include=[np.number]).columns:
            if col in working.columns:
                n_miss = int(working[col].isnull().sum())
                if n_miss > 0:
                    working[col] = working[col].fillna(working[col].median())
                    n_filled += n_miss
        _log(f"✅ **Median imputation** — {n_filled} missing value(s) filled across {working.shape[1]} columns.", 0.84)

        n_capped = 0
        for col in [c for c in working.select_dtypes(include=[np.number]).columns]:
            s = working[col]
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((s < lo) | (s > hi)).sum())
            working[col] = s.clip(lo, hi)
            n_capped += n
        _log(f"✅ **IQR capping (1.5×)** — {n_capped} value(s) capped across all numeric columns.", 0.98)

        progress_bar.progress(1.0)
        st.session_state.df = working
        st.session_state["_auto_prep_done"] = True

        st.success(
            f"Automated preprocessing complete. "
            f"Rows: **{n_rows_start}** → **{len(working)}**  |  "
            f"Columns: **{working.shape[1]}**"
        )
        st.rerun()

    if st.session_state.get("_auto_prep_done"):
        st.info("Automated preprocessing has been applied. See the dataset preview below.")

    st.markdown("---")


# ===========================================================================
# SECTION: PREVIEW & DOWNLOAD
# ===========================================================================

def _render_output(df: pd.DataFrame) -> None:
    _section_header(
        "📋", "Dataset Preview & Download",
        "Review the processed dataset and download it as CSV.",
    )

    row_count, col_count = df.shape
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Rows",    row_count)
    mc2.metric("Columns", col_count)
    mc3.metric("Missing Values", int(df.isnull().sum().sum()))
    mc4.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))

    n_preview = st.slider("Preview rows", min_value=5, max_value=min(200, row_count), value=min(20, row_count), key="preview_rows_slider")
    st.dataframe(df.head(n_preview), use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Processed Dataset (CSV)",
        data=csv_bytes,
        file_name="processed_dataset.csv",
        mime="text/csv",
        use_container_width=False,
    )

    with st.expander("📊 Feature Statistics", expanded=False):
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.dataframe(compute_feature_stats(numeric_df), use_container_width=True)

    st.markdown("---")


# ===========================================================================
# MAIN PAGE RENDERER
# ===========================================================================

def render() -> None:
    st.title("Data Preprocessing")

    # ------------------------------------------------------------------
    # Step 1: Load Dataset
    # ------------------------------------------------------------------
    _step_badge(1, "Load Dataset")
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
                    st.session_state.pop("_auto_prep_done", None)
                    st.success(f"Dataset switched to {history_file_prep}")
                    st.rerun()
                else:
                    st.error(
                        f"Could not load **{history_file_prep}** — the stored data may be "
                        "incompatible. Go to Upload Data, delete this entry, and re-upload."
                    )
    else:
        st.info("No datasets found. Go to **Upload Data** to add one.")

    if st.session_state.df is None:
        st.warning("Please upload data first in the 'Upload Data' tab.")
        return

    df = cast_to_numeric(st.session_state.df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found in the dataset.")
        return

    st.markdown("---")

    # ------------------------------------------------------------------
    # Step 2: Configure Preprocessing
    # ------------------------------------------------------------------
    _step_badge(2, "Configure Preprocessing")

    tab_understand, tab_basic, tab_auto = st.tabs([
        "🔍 Data Understanding",
        "⚙️ Basic Preprocessing",
        "🤖 Automated Preprocessing",
    ])

    with tab_understand:
        _render_data_understanding(df, numeric_cols)

    with tab_basic:
        _render_basic_preprocessing(df, numeric_cols)
        # Refresh after cleaning
        df = cast_to_numeric(st.session_state.df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    with tab_auto:
        _render_automated_preprocessing(df, numeric_cols)

    # ------------------------------------------------------------------
    # Output: Preview & Download
    # ------------------------------------------------------------------
    df_current = cast_to_numeric(st.session_state.df)
    _render_output(df_current)
