"""
src/ui/pages/feature_selection.py
===================================
Renders the "Feature Selection" page.

Page flow
---------
Step 1 : Select Target (Y) Variable
         — always visible; feeds both pathways

Pathway A — Configure Feature Selection
    Step 2a : Configure Analysis  (top_k, corr threshold, VIF threshold)
    Step 2b : Methods Selection   (checkboxes per method category)
    Step 2c : Run Intelligent Feature Selection

Pathway B — Automated Feature Selection
    Step 3  : Run with best-default settings (one click)

Step 4 : Analysis Results          (shared; displayed after either pathway)

Manual Variable Selection          (auto-filled from step 2/3 results)
    — add / remove X features, confirm Y features

Final  : Fallback imputation + Apply Preprocessing & Split Dataset
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.preprocessing import (
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
    _MAX_FEATURES_SFS,
    _MAX_FEATURES_SFS_BK,
    _MAX_FEATURES_NEW,
    _XGBOOST_AVAILABLE,
    _LIGHTGBM_AVAILABLE,
    _SHAP_AVAILABLE,
    _SFS_AVAILABLE,
)

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
_CARD    = "background:rgba(30,41,59,0.7);border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:0.8rem"
_PRIMARY = "#4da6ff"
_ACCENT  = "#10b981"
_WARN    = "#f59e0b"
_DANGER  = "#ef4444"
_MUTED   = "#94a3b8"

_REC_COLORS = {
    "Highly Recommended": _ACCENT,
    "Recommended":        _PRIMARY,
    "Consider":           _WARN,
    "Weak Feature":       _DANGER,
    # legacy keys kept for backward compat with any cached results
    "Optional":           _WARN,
    "Remove":             _DANGER,
}
_REC_ICONS = {
    "Highly Recommended": "🟢",
    "Recommended":        "🔵",
    "Consider":           "🟡",
    "Weak Feature":       "🔴",
    "Optional":           "🟡",
    "Remove":             "🔴",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _rec_badge(rec: str) -> str:
    color = _REC_COLORS.get(rec, "#94a3b8")
    icon  = _REC_ICONS.get(rec, "⚪")
    return (
        f"<span style='background:{color};color:#fff;"
        f"padding:2px 10px;border-radius:12px;"
        f"font-size:0.78rem;font-weight:700'>{icon} {rec}</span>"
    )


def _method_checkboxes(available_methods: List[str], default_enabled: List[str]) -> List[str]:
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
                if st.checkbox(METHOD_LABELS[mid], value=mid in default_enabled, key=f"fs_chk_{mid}"):
                    selected.append(mid)
        col_idx += 1
    return selected


def _sync_checkboxes(x_list: List[str], y_list: List[str], all_cols: List[str]) -> None:
    for col in all_cols:
        st.session_state[f"fs_x_{col}"] = col in x_list
    for col in all_cols:
        if col not in x_list:
            st.session_state[f"fs_y_{col}"] = col in y_list


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _plot_consensus_bar(consensus_df: pd.DataFrame) -> go.Figure:
    df = consensus_df.reset_index()
    score_col = "FinalScore" if "FinalScore" in df.columns else "ConfidenceScore"
    df = df.sort_values(score_col, ascending=True).tail(30)
    colors = [_REC_COLORS.get(r, "#94a3b8") for r in df["Recommendation"]]
    fig = go.Figure(go.Bar(
        x=df[score_col], y=df["Feature"], orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in df[score_col]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Final Score: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title="Feature Final Scores", xaxis_title="Final Score",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc", height=max(400, len(df) * 24),
        margin=dict(l=10, r=60, t=40, b=20),
        xaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig


def _plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Feature Correlation Matrix") -> go.Figure:
    cols = corr_matrix.columns.tolist()[:40]
    data = corr_matrix.loc[cols, cols]
    fig = px.imshow(data, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=title, labels=dict(color="Pearson r"), aspect="auto")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc", height=max(400, len(cols) * 18 + 100),
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
        x=df["VIF"], y=df["Feature"], orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in df["VIF"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>VIF: %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=5,  line_dash="dash", line_color=_REC_COLORS["Optional"],
                  annotation_text="Moderate (5)",  annotation_position="top right")
    fig.add_vline(x=10, line_dash="dash", line_color=_REC_COLORS["Remove"],
                  annotation_text="High (10)", annotation_position="top right")
    fig.update_layout(
        title="Variance Inflation Factor (VIF)", xaxis_title="VIF",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc", height=max(350, len(df) * 22 + 80),
        margin=dict(l=10, r=80, t=40, b=20),
    )
    return fig


def _plot_target_corr_heatmap(corr_with_target: pd.DataFrame) -> go.Figure:
    if corr_with_target.empty:
        return go.Figure()
    data = corr_with_target.head(40)
    fig = px.imshow(data, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Feature–Target Correlation", labels=dict(color="Pearson r"), aspect="auto")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc", height=max(350, len(data) * 22 + 100),
        margin=dict(l=5, r=5, t=50, b=5),
    )
    return fig


def _plot_method_summary(method_results: list) -> go.Figure:
    cat_colors = {
        "Supervised": _PRIMARY, "Feature Importance": _ACCENT,
        "Intrinsic": "#8b5cf6", "Wrapper": _WARN, "Dimensionality Reduction": "#ec4899",
    }
    names, counts, colors_list = [], [], []
    for r in method_results:
        names.append(r.name)
        counts.append(len(r.selected_features) if r.success else 0)
        colors_list.append(cat_colors.get(r.category, "#94a3b8") if r.success else "#4b5563")
    fig = go.Figure(go.Bar(
        x=names, y=counts, marker_color=colors_list,
        text=counts, textposition="outside",
        hovertemplate="<b>%{x}</b><br>Features Selected: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="Features Selected per Method", xaxis_tickangle=-35, yaxis_title="# Features",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc", height=380, margin=dict(l=5, r=5, t=50, b=100),
    )
    return fig


# ---------------------------------------------------------------------------
# Build default method list
# ---------------------------------------------------------------------------

def _build_default_methods(n_feat: int) -> List[str]:
    defaults = [
        "target_correlation", "f_test", "mutual_information",
        "mrmr", "rf_importance", "lasso", "elasticnet", "rfe", "pca_analysis",
    ]
    if n_feat <= _MAX_FEATURES_SFS and _SFS_AVAILABLE:
        defaults.append("sfs_forward")
    if n_feat <= _MAX_FEATURES_SFS_BK and _SFS_AVAILABLE:
        defaults.append("sfs_backward")
    if _XGBOOST_AVAILABLE:
        defaults.append("xgboost_importance")
    if _LIGHTGBM_AVAILABLE:
        defaults.append("lightgbm_importance")
    if n_feat <= _MAX_FEATURES_NEW:
        defaults.append("permutation_importance")
        if _SHAP_AVAILABLE:
            defaults.append("shap_importance")
    return defaults


def _build_available_methods() -> List[str]:
    avail = []
    for mid in ALL_METHOD_IDS:
        if mid == "xgboost_importance" and not _XGBOOST_AVAILABLE:
            continue
        if mid == "lightgbm_importance" and not _LIGHTGBM_AVAILABLE:
            continue
        if mid in ("sfs_forward", "sfs_backward") and not _SFS_AVAILABLE:
            continue
        avail.append(mid)
    return avail


# ---------------------------------------------------------------------------
# Step 4: Analysis Results renderer  (shared by both pathways)
# ---------------------------------------------------------------------------

def _render_analysis_results(
    result: AutoSelectionResult,
    res_y: List[str],
    res_k: int,
    candidate_x: List[str],
    corr_thresh: float,
    numeric_cols: List[str],
) -> None:
    cdf  = result.consensus_df
    info = result.dataset_info

    n_methods_ran = sum(1 for r in result.method_results if r.success)
    n_highly = sum(1 for r in cdf["Recommendation"] if r == "Highly Recommended")
    n_rec    = sum(1 for r in cdf["Recommendation"] if r == "Recommended")
    n_opt    = sum(1 for r in cdf["Recommendation"] if r in ("Consider", "Optional"))
    n_rem    = sum(1 for r in cdf["Recommendation"] if r in ("Weak Feature", "Remove"))

    st.markdown(
        f"<p style='color:{_MUTED};font-size:0.88rem'>"
        f"Top-{res_k} features &nbsp;|&nbsp; Y = <code>{'</code>, <code>'.join(res_y)}</code></p>",
        unsafe_allow_html=True,
    )

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric("Methods Run",          n_methods_ran)
    kc2.metric("🟢 Highly Recommended", n_highly)
    kc3.metric("🔵 Recommended",        n_rec)
    kc4.metric("🟡 Consider",           n_opt)
    kc5.metric("🔴 Weak Feature",       n_rem)

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
            {"Method": r.name, "Category": r.category,
             "Status": "✅ Success" if r.success else "❌ Failed",
             "Selected": len(r.selected_features), "Notes": r.notes}
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
                    pairs.append({"Feature A": cols_m[i], "Feature B": cols_m[j], "|Pearson r|": round(abs(r_val), 4)})
        if pairs:
            st.dataframe(pd.DataFrame(pairs).sort_values("|Pearson r|", ascending=False), use_container_width=True)
            st.caption(f"{len(pairs)} redundant pair(s) detected.")
        else:
            st.success("No highly correlated feature pairs detected at this threshold.")

    with tab2:
        st.markdown("#### Feature Ranking by Final Score")
        st.caption(
            "**Final Score** = 25% × Selection Frequency + 40% × Predictive Strength "
            "+ 20% × Feature Quality + 15% × Stability Score"
        )

        def _style_rec(val: str) -> str:
            color = _REC_COLORS.get(val, "")
            return f"color: {color}; font-weight: bold" if color else ""

        base_cols = [
            "Feature", "SelectionCount", "TotalMethods", "SelectionFreq",
            "PredictiveStrength", "FeatureQuality", "StabilityScore", "FinalScore",
            "CorrWithTarget", "VIF", "PValue",
            "LassoSelected", "ElasticNetSelected", "Recommendation",
        ]
        available_cols = [c for c in base_cols if c in cdf.columns]
        disp_df = cdf.reset_index()[available_cols] if "Rank" not in cdf.columns else cdf[available_cols]
        st.dataframe(disp_df.style.map(_style_rec, subset=["Recommendation"]), use_container_width=True, height=450)
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
        for rec_cat in ["Highly Recommended", "Recommended", "Consider", "Weak Feature"]:
            # also pick up legacy category names from cached results
            legacy = {"Consider": "Optional", "Weak Feature": "Remove"}
            feats_in_cat = cdf[cdf["Recommendation"].isin([rec_cat, legacy.get(rec_cat, "")])]
            if feats_in_cat.empty:
                continue
            color = _REC_COLORS[rec_cat]
            icon  = _REC_ICONS[rec_cat]
            st.markdown(f"<h4 style='color:{color}'>{icon} {rec_cat} ({len(feats_in_cat)} feature(s))</h4>", unsafe_allow_html=True)
            for _, row in feats_in_cat.iterrows():
                feat       = row["Feature"]
                final_score = row.get("FinalScore", row.get("ConfidenceScore", 0))
                ps_val     = row.get("PredictiveStrength")
                fq_val     = row.get("FeatureQuality")
                stab_val   = row.get("StabilityScore")
                n_sel      = int(row["SelectionCount"])
                n_tot      = int(row["TotalMethods"])
                with st.expander(f"**{feat}** — Final Score: {final_score:.0f}  ({n_sel}/{n_tot} methods)", expanded=(rec_cat == "Highly Recommended")):
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Final Score",         f"{final_score:.0f}")
                    mc2.metric("Methods",             f"{n_sel}/{n_tot}")
                    corr_val = row.get("CorrWithTarget")
                    vif_val  = row.get("VIF")
                    if corr_val is not None:
                        mc3.metric("Avg |r| w/ Target", f"{corr_val:.3f}")
                    if vif_val is not None:
                        mc4.metric("VIF", f"{vif_val:.1f}")
                    # Score breakdown row
                    if ps_val is not None:
                        sb1, sb2, sb3 = st.columns(3)
                        sb1.metric("Predictive Strength", f"{ps_val:.1f}")
                        if fq_val is not None:
                            sb2.metric("Feature Quality", f"{fq_val:.1f}")
                        if stab_val is not None:
                            sb3.metric("Stability Score", f"{stab_val:.1f}")
                    st.markdown(result.per_feature_reasoning.get(feat, ""))
                    st.markdown("---")

    with tab5:
        st.markdown("#### Per-Method Feature Rankings")
        for r in result.method_results:
            status = "✅" if r.success else "❌"
            with st.expander(f"{status} **{r.name}** — {r.category}  |  {len(r.selected_features)} features  ({r.notes})"):
                if not r.success:
                    st.error(r.notes)
                    continue
                rows = [
                    {"Rank": rank, "Feature": feat,
                     "Raw Score": round(r.raw_scores.get(feat, 0), 5),
                     "Norm Score": round(r.all_scores.get(feat, 0), 4)}
                    for rank, feat in enumerate(r.selected_features, 1)
                ]
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")


# ---------------------------------------------------------------------------
# Manual Variable Selection  (auto-filled, allows add/remove)
# ---------------------------------------------------------------------------

def _render_manual_variable_selection(
    suggested_x: List[str],
    suggested_y: List[str],
    numeric_cols: List[str],
) -> tuple[List[str], List[str]]:
    _section_header(
        "🎛️", "Manual Variable Selection",
        "Review and adjust the auto-selected X input features and Y target variables.",
    )

    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown("**Input Features (X)**")
        st.caption(f"{len(suggested_x)} feature(s) auto-selected — check/uncheck to adjust.")
        select_all_x = st.checkbox(
            "Select All X",
            value=len(suggested_x) == len(numeric_cols),
            key="fs_sel_all_x",
        )
        x_cols: List[str] = []
        for col in numeric_cols:
            default_checked = True if select_all_x else col in suggested_x
            if st.checkbox(col, value=default_checked, key=f"fs_x_{col}"):
                x_cols.append(col)

    with col_y:
        st.markdown("**Target Variables (Y)**")
        y_options = [c for c in numeric_cols if c not in x_cols]
        select_all_y = st.checkbox(
            "Select All Y",
            value=(len(suggested_y) == len(y_options) and len(y_options) > 0),
            key="fs_sel_all_y",
        )
        y_cols: List[str] = []
        for col in y_options:
            default_checked = True if select_all_y else col in suggested_y
            if st.checkbox(col, value=default_checked, key=f"fs_y_{col}"):
                y_cols.append(col)

    if x_cols:
        st.markdown(
            f"<p style='color:{_MUTED};font-size:0.85rem'>"
            f"<b style='color:#f8fafc'>{len(x_cols)}</b> X feature(s) &nbsp;|&nbsp; "
            f"<b style='color:#f8fafc'>{len(y_cols)}</b> Y target(s)</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    return x_cols, y_cols


# ---------------------------------------------------------------------------
# Final Apply
# ---------------------------------------------------------------------------

def _render_final_apply(df: pd.DataFrame, x_cols: List[str], y_cols: List[str]) -> None:
    _section_header(
        "🚀", "Apply Preprocessing & Split Dataset",
        "Finalise feature selection, handle remaining missing values, configure the train/test split, and scale.",
    )

    df_num = cast_to_numeric(df)

    # ---- Fallback Imputation ------------------------------------------------
    st.markdown(
        f"<p style='color:{_PRIMARY};font-weight:700;margin-bottom:0.3rem'>Fallback Imputation</p>",
        unsafe_allow_html=True,
    )
    st.caption("Fills any NaN values that remain in the selected X and Y columns after the cleaning steps.")

    # Show how many NaN remain in the chosen columns
    if x_cols and y_cols:
        nan_x = int(df_num[x_cols].isnull().sum().sum())
        nan_y = int(df_num[y_cols].isnull().sum().sum())
        total_nan = nan_x + nan_y
        if total_nan > 0:
            st.warning(
                f"**{total_nan}** missing value(s) detected in selected columns "
                f"(X: {nan_x}, Y: {nan_y}). Fallback imputation will be applied."
            )
        else:
            st.success("No missing values in selected columns — fallback imputation will have no effect.")

    imp_c1, _ = st.columns([2, 3])
    with imp_c1:
        imputation_method = st.selectbox(
            "Method",
            ["Mean", "Median", "Zero"],
            index=1,
            help="Mean: column average  |  Median: column median (robust to outliers)  |  Zero: fill with 0",
            key="fs_fallback_impute",
        )

    st.markdown("---")

    # ---- Split Configuration ------------------------------------------------
    st.markdown(
        f"<p style='color:{_PRIMARY};font-weight:700;margin-bottom:0.3rem'>Dataset Split</p>",
        unsafe_allow_html=True,
    )

    sp_c1, sp_c2 = st.columns([2, 2])
    with sp_c1:
        split_method = st.selectbox(
            "Split Method",
            ["Random Split", "Stratified Split", "Sequential Split"],
            key="fs_split_method",
            help=(
                "Random Split: rows are shuffled and divided at the chosen ratio.  \n"
                "Stratified Split: first Y column is quantile-binned so that the "
                "value distribution is similar in train and test sets — useful when "
                "the target has a skewed or imbalanced distribution.  \n"
                "Sequential Split: preserves chronological row order — first N% of "
                "rows go to training, last M% to testing. Recommended for time-series "
                "and process sensor data to prevent future information leakage."
            ),
        )
    with sp_c2:
        train_ratio = st.slider(
            "Train Ratio",
            min_value=0.50, max_value=0.95, value=0.80, step=0.05,
            format="%.2f",
            key="fs_train_ratio",
            help="Fraction of rows used for training. Remainder goes to the test set.",
        )

    test_ratio = round(1.0 - train_ratio, 2)

    if x_cols and y_cols:
        n_total = len(df_num)
        n_train = int(n_total * train_ratio)
        n_test  = n_total - n_train
        st.markdown(
            f"<p style='color:{_MUTED};font-size:0.85rem'>"
            f"Total rows: <b style='color:#f8fafc'>{n_total}</b> &nbsp;|&nbsp; "
            f"Train: <b style='color:{_ACCENT}'>{n_train}</b> ({train_ratio*100:.0f}%) &nbsp;|&nbsp; "
            f"Test: <b style='color:{_WARN}'>{n_test}</b> ({test_ratio*100:.0f}%)"
            "</p>",
            unsafe_allow_html=True,
        )
        if split_method == "Stratified Split":
            st.caption(
                f"Stratification: first Y column (`{y_cols[0]}`) will be binned into "
                "5 equal-frequency quantile groups to ensure balanced distribution across splits."
            )
        elif split_method == "Sequential Split":
            st.caption(
                f"Sequential: rows 1–{n_train} → Train | rows {n_train+1}–{n_total} → Test. "
                "Row order is preserved; no shuffling."
            )

    st.markdown("---")

    # ---- Apply button -------------------------------------------------------
    if st.button("🚀 Apply Preprocessing & Split Dataset", key="fs_apply_final", use_container_width=False):
        if not x_cols or not y_cols:
            st.error("Select at least one X feature and one Y target.")
            return

        st.session_state.x_cols = x_cols
        st.session_state.y_cols = y_cols

        data_x = df_num[x_cols].copy()
        data_y = df_num[y_cols].copy()

        st.markdown("#### Feature-wise Statistics (Before Final Scaling)")
        st.dataframe(compute_feature_stats(data_x), use_container_width=True)

        data_x, data_y = impute(data_x, data_y, imputation_method)

        stratify_bins = 5 if split_method == "Stratified Split" else 0
        seq_split = split_method == "Sequential Split"

        (
            X_train_s, X_test_s,
            y_train_s, y_test_s,
            y_test_raw,
            scaler_x, scaler_y,
        ) = split_and_scale(
            data_x, data_y,
            test_size=test_ratio,
            stratify_bins=stratify_bins,
            split_method="sequential" if seq_split else "random",
        )

        st.session_state.X_train    = X_train_s
        st.session_state.X_test     = X_test_s
        st.session_state.y_train    = y_train_s
        st.session_state.y_test     = y_test_s
        st.session_state.y_test_raw = y_test_raw
        st.session_state.scaler_x   = scaler_x
        st.session_state.scaler_y   = scaler_y

        n_train_actual = X_train_s.shape[0]
        n_test_actual  = X_test_s.shape[0]
        split_label = split_method
        if seq_split:
            split_label = "Sequential Split (rows 1–{} train, {}–{} test)".format(
                n_train_actual, n_train_actual + 1, n_train_actual + n_test_actual
            )
        st.success(
            f"Preprocessing complete — **{len(x_cols)}** X features, **{len(y_cols)}** Y target(s).  \n"
            f"Split: **{n_train_actual}** train rows / **{n_test_actual}** test rows "
            f"({split_label}, {train_ratio*100:.0f}/{test_ratio*100:.0f}).  \n"
            "StandardScaler applied (fitted on train only). Proceed to the **Train Model** tab."
        )

        st.markdown("#### Feature-wise Statistics (After Scaling — Train Set)")
        st.dataframe(
            compute_feature_stats(pd.DataFrame(X_train_s, columns=x_cols)),
            use_container_width=True,
        )


# ===========================================================================
# MAIN PAGE RENDERER
# ===========================================================================

def render() -> None:
    st.title("Feature Selection")

    if st.session_state.df is None:
        st.warning(
            "No dataset loaded. Complete **Preprocessing** first, then return here."
        )
        return

    df = cast_to_numeric(st.session_state.df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found in the dataset.")
        return

    # ------------------------------------------------------------------
    # Step 1: Select Target (Y) Variable
    # ------------------------------------------------------------------
    _step_badge(1, "Select Target (Y) Variable")
    _section_header(
        "🎯", "Target Variable Selection",
        "Choose one or more columns to predict. These become Y; all others are candidate X features.",
    )

    auto_y_cols = st.multiselect(
        "Target KPI column(s)",
        options=numeric_cols,
        default=st.session_state.get("y_cols", []),
        key="fs_y_selector",
    )

    if not auto_y_cols:
        st.info("Select at least one target (Y) variable to proceed.")
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
    st.markdown("---")

    # ------------------------------------------------------------------
    # Pathway choice
    # ------------------------------------------------------------------
    st.markdown(
        f"<h4 style='color:{_PRIMARY};font-family:Outfit,sans-serif;margin-bottom:0.5rem'>"
        "Choose Feature Selection Mode</h4>",
        unsafe_allow_html=True,
    )

    pathway = st.radio(
        "Mode",
        ["🔧 Configure Feature Selection", "⚡ Automated Feature Selection"],
        horizontal=True,
        label_visibility="collapsed",
        key="fs_pathway",
    )
    st.markdown("---")

    avail_methods   = _build_available_methods()
    default_methods = _build_default_methods(len(candidate_x))
    n_feat          = len(candidate_x)

    # ------------------------------------------------------------------
    # Pathway A — Configure Feature Selection
    # ------------------------------------------------------------------
    if pathway == "🔧 Configure Feature Selection":

        _step_badge(2, "Configure Feature Selection")

        tab_configure, tab_methods = st.tabs([
            "⚙️ Configure Analysis",
            "📋 Methods Selection",
        ])

        with tab_configure:
            _section_header("⚙️", "Configure Analysis", "Set top-K, collinearity and VIF thresholds.")
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                top_k = st.slider(
                    "Top-K features per method",
                    min_value=2, max_value=min(25, n_feat),
                    value=min(10, n_feat), key="fs_top_k",
                )
            with c2:
                corr_thresh = st.number_input(
                    "X–X collinearity flag threshold",
                    min_value=0.50, max_value=0.99, value=0.85, step=0.05,
                    format="%.2f", key="fs_corr_thresh",
                    help="Feature pairs with |r| > threshold are flagged as redundant.",
                )
            with c3:
                vif_thresh = st.number_input(
                    "VIF threshold",
                    min_value=2.0, max_value=50.0, value=10.0, step=1.0,
                    format="%.1f", key="fs_vif_thresh",
                    help="VIF above this value flags high multicollinearity.",
                )

        with tab_methods:
            _section_header("📋", "Methods Selection", "Enable / disable individual feature selection methods.")

            for mid in ALL_METHOD_IDS:
                if mid == "xgboost_importance" and not _XGBOOST_AVAILABLE:
                    st.caption(f"⚠️ {METHOD_LABELS[mid]} — install `xgboost` to enable")
                elif mid == "lightgbm_importance" and not _LIGHTGBM_AVAILABLE:
                    st.caption(f"⚠️ {METHOD_LABELS[mid]} — install `lightgbm` to enable")
                elif mid in ("sfs_forward", "sfs_backward") and not _SFS_AVAILABLE:
                    st.caption(f"⚠️ {METHOD_LABELS[mid]} — upgrade scikit-learn to enable")
                elif mid == "shap_importance" and not _SHAP_AVAILABLE:
                    st.caption(f"⚠️ {METHOD_LABELS[mid]} — install `shap` to enable")

            enabled_methods = _method_checkboxes(avail_methods, default_methods)
            st.markdown(
                f"<p style='color:{_MUTED};font-size:0.85rem'><b style='color:#f8fafc'>"
                f"{len(enabled_methods)}</b> method(s) selected.</p>",
                unsafe_allow_html=True,
            )

        if not st.session_state.get("fs_top_k"):
            top_k       = min(10, n_feat)
            corr_thresh = 0.85
            vif_thresh  = 10.0

        top_k_val       = st.session_state.get("fs_top_k", min(10, n_feat))
        corr_thresh_val = st.session_state.get("fs_corr_thresh", 0.85)
        vif_thresh_val  = st.session_state.get("fs_vif_thresh", 10.0)

        # Run button
        st.markdown("---")
        run_btn = st.button("🔍 Run Intelligent Feature Selection", key="fs_run_configure")

        if run_btn:
            if not enabled_methods:
                st.warning("Select at least one method to run.")
            else:
                for k in ["_fs_result", "_fs_y_cols", "_fs_top_k", "_fs_corr_thresh"]:
                    st.session_state.pop(k, None)

                df_num  = cast_to_numeric(df)
                X_cand  = df_num[candidate_x]
                y_targ  = df_num[auto_y_cols]

                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                steps: List[str] = []

                def _cb(msg: str) -> None:
                    steps.append(msg)
                    progress_placeholder.caption(f"⏳ {msg}")
                    progress_bar.progress(min(len(steps) / (len(enabled_methods) + 5), 0.95))

                with st.spinner("Analysing features — this may take 20–60 seconds…"):
                    try:
                        result = run_auto_feature_selection(
                            X_df=X_cand, y_df=y_targ,
                            top_k=top_k_val,
                            enabled_methods=enabled_methods,
                            corr_threshold=corr_thresh_val,
                            vif_threshold=vif_thresh_val,
                            progress_callback=_cb,
                        )
                        st.session_state["_fs_result"]      = result
                        st.session_state["_fs_y_cols"]      = auto_y_cols
                        st.session_state["_fs_top_k"]       = top_k_val
                        st.session_state["_fs_corr_thresh"] = corr_thresh_val
                    except Exception as exc:
                        st.error(f"Analysis failed: {exc}")
                        progress_bar.empty()
                        progress_placeholder.empty()
                        return

                progress_bar.progress(1.0)
                progress_placeholder.empty()
                st.success(f"Analysis complete — {len(enabled_methods)} methods ran on {len(candidate_x)} features.")
                st.rerun()

    # ------------------------------------------------------------------
    # Pathway B — Automated Feature Selection
    # ------------------------------------------------------------------
    else:
        _step_badge(3, "Automated Feature Selection")
        _section_header(
            "⚡", "Automated Feature Selection",
            "Runs all available methods with best-default parameters in one click.",
        )

        auto_top_k = min(10, n_feat)
        st.markdown(
            f"<p style='color:{_MUTED};font-size:0.88rem'>"
            f"Will run <b style='color:#f8fafc'>{len(avail_methods)}</b> method(s) &nbsp;|&nbsp; "
            f"Top-K = <b style='color:#f8fafc'>{auto_top_k}</b> &nbsp;|&nbsp; "
            f"Collinearity threshold = <b style='color:#f8fafc'>0.85</b> &nbsp;|&nbsp; "
            f"VIF threshold = <b style='color:#f8fafc'>10.0</b>"
            "</p>",
            unsafe_allow_html=True,
        )

        auto_run_col, _ = st.columns([1, 3])
        with auto_run_col:
            auto_run_btn = st.button("⚡ Run Automated Feature Selection", key="fs_run_auto", use_container_width=True)

        if auto_run_btn:
            for k in ["_fs_result", "_fs_y_cols", "_fs_top_k", "_fs_corr_thresh"]:
                st.session_state.pop(k, None)

            df_num = cast_to_numeric(df)
            X_cand = df_num[candidate_x]
            y_targ = df_num[auto_y_cols]

            progress_placeholder = st.empty()
            progress_bar = st.progress(0)
            steps: List[str] = []

            def _auto_cb(msg: str) -> None:
                steps.append(msg)
                progress_placeholder.caption(f"⏳ {msg}")
                progress_bar.progress(min(len(steps) / (len(avail_methods) + 5), 0.95))

            with st.spinner("Running automated analysis — this may take 20–90 seconds…"):
                try:
                    result = run_auto_feature_selection(
                        X_df=X_cand, y_df=y_targ,
                        top_k=auto_top_k,
                        enabled_methods=avail_methods,
                        corr_threshold=0.85,
                        vif_threshold=10.0,
                        progress_callback=_auto_cb,
                    )
                    st.session_state["_fs_result"]      = result
                    st.session_state["_fs_y_cols"]      = auto_y_cols
                    st.session_state["_fs_top_k"]       = auto_top_k
                    st.session_state["_fs_corr_thresh"] = 0.85
                except Exception as exc:
                    st.error(f"Automated analysis failed: {exc}")
                    progress_bar.empty()
                    progress_placeholder.empty()
                    return

            progress_bar.progress(1.0)
            progress_placeholder.empty()
            st.success(f"Automated analysis complete — {len(avail_methods)} methods ran on {len(candidate_x)} features.")
            st.rerun()

    # ------------------------------------------------------------------
    # Step 4: Analysis Results  (shown after either pathway completes)
    # ------------------------------------------------------------------
    if "_fs_result" not in st.session_state:
        return

    result: AutoSelectionResult = st.session_state["_fs_result"]
    res_y   = st.session_state.get("_fs_y_cols", auto_y_cols)
    res_k   = st.session_state.get("_fs_top_k", min(10, n_feat))
    c_thresh = st.session_state.get("_fs_corr_thresh", 0.85)

    _step_badge(4, "Analysis Results")
    _render_analysis_results(result, res_y, res_k, candidate_x, c_thresh, numeric_cols)

    # Summary of selections
    all_keep = result.recommended_features + result.optional_features
    st.markdown(
        f"**Highly Recommended + Recommended X ({len(result.recommended_features)}):** "
        f"`{', '.join(result.recommended_features) or 'None'}`  \n"
        f"**Consider X ({len(result.optional_features)}):** "
        f"`{', '.join(result.optional_features) or 'None'}`  \n"
        f"**Weak Feature ({len(result.features_to_remove)}):** "
        f"`{', '.join(result.features_to_remove) or 'None'}`"
    )

    # Quick-apply buttons before manual selection
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button(f"✅ Use Recommended ({len(result.recommended_features)})", key="fs_qa_rec"):
            _sync_checkboxes(result.recommended_features, res_y, numeric_cols)
            st.session_state["_fs_suggested_x"] = result.recommended_features
            st.rerun()
    with qa2:
        if st.button(f"⭐ Use Rec + Optional ({len(all_keep)})", key="fs_qa_all_keep"):
            _sync_checkboxes(all_keep, res_y, numeric_cols)
            st.session_state["_fs_suggested_x"] = all_keep
            st.rerun()
    with qa3:
        if st.button("🗑️ Clear Results", key="fs_clear"):
            for k in ["_fs_result", "_fs_y_cols", "_fs_top_k", "_fs_corr_thresh", "_fs_suggested_x"]:
                st.session_state.pop(k, None)
            st.rerun()
    with qa4:
        pass  # spacer

    st.markdown("---")

    # ------------------------------------------------------------------
    # Manual Variable Selection  (auto-filled from results)
    # ------------------------------------------------------------------
    suggested_x = st.session_state.get("_fs_suggested_x", result.recommended_features)
    suggested_y = res_y

    x_cols, y_cols = _render_manual_variable_selection(suggested_x, suggested_y, numeric_cols)

    # ------------------------------------------------------------------
    # Final Apply
    # ------------------------------------------------------------------
    _render_final_apply(df, x_cols, y_cols)
