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

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.preprocessing import (
    cast_to_numeric,
    compute_feature_stats,
    impute,
    split_and_scale,
)
from src.feature_selection.auto_selector import (
    METHOD_CATEGORIES,
    METHOD_LABELS,
    AutoSelectionResult,
    PerTargetSelectionResult,
    run_auto_feature_selection,
    run_per_target_auto_selection,
    _MAX_FEATURES_NEW,
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


def _plot_ranking_matrix(
    consensus_df: pd.DataFrame,
    method_results: list,
    sort_by: str = "Final Score",
    filter_cats: List[str] = None,
) -> go.Figure:
    """
    Feature Ranking Matrix — heatmap showing each feature's rank within every
    active scoring method.  Purely informational; does not affect any score.
    """
    # --- sort ---
    sort_col_map = {
        "Final Score":         "FinalScore",
        "Predictive Strength": "PredictiveStrength",
        "Selection Frequency": "SelectionFreq",
        "Average Rank":        "AvgRank",
    }
    sort_col = sort_col_map.get(sort_by, "FinalScore")
    df = consensus_df.reset_index()

    # --- filter ---
    if filter_cats:
        legacy = {"Consider": "Optional", "Weak Feature": "Remove"}
        all_cats = filter_cats + [legacy.get(c, "") for c in filter_cats]
        df = df[df["Recommendation"].isin(all_cats)]
    if df.empty:
        return go.Figure()

    ascending = sort_col == "AvgRank"
    df = df.sort_values(sort_col, ascending=ascending)
    features = df["Feature"].tolist()

    # --- method order (scoring methods only, those that ran) ---
    METHOD_ORDER = [
        ("target_correlation",     "Correlation"),
        ("mutual_information",     "Mut. Info"),
        ("mrmr",                   "mRMR"),
        ("permutation_importance", "Permutation"),
        ("elasticnet",             "ElasticNet"),
    ]
    SELECTION_METHODS = {"elasticnet"}

    ran = {r.method_id: r for r in method_results if r.success}
    active = [(mid, lbl) for mid, lbl in METHOD_ORDER if mid in ran]
    if not active:
        return go.Figure()

    method_ids    = [mid for mid, _ in active]
    method_labels = [lbl for _, lbl in active]

    # Precompute full-dataset rank maps (rank within ALL features seen by each method)
    rank_maps: Dict[str, tuple] = {}
    for mid in method_ids:
        r = ran[mid]
        all_feats = list(r.raw_scores.keys())
        sorted_f  = sorted(all_feats, key=lambda f: r.raw_scores.get(f, 0.0), reverse=True)
        rank_maps[mid] = ({f: i + 1 for i, f in enumerate(sorted_f)}, len(sorted_f))

    # --- build z / text / hover matrices ---
    z_vals, texts, hovers = [], [], []
    for feat in features:
        z_row, t_row, h_row = [], [], []
        for mid in method_ids:
            r = ran[mid]
            if mid in SELECTION_METHODS:
                selected = feat in r.selected_features
                z_row.append(1.0 if selected else 0.0)
                t_row.append("✓" if selected else "✗")
                h_row.append(
                    f"<b>{feat}</b><br>Method: {METHOD_LABELS[mid]}<br>"
                    f"{'Selected' if selected else 'Not Selected'}"
                )
            else:
                rank_map, n_total = rank_maps[mid]
                if feat in rank_map:
                    rank     = rank_map[feat]
                    goodness = (n_total - rank) / max(n_total - 1, 1)
                    z_row.append(goodness)
                    t_row.append(str(rank))
                    h_row.append(
                        f"<b>{feat}</b><br>Method: {METHOD_LABELS[mid]}<br>"
                        f"Rank: {rank} of {n_total}"
                    )
                else:
                    z_row.append(None)
                    t_row.append("")
                    h_row.append(f"<b>{feat}</b><br>Method: {METHOD_LABELS[mid]}<br>Not evaluated")
        z_vals.append(z_row)
        texts.append(t_row)
        hovers.append(h_row)

    colorscale = [
        [0.00, "#ef4444"],   # red  — poor / not selected
        [0.30, "#f59e0b"],   # amber — lower ranks
        [0.65, "#86efac"],   # light green — top 10
        [1.00, "#16a34a"],   # dark green — top 3 / selected
    ]

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=method_labels,
        y=features,
        text=texts,
        customdata=hovers,
        hovertemplate="%{customdata}<extra></extra>",
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorscale=colorscale,
        showscale=False,
        zmin=0.0,
        zmax=1.0,
    ))
    fig.update_layout(
        title="Feature Ranking Matrix — Cross-Method Comparison",
        xaxis=dict(
            title="Feature Selection Method",
            tickangle=-30,
            side="top",
            tickfont=dict(size=11),
        ),
        yaxis=dict(title="Feature", autorange="reversed", tickfont=dict(size=11)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        height=max(420, len(features) * 32 + 160),
        margin=dict(l=10, r=10, t=110, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Build default method list
# ---------------------------------------------------------------------------

# Fixed ordered list of the 5 core scoring methods (no optional libraries needed)
_CORE_METHOD_ORDER = [
    "target_correlation",
    "mutual_information",
    "mrmr",
    "permutation_importance",
    "elasticnet",
]


def _build_available_methods() -> List[str]:
    """Return the 5 core scoring methods available for user selection."""
    return list(_CORE_METHOD_ORDER)


def _build_default_methods(n_feat: int) -> List[str]:
    """Return the 5 core scoring methods enabled by default.

    Permutation Importance is skipped when n_feat exceeds the performance guard
    threshold (_MAX_FEATURES_NEW) to avoid excessive runtime on wide datasets.
    """
    if n_feat > _MAX_FEATURES_NEW:
        return [m for m in _CORE_METHOD_ORDER if m != "permutation_importance"]
    return list(_CORE_METHOD_ORDER)


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
    pt_result: Optional[PerTargetSelectionResult] = None,
) -> None:
    cdf  = result.consensus_df
    info = result.dataset_info

    n_scoring_ran = sum(1 for r in result.method_results if r.success)
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
    kc1.metric("Scoring Methods Run",   n_scoring_ran)
    kc2.metric("🟢 Highly Recommended", n_highly)
    kc3.metric("🔵 Recommended",        n_rec)
    kc4.metric("🟡 Consider",           n_opt)
    kc5.metric("🔴 Weak Feature",       n_rem)

    tab_labels = [
        "📊 Overview",
        "🏆 Consensus Rankings",
        "📈 Visualizations",
        "🎯 Recommendations",
        "🔬 Method Details",
    ]
    if pt_result is not None:
        tab_labels.append("📋 Per-Target Summary")
    tabs = st.tabs(tab_labels)
    tab1, tab2, tab3, tab4, tab5 = tabs[:5]
    tab6 = tabs[5] if len(tabs) > 5 else None

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

        if info.get("vif_skipped"):
            st.warning(
                "**VIF computation skipped** — dataset has more than 80 features. "
                "The Highly Recommended hard gate (VIF < 10) is inactive: a multicollinear feature "
                "can still reach Highly Recommended on this dataset. "
                "Consider reducing features in Preprocessing first."
            )

        if info.get("permutation_skipped"):
            st.warning(
                "**Permutation Importance skipped** — dataset has more than 100 features. "
                "Its 30% weight has been redistributed proportionally across the remaining 4 methods. "
                "Scores are still valid but specialist/nonlinear features may be under-scored."
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
            "**Final Score** = 30% × Selection Frequency (damped) + 50% × Predictive Strength "
            "+ 20% × Stability Score"
        )

        def _style_rec(val: str) -> str:
            color = _REC_COLORS.get(val, "")
            return f"color: {color}; font-weight: bold" if color else ""

        base_cols = [
            "Feature",
            "CoverageCount", "CoverageRatio", "CoveragePercent",   # multi-Y coverage (absent for single-Y)
            "SelectionCount", "TotalMethods", "SelectionFreq",
            "AvgRank", "PredictiveStrength", "StabilityScore", "FinalScore",
            "CorrWithTarget", "VIF",
            "ElasticNetSelected", "Recommendation", "MulticollinearWith",
        ]
        available_cols = [c for c in base_cols if c in cdf.columns]
        disp_df = cdf.reset_index()[available_cols] if "Rank" not in cdf.columns else cdf[available_cols]
        st.dataframe(disp_df.style.map(_style_rec, subset=["Recommendation"]), use_container_width=True, height=450)
        st.plotly_chart(_plot_consensus_bar(cdf), use_container_width=True)

    with tab3:
        st.markdown("#### Feature Ranking Matrix")
        st.caption(
            "Each cell shows a feature's rank within that method (1 = top ranked). "
            "Lasso / ElasticNet show ✓ (selected) or ✗ (not selected). "
            "Color: 🟢 top-ranked → 🔴 lower-ranked."
        )

        ctrl1, ctrl2 = st.columns([1, 2])
        with ctrl1:
            viz_sort = st.selectbox(
                "Sort features by",
                ["Final Score", "Predictive Strength", "Selection Frequency", "Average Rank"],
                key="fs_viz_sort",
            )
        with ctrl2:
            all_rec_cats = ["Highly Recommended", "Recommended", "Consider", "Weak Feature"]
            viz_filter = st.multiselect(
                "Filter by recommendation",
                all_rec_cats,
                default=all_rec_cats,
                key="fs_viz_filter",
            )

        fig_matrix = _plot_ranking_matrix(cdf, result.method_results, viz_sort, viz_filter)
        st.plotly_chart(fig_matrix, use_container_width=True)

        st.markdown(
            "<div style='font-size:0.82rem;color:#94a3b8'>"
            "<b style='color:#16a34a'>Dark green</b> = top 3 ranks &nbsp;|&nbsp; "
            "<b style='color:#86efac'>Light green</b> = top 10 ranks &nbsp;|&nbsp; "
            "<b style='color:#f59e0b'>Amber</b> = lower ranks &nbsp;|&nbsp; "
            "<b style='color:#ef4444'>Red</b> = poor ranking / not selected"
            "</div>",
            unsafe_allow_html=True,
        )

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
                avg_rank_val = row.get("AvgRank")
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
                    # Per-Y correlation row (only shown when > 1 Y target)
                    if (not result.corr_with_target.empty
                            and feat in result.corr_with_target.index
                            and result.corr_with_target.shape[1] > 1):
                        per_y_cols = result.corr_with_target.columns.tolist()
                        py_cols_ui = st.columns(len(per_y_cols))
                        for ci, y_col in enumerate(per_y_cols):
                            r_val = result.corr_with_target.loc[feat, y_col]
                            py_cols_ui[ci].metric(f"|r| {y_col}", f"{abs(r_val):.3f}")
                    # Score breakdown row
                    if ps_val is not None:
                        sb1, sb2, sb3, sb4 = st.columns(4)
                        sb1.metric("Predictive Strength", f"{ps_val:.1f}")
                        if fq_val is not None:
                            sb2.metric("Feature Quality", f"{fq_val:.1f}")
                        if stab_val is not None:
                            sb3.metric("Stability Score", f"{stab_val:.1f}")
                        if avg_rank_val is not None:
                            rank_label = "Top Ranked" if avg_rank_val <= 3 else ("Mid Ranked" if avg_rank_val <= 7 else "Lower Ranked")
                            sb4.metric("Avg Rank", f"{avg_rank_val:.1f}", help=rank_label)
                    if pt_result is not None and feat in pt_result.feature_target_map:
                        targets_for_feat = pt_result.feature_target_map[feat]
                        n_total_targets = len(pt_result.target_results)
                        st.markdown(
                            f"<span style='background:#1e3a5f;color:#60a5fa;padding:3px 10px;"
                            f"border-radius:6px;font-size:0.82rem;font-weight:600'>"
                            f"Selected for: {', '.join(targets_for_feat)} &nbsp;"
                            f"({len(targets_for_feat)}/{n_total_targets} targets)</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("")
                    st.markdown(result.per_feature_reasoning.get(feat, ""))
                    st.markdown("---")

    with tab5:
        st.markdown("#### Per-Method Feature Rankings")

        if pt_result is not None:
            # ----------------------------------------------------------------
            # Multi-Y path: each method ran independently per target.
            # Show one expander per method with a score column for every Y.
            # ----------------------------------------------------------------
            st.caption(
                "Each method ran independently for every Y target. "
                "Scores shown per target alongside the aggregated average. "
                "Rank is based on the aggregated average score."
            )

            n_t = len(pt_result.target_results)
            y_cols_ordered = list(pt_result.target_results.keys())

            for agg_r in result.method_results:
                # Collect per-target MethodResult for this method
                t_method: Dict[str, any] = {}
                for y_col in y_cols_ordered:
                    t_res = pt_result.target_results[y_col]
                    mr = next(
                        (r for r in t_res.method_results if r.method_id == agg_r.method_id and r.success),
                        None,
                    )
                    if mr is not None:
                        t_method[y_col] = mr

                n_ran = len(t_method)
                status = "✅" if n_ran > 0 else "❌"
                header = (
                    f"{status} **{agg_r.name}** — {agg_r.category}  |  "
                    f"ran for {n_ran}/{n_t} targets  |  "
                    f"{len(agg_r.selected_features)} features selected (≥50% targets)"
                )
                with st.expander(header):
                    if n_ran == 0:
                        st.error("Method failed for all targets.")
                        continue

                    # All features in the union, sorted by aggregated avg score desc
                    all_scope = pt_result.union_features + pt_result.optional_union
                    all_scope_sorted = sorted(
                        all_scope,
                        key=lambda f: agg_r.all_scores.get(f, 0.0),
                        reverse=True,
                    )

                    # For Target Correlation: use signed Pearson r from corr_with_target
                    # (raw_scores store |r| for scoring; corr_with_target stores signed r for display)
                    is_corr_method = agg_r.method_id == "target_correlation"
                    corr_lookup = (
                        pt_result.corr_with_target
                        if is_corr_method and not pt_result.corr_with_target.empty
                        else None
                    )

                    rows = []
                    for rank, feat in enumerate(all_scope_sorted, 1):
                        row_dict: dict = {"Rank": rank, "Feature": feat}
                        for y_col in y_cols_ordered:
                            if is_corr_method and corr_lookup is not None:
                                # Signed Pearson r — shows direction of relationship
                                val = (
                                    corr_lookup.loc[feat, y_col]
                                    if feat in corr_lookup.index and y_col in corr_lookup.columns
                                    else None
                                )
                                row_dict[y_col] = round(float(val), 4) if val is not None else None
                            elif y_col in t_method:
                                row_dict[y_col] = round(t_method[y_col].raw_scores.get(feat, 0.0), 5)
                            else:
                                row_dict[y_col] = None
                        row_dict["Avg |r|" if is_corr_method else "Avg Score"] = round(agg_r.raw_scores.get(feat, 0.0), 5)
                        row_dict["Norm Score"] = round(agg_r.all_scores.get(feat, 0.0), 4)
                        rows.append(row_dict)

                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420)
                        if is_corr_method:
                            st.caption(
                                "Y columns show signed Pearson r (+/– indicates direction).  "
                                "Avg |r| = mean absolute correlation across targets (used in scoring).  "
                                "Norm Score = normalised 0–1 (used in Final Score)."
                            )
                        else:
                            st.caption(
                                f"Each Y column shows the raw score for that target run.  "
                                f"Avg Score = mean raw score across {n_ran} target(s).  "
                                f"Norm Score = normalised 0–1 (used in Final Score)."
                            )

        else:
            # ----------------------------------------------------------------
            # Single-Y path: original behaviour unchanged.
            # ----------------------------------------------------------------
            st.caption(
                "Methods that train per Y target show individual raw scores alongside "
                "the averaged raw score and normalized score. Methods that reduce to a "
                "single averaged Y (Permutation, mRMR) show only Avg Raw and Norm Score."
            )
            for r in result.method_results:
                status = "✅" if r.success else "❌"
                with st.expander(f"{status} **{r.name}** — {r.category}  |  {len(r.selected_features)} features  ({r.notes})"):
                    if not r.success:
                        st.error(r.notes)
                        continue
                    has_per_y = bool(r.per_target_scores)
                    if has_per_y:
                        sample_feat = r.selected_features[0] if r.selected_features else None
                        y_col_names = list(r.per_target_scores.get(sample_feat, {}).keys()) if sample_feat else []
                    else:
                        y_col_names = []
                    is_corr = r.method_id == "target_correlation"
                    signed_corr = result.corr_with_target if is_corr and not result.corr_with_target.empty else None
                    rows = []
                    for rank, feat in enumerate(r.selected_features, 1):
                        row_dict: dict = {"Rank": rank, "Feature": feat}
                        if is_corr and signed_corr is not None:
                            for yc in signed_corr.columns:
                                val = signed_corr.loc[feat, yc] if feat in signed_corr.index else None
                                row_dict[yc] = round(float(val), 4) if val is not None else None
                            row_dict["Avg |r|"] = round(r.raw_scores.get(feat, 0), 5)
                        else:
                            if has_per_y and y_col_names:
                                for yc in y_col_names:
                                    row_dict[f"{yc} Raw"] = round(r.per_target_scores.get(feat, {}).get(yc, 0.0), 5)
                            row_dict["Avg Raw"] = round(r.raw_scores.get(feat, 0), 5)
                        row_dict["Norm Score"] = round(r.all_scores.get(feat, 0), 4)
                        rows.append(row_dict)
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                        if is_corr:
                            st.caption("Y column shows signed Pearson r. Avg |r| = absolute correlation (used in scoring).")

    if tab6 is not None and pt_result is not None:
        with tab6:
            st.markdown("#### Target-wise Feature Selection Summary")
            st.caption(
                "Shows which Y targets each feature was selected for (Highly Recommended or "
                "Recommended) when running feature selection independently per target. "
                "Features are sorted by coverage (most targets first)."
            )

            n_targets = len(pt_result.target_results)
            target_names = list(pt_result.target_results.keys())

            # Build summary rows
            summary_rows = []
            for feat in pt_result.union_features:
                selected_for = pt_result.feature_target_map.get(feat, [])
                row: dict = {"Feature": feat}
                for t in target_names:
                    row[t] = "✅" if t in selected_for else "—"
                row["Coverage"] = f"{len(selected_for)}/{n_targets}"
                # Overall recommendation from consensus result
                if feat in cdf["Feature"].values:
                    rec = cdf.loc[cdf["Feature"] == feat, "Recommendation"].values[0]
                else:
                    rec = "—"
                row["Consensus Rec"] = rec
                summary_rows.append(row)

            # Also show features that were only in optional_union
            for feat in pt_result.optional_union:
                if feat in [r["Feature"] for r in summary_rows]:
                    continue
                selected_for = pt_result.feature_target_map.get(feat, [])
                row = {"Feature": feat}
                for t in target_names:
                    row[t] = "○" if t in selected_for else "—"
                row["Coverage"] = f"{len(selected_for)}/{n_targets} (optional)"
                if feat in cdf["Feature"].values:
                    rec = cdf.loc[cdf["Feature"] == feat, "Recommendation"].values[0]
                else:
                    rec = "—"
                row["Consensus Rec"] = rec
                summary_rows.append(row)

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, use_container_width=True, height=420)
                st.caption(
                    "✅ = selected (Highly Recommended / Recommended for that target) &nbsp;|&nbsp; "
                    "○ = optional/consider &nbsp;|&nbsp; — = not selected"
                )

            st.markdown("#### Per-Target Recommendation Counts")
            count_rows = []
            for t, t_res in pt_result.target_results.items():
                t_cdf = t_res.consensus_df
                count_rows.append({
                    "Target": t,
                    "Highly Recommended": sum(1 for v in t_cdf["Recommendation"] if v == "Highly Recommended"),
                    "Recommended":        sum(1 for v in t_cdf["Recommendation"] if v == "Recommended"),
                    "Consider":           sum(1 for v in t_cdf["Recommendation"] if v in ("Consider", "Optional")),
                    "Weak Feature":       sum(1 for v in t_cdf["Recommendation"] if v in ("Weak Feature", "Remove")),
                })
            st.dataframe(pd.DataFrame(count_rows), use_container_width=True)

            st.markdown("#### Per-Target Predictive Strength")
            st.caption(
                "Predictive Strength each target assigned to a feature independently. "
                "The aggregated consensus uses the mean of these values."
            )
            ps_rows = []
            for feat in pt_result.union_features + pt_result.optional_union:
                n_cov = len(pt_result.feature_target_map.get(feat, []))
                row_ps: dict = {
                    "Feature":  feat,
                    "Coverage": f"{n_cov}/{n_targets}",
                }
                for t, t_res in pt_result.target_results.items():
                    t_cdf = t_res.consensus_df
                    if not t_cdf.empty and "Feature" in t_cdf.columns:
                        match = t_cdf.loc[t_cdf["Feature"] == feat, "PredictiveStrength"]
                        row_ps[f"PS: {t}"] = round(float(match.values[0]), 1) if len(match) else None
                    else:
                        row_ps[f"PS: {t}"] = None
                ps_rows.append(row_ps)
            if ps_rows:
                st.dataframe(pd.DataFrame(ps_rows), use_container_width=True)

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
            _section_header("📋", "Methods Selection", "Enable / disable the 5 core scoring methods.")

            enabled_methods = _method_checkboxes(avail_methods, default_methods)
            st.markdown(
                f"<p style='color:{_MUTED};font-size:0.85rem'><b style='color:#f8fafc'>"
                f"{len(enabled_methods)}</b> of 5 core scoring method(s) selected.</p>",
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
                        if len(auto_y_cols) > 1:
                            # Multi-Y: per-target run is the sole engine
                            pt_result = run_per_target_auto_selection(
                                X_df=X_cand, y_df=y_targ,
                                top_k=top_k_val,
                                enabled_methods=enabled_methods,
                                corr_threshold=corr_thresh_val,
                                vif_threshold=vif_thresh_val,
                                progress_callback=_cb,
                            )
                            # Build pseudo AutoSelectionResult for display compatibility
                            result = AutoSelectionResult(
                                method_results=pt_result.method_results,
                                consensus_df=pt_result.consensus_df,
                                correlation_matrix=pt_result.correlation_matrix,
                                corr_with_target=pt_result.corr_with_target,
                                vif_df=pt_result.vif_df,
                                dataset_info=pt_result.dataset_info,
                                recommended_features=pt_result.recommended_features,
                                optional_features=pt_result.optional_features,
                                features_to_remove=pt_result.features_to_remove,
                                per_feature_reasoning=pt_result.per_feature_reasoning,
                            )
                            st.session_state["_fs_per_target_result"] = pt_result
                        else:
                            # Single-Y: unchanged path
                            result = run_auto_feature_selection(
                                X_df=X_cand, y_df=y_targ,
                                top_k=top_k_val,
                                enabled_methods=enabled_methods,
                                corr_threshold=corr_thresh_val,
                                vif_threshold=vif_thresh_val,
                                progress_callback=_cb,
                            )
                            st.session_state.pop("_fs_per_target_result", None)

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
                n_ran = sum(1 for r in result.method_results if r.success)
                st.success(f"Analysis complete — {n_ran} of {len(enabled_methods)} methods ran on {len(candidate_x)} features.")
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

        # Scale top_k with dataset width: larger feature sets need a wider top-k
        # so SelectionFreq remains meaningful. Cap at 30 to keep compute time reasonable.
        auto_top_k = min(max(10, n_feat // 5), 30)
        st.markdown(
            f"<p style='color:{_MUTED};font-size:0.88rem'>"
            f"Will run <b style='color:#f8fafc'>{len(avail_methods)}</b> independent scoring method(s) &nbsp;|&nbsp; "
            f"Top-K = <b style='color:#f8fafc'>{auto_top_k}</b> (auto-scaled to feature count) &nbsp;|&nbsp; "
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
                    if len(auto_y_cols) > 1:
                        # Multi-Y: per-target run is the sole engine
                        pt_result = run_per_target_auto_selection(
                            X_df=X_cand, y_df=y_targ,
                            top_k=auto_top_k,
                            enabled_methods=avail_methods,
                            corr_threshold=0.85,
                            vif_threshold=10.0,
                            progress_callback=_auto_cb,
                        )
                        # Build pseudo AutoSelectionResult for display compatibility
                        result = AutoSelectionResult(
                            method_results=pt_result.method_results,
                            consensus_df=pt_result.consensus_df,
                            correlation_matrix=pt_result.correlation_matrix,
                            corr_with_target=pt_result.corr_with_target,
                            vif_df=pt_result.vif_df,
                            dataset_info=pt_result.dataset_info,
                            recommended_features=pt_result.recommended_features,
                            optional_features=pt_result.optional_features,
                            features_to_remove=pt_result.features_to_remove,
                            per_feature_reasoning=pt_result.per_feature_reasoning,
                        )
                        st.session_state["_fs_per_target_result"] = pt_result
                    else:
                        # Single-Y: unchanged path
                        result = run_auto_feature_selection(
                            X_df=X_cand, y_df=y_targ,
                            top_k=auto_top_k,
                            enabled_methods=avail_methods,
                            corr_threshold=0.85,
                            vif_threshold=10.0,
                            progress_callback=_auto_cb,
                        )
                        st.session_state.pop("_fs_per_target_result", None)

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
            n_ran = sum(1 for r in result.method_results if r.success)
            st.success(f"Automated analysis complete — {n_ran} of {len(avail_methods)} methods ran on {len(candidate_x)} features.")
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

    pt_result: Optional[PerTargetSelectionResult] = st.session_state.get("_fs_per_target_result")

    _step_badge(4, "Analysis Results")
    _render_analysis_results(result, res_y, res_k, candidate_x, c_thresh, numeric_cols, pt_result)

    # Summary of selections — for multi-Y use aggregated per-target consensus
    if pt_result is not None:
        rec      = pt_result.recommended_features
        opt      = pt_result.optional_features
        weak     = pt_result.features_to_remove
        all_keep = rec + opt
        st.markdown(
            f"**Per-Target Recommended X — {len(rec)} features** "
            f"(coverage-based union across {len(res_y)} targets):  \n"
            f"`{', '.join(rec) or 'None'}`  \n"
            f"**Consider X ({len(opt)}):** "
            f"`{', '.join(opt) or 'None'}`  \n"
            f"**Weak Feature ({len(weak)}):** "
            f"`{', '.join(weak) or 'None'}`"
        )
    else:
        all_keep = result.recommended_features + result.optional_features
        st.markdown(
            f"**Highly Recommended + Recommended X ({len(result.recommended_features)}):** "
            f"`{', '.join(result.recommended_features) or 'None'}`  \n"
            f"**Consider X ({len(result.optional_features)}):** "
            f"`{', '.join(result.optional_features) or 'None'}`  \n"
            f"**Weak Feature ({len(result.features_to_remove)}):** "
            f"`{', '.join(result.features_to_remove) or 'None'}`"
        )

    # Quick-apply buttons — for multi-Y use aggregated recommended_features
    rec_features = pt_result.recommended_features if pt_result else result.recommended_features
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button(f"✅ Use Recommended ({len(rec_features)})", key="fs_qa_rec"):
            _sync_checkboxes(rec_features, res_y, numeric_cols)
            st.session_state["_fs_suggested_x"] = rec_features
            st.rerun()
    with qa2:
        if st.button(f"⭐ Use Rec + Optional ({len(all_keep)})", key="fs_qa_all_keep"):
            _sync_checkboxes(all_keep, res_y, numeric_cols)
            st.session_state["_fs_suggested_x"] = all_keep
            st.rerun()
    with qa3:
        if st.button("🗑️ Clear Results", key="fs_clear"):
            for k in ["_fs_result", "_fs_y_cols", "_fs_top_k", "_fs_corr_thresh",
                      "_fs_suggested_x", "_fs_per_target_result"]:
                st.session_state.pop(k, None)
            st.rerun()
    with qa4:
        pass  # spacer

    st.markdown("---")

    # ------------------------------------------------------------------
    # Manual Variable Selection  (auto-filled from results)
    # ------------------------------------------------------------------
    suggested_x = st.session_state.get("_fs_suggested_x", rec_features)
    suggested_y = res_y

    x_cols, y_cols = _render_manual_variable_selection(suggested_x, suggested_y, numeric_cols)

    # ------------------------------------------------------------------
    # Final Apply
    # ------------------------------------------------------------------
    _render_final_apply(df, x_cols, y_cols)
