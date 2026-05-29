"""
src/ui/pages/preprocess.py
===========================
Renders the "Preprocess" page.

Page sections
--------------
1. Dataset switcher
2. Intelligent Auto Feature Selection  ← comprehensive 12-method engine
3. Manual Variable Selection
4. Missing Data & Outliers
5. Custom Min-Max Filter
6. Apply Preprocessing
"""
from __future__ import annotations

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
# UI colour constants
# ---------------------------------------------------------------------------
_REC_COLORS = {
    "Highly Recommended": "#10b981",
    "Recommended":        "#3b82f6",
    "Optional":           "#f59e0b",
    "Remove":             "#ef4444",
}
_REC_ICONS = {
    "Highly Recommended": "🟢",
    "Recommended":        "🔵",
    "Optional":           "🟡",
    "Remove":             "🔴",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    available_methods: list[str],
    default_enabled: list[str],
) -> list[str]:
    """Render per-category method toggle checkboxes; return selected IDs."""
    # Group by category
    cat_map: dict[str, list[str]] = {}
    for mid in available_methods:
        cat = METHOD_CATEGORIES[mid]
        cat_map.setdefault(cat, []).append(mid)

    selected: list[str] = []
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


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

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


def _plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Feature Correlation Matrix") -> go.Figure:
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
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
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
        _REC_COLORS["Remove"] if v > 10
        else _REC_COLORS["Optional"] if v > 5
        else _REC_COLORS["Recommended"]
        for v in df["VIF"]
    ]
    fig = go.Figure(go.Bar(
        x=df["VIF"],
        y=df["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in df["VIF"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>VIF: %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=5,  line_dash="dash", line_color=_REC_COLORS["Optional"],
                  annotation_text="Moderate (5)", annotation_position="top right")
    fig.add_vline(x=10, line_dash="dash", line_color=_REC_COLORS["Remove"],
                  annotation_text="High (10)", annotation_position="top right")
    fig.update_layout(
        title="Variance Inflation Factor (VIF)",
        xaxis_title="VIF",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        height=max(350, len(df) * 22 + 80),
        margin=dict(l=10, r=80, t=40, b=20),
    )
    return fig


def _plot_target_corr_heatmap(corr_with_target: pd.DataFrame) -> go.Figure:
    if corr_with_target.empty:
        return go.Figure()
    data = corr_with_target.head(40)  # limit rows for readability
    fig = px.imshow(
        data,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature–Target Correlation",
        labels=dict(color="Pearson r"),
        aspect="auto",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        height=max(350, len(data) * 22 + 100),
        margin=dict(l=5, r=5, t=50, b=5),
    )
    return fig


def _plot_method_summary(method_results: list) -> go.Figure:
    names, counts, cats, colors_list = [], [], [], []
    cat_colors = {
        "Supervised": "#3b82f6",
        "Feature Importance": "#10b981",
        "Intrinsic": "#8b5cf6",
        "Wrapper": "#f59e0b",
        "Dimensionality Reduction": "#ec4899",
    }
    for r in method_results:
        names.append(r.name)
        counts.append(len(r.selected_features) if r.success else 0)
        cats.append(r.category)
        colors_list.append(cat_colors.get(r.category, "#94a3b8") if r.success else "#4b5563")

    fig = go.Figure(go.Bar(
        x=names,
        y=counts,
        marker_color=colors_list,
        text=counts,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Features Selected: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="Features Selected per Method",
        xaxis_tickangle=-35,
        yaxis_title="# Features",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        height=380,
        margin=dict(l=5, r=5, t=50, b=100),
    )
    return fig


# ---------------------------------------------------------------------------
# Main auto-selection section
# ---------------------------------------------------------------------------

def _render_intelligent_feature_selection(df: pd.DataFrame, numeric_cols: list) -> None:
    with st.expander("🤖 Intelligent Auto Feature Selection", expanded=False):
        st.markdown(
            "Select your **target Y** variables, choose which methods to run, "
            "and let the consensus engine rank and recommend the best **X input features** "
            "with full explainability."
        )

        # ---- Step 1: Y target selector -----------------------------------
        st.markdown("### Step 1 — Select Target Variable(s) Y")
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
            f"**{len(candidate_x)}** candidate X features | "
            f"**{len(auto_y_cols)}** target(s): `{', '.join(auto_y_cols)}`"
        )

        # ---- Step 2: Configuration ----------------------------------------
        st.markdown("### Step 2 — Configure Analysis")
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
            # Determine defaults based on dataset size
            n_rows = len(df)
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

            # Mark unavailable methods
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

        st.markdown(f"**{len(enabled_methods)}** method(s) selected.")

        # ---- Step 3: Run -------------------------------------------------
        run_btn = st.button("🔍 Run Intelligent Feature Analysis", key="ias_run")

        if run_btn:
            # Clear previous results
            for k in ["_ias_result", "_ias_y_cols", "_ias_top_k", "_ias_methods"]:
                st.session_state.pop(k, None)

            df_num = cast_to_numeric(df)
            X_cand = df_num[candidate_x]
            y_targ = df_num[auto_y_cols]

            progress_placeholder = st.empty()
            progress_bar = st.progress(0)
            steps = []

            def progress_cb(msg: str) -> None:
                steps.append(msg)
                progress_placeholder.caption(f"⏳ {msg}")
                progress_bar.progress(min(len(steps) / (len(enabled_methods) + 5), 0.95))

            with st.spinner("Analysing features — this may take 20–60 seconds…"):
                try:
                    result: AutoSelectionResult = run_auto_feature_selection(
                        X_df=X_cand,
                        y_df=y_targ,
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

        result: AutoSelectionResult  = st.session_state["_ias_result"]
        res_y:  list                 = st.session_state.get("_ias_y_cols", auto_y_cols)
        res_k:  int                  = st.session_state.get("_ias_top_k", top_k)
        cdf = result.consensus_df
        info = result.dataset_info

        st.markdown(f"---")
        st.markdown(
            f"### Analysis Results — Top-{res_k} features | "
            f"Y = `{', '.join(res_y)}`"
        )

        # KPI banner
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

        # ------------------------------------------------------------------
        # Tabbed dashboard
        # ------------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "🏆 Consensus Rankings",
            "📈 Visualizations",
            "🎯 Recommendations",
            "🔬 Method Details",
        ])

        # ---- TAB 1: Overview -------------------------------------------
        with tab1:
            st.markdown("#### Dataset Quality Report")
            qc1, qc2, qc3, qc4 = st.columns(4)
            qc1.metric("Total Rows",        info.get("n_rows", "?"))
            qc2.metric("Clean Features",    info.get("n_clean_features", "?"))
            qc3.metric("Missing % (X)",     f"{info.get('missing_pct_x', 0):.1f}%")
            qc4.metric("Missing % (Y)",     f"{info.get('missing_pct_y', 0):.1f}%")

            if info.get("constant_features"):
                st.warning(
                    f"**{len(info['constant_features'])} constant feature(s) removed** "
                    f"(zero variance): `{', '.join(info['constant_features'])}`"
                )

            st.markdown("#### Method Execution Summary")
            method_rows = []
            for r in result.method_results:
                method_rows.append({
                    "Method":     r.name,
                    "Category":   r.category,
                    "Status":     "✅ Success" if r.success else "❌ Failed",
                    "Selected":   len(r.selected_features),
                    "Notes":      r.notes,
                })
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
                            "Feature A": cols_m[i],
                            "Feature B": cols_m[j],
                            "|Pearson r|": round(abs(r_val), 4),
                        })
            if pairs:
                pairs_df = pd.DataFrame(pairs).sort_values("|Pearson r|", ascending=False)
                st.dataframe(pairs_df, use_container_width=True)
                st.caption(
                    f"{len(pairs)} redundant pair(s) detected. "
                    "Consider keeping only the higher-ranked feature from each pair."
                )
            else:
                st.success("No highly correlated feature pairs detected at this threshold.")

        # ---- TAB 2: Consensus Rankings ----------------------------------
        with tab2:
            st.markdown("#### Feature Ranking by Consensus Score")
            st.caption(
                "Confidence Score = 60% × Selection Frequency + 40% × Avg Normalized Score. "
                "Sort by any column."
            )

            # Coloured table via styling
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

        # ---- TAB 3: Visualizations -------------------------------------
        with tab3:
            viz1, viz2 = st.columns(2)
            with viz1:
                st.plotly_chart(
                    _plot_correlation_heatmap(result.correlation_matrix),
                    use_container_width=True,
                )
            with viz2:
                st.plotly_chart(
                    _plot_target_corr_heatmap(result.corr_with_target),
                    use_container_width=True,
                )

            viz3, viz4 = st.columns(2)
            with viz3:
                st.plotly_chart(
                    _plot_vif_chart(result.vif_df),
                    use_container_width=True,
                )
            with viz4:
                st.plotly_chart(
                    _plot_method_summary(result.method_results),
                    use_container_width=True,
                )

        # ---- TAB 4: Recommendations ------------------------------------
        with tab4:
            st.markdown("#### Feature Recommendation Cards")
            st.caption(
                "Features ranked by confidence score. "
                "Click any card to expand its detailed reasoning."
            )

            for rec_cat in ["Highly Recommended", "Recommended", "Optional", "Remove"]:
                feats_in_cat = cdf[cdf["Recommendation"] == rec_cat]
                if feats_in_cat.empty:
                    continue

                icon  = _REC_ICONS[rec_cat]
                color = _REC_COLORS[rec_cat]
                st.markdown(
                    f"<h4 style='color:{color}'>{icon} {rec_cat} "
                    f"({len(feats_in_cat)} feature(s))</h4>",
                    unsafe_allow_html=True,
                )

                for _, row in feats_in_cat.iterrows():
                    feat     = row["Feature"]
                    conf     = row["ConfidenceScore"]
                    n_sel    = int(row["SelectionCount"])
                    n_tot    = int(row["TotalMethods"])
                    corr_val = row.get("CorrWithTarget")
                    vif_val  = row.get("VIF")
                    reasoning = result.per_feature_reasoning.get(feat, "")

                    with st.expander(
                        f"**{feat}** — Confidence: {conf:.0f}%  "
                        f"({n_sel}/{n_tot} methods)",
                        expanded=(rec_cat == "Highly Recommended"),
                    ):
                        # Quick metrics row
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Confidence", f"{conf:.0f}%")
                        mc2.metric("Methods", f"{n_sel}/{n_tot}")
                        if corr_val is not None:
                            mc3.metric("Avg |r| w/ Target", f"{corr_val:.3f}")
                        if vif_val is not None:
                            mc4.metric("VIF", f"{vif_val:.1f}")

                        st.markdown(reasoning, unsafe_allow_html=False)
                        st.markdown("---")

        # ---- TAB 5: Method Details --------------------------------------
        with tab5:
            st.markdown("#### Per-Method Feature Rankings")
            for r in result.method_results:
                status_icon = "✅" if r.success else "❌"
                with st.expander(
                    f"{status_icon} **{r.name}** — {r.category}  |  "
                    f"{len(r.selected_features)} features selected  ({r.notes})"
                ):
                    if not r.success:
                        st.error(r.notes)
                        continue

                    rows = []
                    for rank, feat in enumerate(r.selected_features, 1):
                        rows.append({
                            "Rank":         rank,
                            "Feature":      feat,
                            "Raw Score":    round(r.raw_scores.get(feat, 0), 5),
                            "Norm Score":   round(r.all_scores.get(feat, 0), 4),
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # ---- Step 5: Apply / Clear buttons --------------------------------
        st.markdown("---")
        all_keep = result.recommended_features + result.optional_features
        st.markdown(
            f"**Recommended X features ({len(result.recommended_features)}):** "
            f"`{', '.join(result.recommended_features) if result.recommended_features else 'None'}`\n\n"
            f"**Optional X features ({len(result.optional_features)}):** "
            f"`{', '.join(result.optional_features) if result.optional_features else 'None'}`\n\n"
            f"**Features to Remove ({len(result.features_to_remove)}):** "
            f"`{', '.join(result.features_to_remove) if result.features_to_remove else 'None'}`"
        )

        ba1, ba2, ba3, ba4 = st.columns(4)
        with ba1:
            if st.button(
                f"✅ Apply Recommended ({len(result.recommended_features)})",
                key="ias_apply_rec",
            ):
                st.session_state.x_cols = result.recommended_features
                st.session_state.y_cols = res_y
                st.success("Applied Highly Recommended + Recommended features as X.")
                st.rerun()

        with ba2:
            if st.button(
                f"⭐ Apply Recommended + Optional ({len(all_keep)})",
                key="ias_apply_all_keep",
            ):
                st.session_state.x_cols = all_keep
                st.session_state.y_cols = res_y
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
                st.success(f"Applied {len(custom_sel)} custom X features.")
                st.rerun()

        with ba4:
            if st.button("🗑️ Clear Results", key="ias_clear"):
                for k in ["_ias_result", "_ias_y_cols", "_ias_top_k", "_ias_methods"]:
                    st.session_state.pop(k, None)
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

    if st.session_state.df is None:
        st.warning("Please upload data first in the 'Upload Data' tab.")
        return

    df = cast_to_numeric(st.session_state.df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------ #
    # Section 2: Intelligent Auto Feature Selection
    # ------------------------------------------------------------------ #
    _render_intelligent_feature_selection(df, numeric_cols)
    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Section 3: Manual Variable Selection
    # ------------------------------------------------------------------ #
    st.subheader("Variable Selection")
    st.caption(
        "Checkboxes are pre-filled when you apply Auto Feature Selection above. "
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
            default_checked = True if select_all_x else col in st.session_state.x_cols
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
            default_checked = True if select_all_y else col in st.session_state.y_cols
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
            ["None", "IQR Capping", "Min-Max Percentile Capping (1% - 99%)"],
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

        st.markdown("### Feature-wise Statistics (Before Imputation)")
        st.dataframe(compute_feature_stats(data_x))

        data_x, data_y = impute(data_x, data_y, imputation_method)
        data_x, data_y = apply_outlier_treatment(data_x, data_y, outlier_method)
        data_x, data_y = apply_custom_filters(data_x, data_y, custom_filters)

        st.markdown("### Feature-wise Statistics (After Preprocessing)")
        st.dataframe(compute_feature_stats(data_x))

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
            f"Preprocessing complete! Applied **{outlier_method}**. "
            "Train/Test split created and features scaled. "
            f"X = {len(x_cols)} features, Y = {len(y_cols)} targets."
        )
