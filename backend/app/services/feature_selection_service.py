"""
backend/app/services/feature_selection_service.py
=====================================================
Wraps src.feature_selection.auto_selector's two entry points for the job
manager. Both already accept a progress_callback(message: str) — no changes
needed there; the job manager's "message" progress_mode binds it directly.

run_auto_feature_selection / run_per_target_auto_selection return dataclasses
holding pandas DataFrames (consensus_df, vif_df, corr_with_target, ...). We
serialize those to plain JSON-safe dicts here, in the job's target function,
so the job manager only ever holds JSON-safe data in memory.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import pandas as pd

from src.data.database import load_dataset_from_db
from src.feature_selection.auto_selector import (
    AutoSelectionResult,
    MethodResult,
    PerTargetSelectionResult,
    run_auto_feature_selection,
    run_per_target_auto_selection,
)


def _df_records(df: pd.DataFrame, index_name: Optional[str] = None) -> List[dict]:
    if df is None or df.empty:
        return []
    out = df.reset_index() if index_name is None else df.reset_index().rename(
        columns={df.index.name or "index": index_name}
    )
    return json.loads(out.to_json(orient="records"))


def _method_results_out(results: List[MethodResult]) -> List[dict]:
    return [
        {
            "name": r.name,
            "method_id": r.method_id,
            "category": r.category,
            "selected_features": r.selected_features,
            "notes": r.notes,
            "success": r.success,
            # raw_scores (not normalised) is what the ranking-matrix visualization
            # ranks features by, per method — mirrors _plot_ranking_matrix's
            # rank_maps construction in src/ui/pages/feature_selection.py.
            "raw_scores": r.raw_scores,
            # all_scores (normalised 0-1) and per_target_scores back the
            # Method Details table's "Norm Score" and "{Y} Raw" columns —
            # mirrors the st.dataframe built in feature_selection.py's tab5.
            "all_scores": r.all_scores,
            "per_target_scores": r.per_target_scores,
        }
        for r in results
    ]


def _serialize_single(result: AutoSelectionResult) -> dict:
    return {
        "mode": "combined",
        "consensus": _df_records(result.consensus_df, index_name="Rank"),
        "recommended_features": result.recommended_features,
        "optional_features": result.optional_features,
        "features_to_remove": result.features_to_remove,
        "per_feature_reasoning": result.per_feature_reasoning,
        "dataset_info": result.dataset_info,
        "method_results": _method_results_out(result.method_results),
        "vif": _df_records(result.vif_df),
        "corr_with_target": _df_records(result.corr_with_target, index_name="Feature"),
    }


def _serialize_per_target(result: PerTargetSelectionResult) -> dict:
    out = {
        "mode": "per_target",
        "consensus": _df_records(result.consensus_df, index_name="Rank"),
        "recommended_features": result.recommended_features,
        "optional_features": result.optional_features,
        "features_to_remove": result.features_to_remove,
        "per_feature_reasoning": result.per_feature_reasoning,
        "dataset_info": result.dataset_info,
        "method_results": _method_results_out(result.method_results),
        "vif": _df_records(result.vif_df),
        "corr_with_target": _df_records(result.corr_with_target, index_name="Feature"),
        "feature_target_map": result.feature_target_map,
    }
    out["per_target_summary"] = {
        y_col: {
            "recommended_features": res.recommended_features,
            "optional_features": res.optional_features,
        }
        for y_col, res in result.target_results.items()
    }
    return out


def run_feature_selection_job(
    dataset_name: str,
    y_cols: List[str],
    x_cols: Optional[List[str]],
    top_k: int,
    enabled_methods: Optional[List[str]],
    corr_threshold: float,
    vif_threshold: float,
    per_target: bool,
    progress_callback=None,
) -> dict:
    """
    Target function submitted to the job manager (progress_mode='message').

    Runs in a background thread, not a request handler — validation failures
    here become job.error text surfaced through GET /api/jobs/{id}, not an
    immediate HTTP response, so plain exceptions are used rather than
    HTTPException.
    """
    df = load_dataset_from_db(dataset_name)
    if df is None:
        raise ValueError(f"Dataset '{dataset_name}' could not be loaded.")

    missing_y = [c for c in y_cols if c not in df.columns]
    if missing_y:
        raise ValueError(f"Target columns not found: {missing_y}")

    resolved_x_cols = x_cols or [
        c for c in df.select_dtypes(include="number").columns if c not in y_cols
    ]
    missing_x = [c for c in resolved_x_cols if c not in df.columns]
    if missing_x:
        raise ValueError(f"Feature columns not found: {missing_x}")

    X_df = df[resolved_x_cols]
    y_df = df[y_cols]

    if per_target:
        result = run_per_target_auto_selection(  # unchanged
            X_df=X_df,
            y_df=y_df,
            top_k=top_k,
            enabled_methods=enabled_methods,
            corr_threshold=corr_threshold,
            vif_threshold=vif_threshold,
            progress_callback=progress_callback,
        )
        return _serialize_per_target(result)

    result = run_auto_feature_selection(  # unchanged
        X_df=X_df,
        y_df=y_df,
        top_k=top_k,
        enabled_methods=enabled_methods,
        corr_threshold=corr_threshold,
        vif_threshold=vif_threshold,
        progress_callback=progress_callback,
    )
    return _serialize_single(result)
