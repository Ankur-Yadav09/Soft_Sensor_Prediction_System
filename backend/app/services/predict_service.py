"""
backend/app/services/predict_service.py
===========================================
Takes an explicit model_name (always loaded from disk/registry — the old
"session model vs registry model" ambiguity from the Streamlit app doesn't
exist here, since there's no in-memory "session model" to distinguish it
from) plus an explicit data source: either a project's persisted test
split, or a stored dataset (optionally row-sliced). Synchronous — inference
is fast enough not to need the job manager.
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd
from fastapi import HTTPException

from src.data.database import load_dataset_from_db
from src.evaluation.metrics import compute_metrics
from src.persistence.model_store import list_saved_models, load_model_from_disk

from backend.app.services import project_service


def run_predict(
    model_name: str,
    source: str,
    project_id: Optional[str],
    dataset_name: Optional[str],
    row_start: Optional[int],
    row_end: Optional[int],
) -> dict:
    known_names = {m["name"] for m in list_saved_models()}  # unchanged
    if model_name not in known_names:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    wrapper, scaler_x, scaler_y, x_cols, y_cols = load_model_from_disk(model_name)  # unchanged

    if source == "project_test":
        if not project_id:
            raise HTTPException(status_code=422, detail="project_id is required when source='project_test'.")
        project = project_service.load_project(project_id)  # unchanged, 404s on unknown id
        X_scaled = project.X_test
        X_display = pd.DataFrame(scaler_x.inverse_transform(X_scaled), columns=project.x_cols)
        actual_df = project.y_test_raw.reset_index(drop=True)
        resolved_y_cols = project.y_cols
        has_actuals = True

    elif source == "dataset":
        if not dataset_name:
            raise HTTPException(status_code=422, detail="dataset_name is required when source='dataset'.")
        df = load_dataset_from_db(dataset_name)  # unchanged
        if df is None:
            raise HTTPException(status_code=422, detail=f"Dataset '{dataset_name}' could not be loaded.")
        if row_start is not None or row_end is not None:
            df = df.iloc[row_start or 0 : row_end]
        missing_x = [c for c in x_cols if c not in df.columns]
        if missing_x:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset '{dataset_name}' is missing columns required by this model: {missing_x}",
            )
        X_display = df[x_cols].reset_index(drop=True)
        X_scaled = scaler_x.transform(X_display)
        resolved_y_cols = y_cols
        has_actuals = all(c in df.columns for c in y_cols)
        actual_df = df[y_cols].reset_index(drop=True) if has_actuals else None

    else:
        raise HTTPException(status_code=422, detail="source must be 'project_test' or 'dataset'.")

    preds = scaler_y.inverse_transform(wrapper.predict_scaled(X_scaled))  # unchanged

    rows_df = X_display.copy()
    for i, col in enumerate(resolved_y_cols):
        rows_df[f"{col}_predicted"] = preds[:, i]
        if has_actuals:
            rows_df[f"{col}_actual"] = actual_df[col].values
            rows_df[f"{col}_error"] = rows_df[f"{col}_actual"] - rows_df[f"{col}_predicted"]

    metrics = None
    if has_actuals:
        metrics_df = compute_metrics(actual_df, preds, resolved_y_cols)  # unchanged
        metrics_df = metrics_df.reset_index().rename(columns={"index": "Target"})
        metrics = json.loads(metrics_df.to_json(orient="records"))

    return {
        "x_cols": x_cols,
        "y_cols": resolved_y_cols,
        "rows": json.loads(rows_df.to_json(orient="records")),
        "has_actuals": has_actuals,
        "metrics": metrics,
    }
