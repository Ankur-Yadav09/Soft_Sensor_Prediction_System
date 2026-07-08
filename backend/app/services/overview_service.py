"""
backend/app/services/overview_service.py
============================================
GET /api/overview is pure aggregation (dataset + saved-model inventories) —
no per-model computation, no state.

GET /api/models/{model_name}/performance recomputes metrics on demand, using
only `model_name` as input: it reloads the model, its scalers, and its
originating dataset from disk/DB every call. There is nothing cached
between requests, so a backend restart never loses this information (unlike
the in-memory Workspace approach considered and dropped during planning).
"""
from __future__ import annotations

from fastapi import HTTPException

from src.data.database import list_datasets_from_db, list_models_from_registry, load_dataset_from_db
from src.evaluation.metrics import compute_metrics, grade_r2
from src.persistence.model_store import list_saved_models, load_model_from_disk

from backend.app.schemas.datasets import DatasetSummary
from backend.app.schemas.overview import (
    ModelPerformance,
    OverviewResponse,
    SavedModelSummary,
    TargetMetric,
)


def get_overview() -> OverviewResponse:
    datasets = [
        DatasetSummary(name=r[0], uploaded_at=r[1], rows=r[2], cols=r[3])
        for r in list_datasets_from_db()  # src.data.database — unchanged
    ]

    registry_by_name = {
        r["model_name"]: r for r in list_models_from_registry()  # unchanged
    }

    saved_models = []
    for m in list_saved_models():  # src.persistence.model_store — unchanged
        reg = registry_by_name.get(m["name"])
        saved_models.append(
            SavedModelSummary(
                name=m["name"],
                saved_at=m["saved_at"],
                input_dim=m["input_dim"],
                output_dim=m["output_dim"],
                algorithm=(reg["algorithm"] if reg else m.get("model_type")),
                avg_r2=reg["avg_r2"] if reg else None,
                avg_rmse=reg["avg_rmse"] if reg else None,
                avg_mae=reg["avg_mae"] if reg else None,
            )
        )

    return OverviewResponse(datasets=datasets, saved_models=saved_models)


def get_model_performance(model_name: str) -> ModelPerformance:
    known_names = {m["name"] for m in list_saved_models()}
    if model_name not in known_names:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    registry_entry = next(
        (r for r in list_models_from_registry() if r["model_name"] == model_name),
        None,
    )
    if registry_entry is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model '{model_name}' has no registry entry (its originating "
                "dataset is unknown), so live performance can't be recomputed."
            ),
        )
    dataset_name = registry_entry["dataset_name"]

    wrapper, scaler_x, scaler_y, x_cols, y_cols = load_model_from_disk(model_name)  # unchanged

    df = load_dataset_from_db(dataset_name)  # unchanged
    if df is None:
        raise HTTPException(
            status_code=422,
            detail=f"Originating dataset '{dataset_name}' could not be loaded.",
        )
    missing = [c for c in (*x_cols, *y_cols) if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset '{dataset_name}' is missing columns required by this model: {missing}",
        )

    # Recomputing over the raw stored dataset (not the cleaned train/test split
    # the model was actually trained on — that split isn't persisted anywhere
    # today, see the migration plan) can hit rows the original Preprocess step
    # would have imputed or dropped. Drop incomplete rows here so a handful of
    # missing values in the raw data doesn't take down an Overview summary.
    complete = df.dropna(subset=[*x_cols, *y_cols])
    if complete.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dataset '{dataset_name}' has no rows with complete values for "
                "this model's columns, so live performance can't be recomputed."
            ),
        )

    X_scaled = scaler_x.transform(complete[x_cols])
    preds = scaler_y.inverse_transform(wrapper.predict_scaled(X_scaled))  # unchanged
    metrics_df = compute_metrics(complete[y_cols], preds, y_cols)  # unchanged

    avg_r2 = float(metrics_df["R2 Score"].mean())
    grade, emoji = grade_r2(avg_r2)  # unchanged

    per_target = [
        TargetMetric(
            name=c,
            r2=float(metrics_df.loc[c, "R2 Score"]),
            mae=float(metrics_df.loc[c, "MAE"]),
        )
        for c in y_cols
    ]

    return ModelPerformance(
        model_name=model_name,
        dataset_name=dataset_name,
        per_target=per_target,
        avg_r2=avg_r2,
        grade=grade,
        emoji=emoji,
        x_cols=x_cols,
        y_cols=y_cols,
    )
