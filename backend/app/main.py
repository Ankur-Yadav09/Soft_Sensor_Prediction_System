"""
backend/app/main.py
======================
FastAPI entry point. Runs alongside the existing Streamlit app (app.py,
port 8501) against the same dashboard.db / saved_models — see the migration
plan for the rollout strategy. Nothing in src/ or config/ is imported here
except via the service layer, and no module in src/ is modified for this.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import (
    datasets,
    feature_selection,
    jobs,
    overview,
    predict,
    preprocess,
    training,
)
from backend.app.core.config import CORS_ORIGINS

app = FastAPI(title="Soft Sensor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    from src.data.database import init_db  # unchanged

    init_db()


app.include_router(overview.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(preprocess.router, prefix="/api")
app.include_router(feature_selection.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(predict.router, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
