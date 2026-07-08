"""
backend/app/services/project_service.py
===========================================
Persists the one genuine state gap in the existing codebase: preprocessed
train/test split arrays, fitted scalers, and the x_cols/y_cols selection
that Streamlit only ever held in st.session_state between the Preprocess
and Train steps. Persisted to artifacts/<project_id>/, mirroring the exact
directory-per-artifact + metadata convention src.persistence.model_store
already uses for saved_models/<name>/ — a backend-owned filesystem
convention, not a new database table, so src/data/database.py stays
untouched. See §2 ("Identifiers") of the migration plan.

Every later page (Train, Predict) takes an explicit project_id and reloads
these artifacts fresh from disk — nothing is cached in server memory
between requests.
"""
from __future__ import annotations

import datetime
import json
import os
import pickle
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import HTTPException

PROJECT_DIR = "artifacts"


@dataclass
class ProjectArtifacts:
    project_id: str
    dataset_name: str
    x_cols: List[str]
    y_cols: List[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_test_raw: pd.DataFrame
    scaler_x: Any
    scaler_y: Any
    config: Dict[str, Any]


def create_project(
    dataset_name: str,
    x_cols: List[str],
    y_cols: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_test_raw: pd.DataFrame,
    scaler_x: Any,
    scaler_y: Any,
    config: Dict[str, Any],
) -> str:
    project_id = uuid.uuid4().hex[:12]
    path = os.path.join(PROJECT_DIR, project_id)
    os.makedirs(path, exist_ok=True)

    np.save(os.path.join(path, "X_train.npy"), X_train)
    np.save(os.path.join(path, "X_test.npy"), X_test)
    np.save(os.path.join(path, "y_train.npy"), y_train)
    np.save(os.path.join(path, "y_test.npy"), y_test)
    with open(os.path.join(path, "y_test_raw.pkl"), "wb") as fh:
        pickle.dump(y_test_raw, fh)
    with open(os.path.join(path, "scaler_x.pkl"), "wb") as fh:
        pickle.dump(scaler_x, fh)
    with open(os.path.join(path, "scaler_y.pkl"), "wb") as fh:
        pickle.dump(scaler_y, fh)

    meta = {
        "project_id": project_id,
        "dataset_name": dataset_name,
        "x_cols": x_cols,
        "y_cols": y_cols,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "config": config,
    }
    with open(os.path.join(path, "metadata.json"), "w") as fh:
        json.dump(meta, fh)

    return project_id


def load_project(project_id: str) -> ProjectArtifacts:
    path = os.path.join(PROJECT_DIR, project_id)
    meta_path = os.path.join(path, "metadata.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    with open(meta_path) as fh:
        meta = json.load(fh)

    with open(os.path.join(path, "y_test_raw.pkl"), "rb") as fh:
        y_test_raw = pickle.load(fh)
    with open(os.path.join(path, "scaler_x.pkl"), "rb") as fh:
        scaler_x = pickle.load(fh)
    with open(os.path.join(path, "scaler_y.pkl"), "rb") as fh:
        scaler_y = pickle.load(fh)

    return ProjectArtifacts(
        project_id=project_id,
        dataset_name=meta["dataset_name"],
        x_cols=meta["x_cols"],
        y_cols=meta["y_cols"],
        X_train=np.load(os.path.join(path, "X_train.npy")),
        X_test=np.load(os.path.join(path, "X_test.npy")),
        y_train=np.load(os.path.join(path, "y_train.npy")),
        y_test=np.load(os.path.join(path, "y_test.npy")),
        y_test_raw=y_test_raw,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        config=meta.get("config", {}),
    )


def list_projects() -> List[dict]:
    if not os.path.isdir(PROJECT_DIR):
        return []
    projects = []
    for name in os.listdir(PROJECT_DIR):
        meta_path = os.path.join(PROJECT_DIR, name, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as fh:
                    projects.append(json.load(fh))
            except Exception:
                pass
    return sorted(projects, key=lambda m: m["created_at"], reverse=True)
