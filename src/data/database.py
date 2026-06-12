"""
src/data/database.py
====================
SQLite-backed dataset versioning layer.

Each uploaded dataset is serialised as Parquet (via pyarrow) and stored as a
BLOB in the ``datasets`` table.  This lets users switch between datasets
without re-uploading files on every session restart.

Schema
------
datasets(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    UNIQUE,
    upload_time TEXT,
    num_rows    INTEGER,
    num_cols    INTEGER,
    data        BLOB
)

Public API
----------
init_db()
save_dataset_to_db(name, df)
list_datasets_from_db()          → list[tuple]
load_dataset_from_db(name)       → DataFrame | None
delete_dataset_from_db(name)
"""
from __future__ import annotations

import datetime
import io
import json
import sqlite3
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.settings import DB_PATH

# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create all required tables if they do not already exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    UNIQUE,
                upload_time TEXT,
                num_rows    INTEGER,
                num_cols    INTEGER,
                data        BLOB
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name   TEXT,
                algorithm    TEXT,
                created_at   TEXT,
                dataset_name TEXT,
                x_cols       TEXT,
                y_cols       TEXT,
                avg_r2       REAL,
                avg_rmse     REAL,
                avg_mae      REAL,
                file_path    TEXT
            )
            """
        )
        conn.commit()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve mixed-type object columns so PyArrow can serialise to Parquet.

    For each object column:
    - If all non-null values are numeric, cast to float.
    - Otherwise, cast every value to str (None preserved for nulls).
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        as_num = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna()
        if non_null.sum() == 0 or as_num[non_null].notna().all():
            df[col] = as_num
        else:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)
    return df


def save_dataset_to_db(name: str, df: pd.DataFrame) -> None:
    """
    Upsert a DataFrame into the database.

    If a dataset with the same *name* already exists it is replaced
    (INSERT OR REPLACE semantics).
    """
    blob = _sanitize_for_parquet(df).to_parquet()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO datasets
                (name, upload_time, num_rows, num_cols, data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, now, len(df), len(df.columns), blob),
        )
        conn.commit()


def list_datasets_from_db() -> List[Tuple]:
    """
    Return summary rows ordered by most recently uploaded.

    Each row is ``(name, upload_time, num_rows, num_cols)``.
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT name, upload_time, num_rows, num_cols "
            "FROM datasets ORDER BY upload_time DESC"
        ).fetchall()
    return rows


def load_dataset_from_db(name: str) -> Optional[pd.DataFrame]:
    """
    Retrieve a DataFrame by name.

    Returns ``None`` if the name is not found or if the stored Parquet blob is
    incompatible with the current PyArrow version (e.g. saved by an older
    version).  Callers should surface the None case to the user.
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT data FROM datasets WHERE name = ?", (name,)
        ).fetchone()
    if row:
        try:
            return pd.read_parquet(io.BytesIO(row[0]))
        except Exception:
            return None
    return None


def delete_dataset_from_db(name: str) -> None:
    """Remove a dataset record (and its Parquet blob) by name."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM datasets WHERE name = ?", (name,))
        conn.commit()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def save_model_to_registry(
    model_name: str,
    algorithm: str,
    dataset_name: str,
    x_cols: List[str],
    y_cols: List[str],
    avg_r2: float,
    avg_rmse: float,
    avg_mae: float,
    file_path: str,
) -> int:
    """Insert a model record into the registry. Returns the new row id."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO model_registry
                (model_name, algorithm, created_at, dataset_name,
                 x_cols, y_cols, avg_r2, avg_rmse, avg_mae, file_path)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                model_name, algorithm, now, dataset_name,
                json.dumps(x_cols), json.dumps(y_cols),
                round(float(avg_r2), 4), round(float(avg_rmse), 4), round(float(avg_mae), 4),
                file_path,
            ),
        )
        conn.commit()
        return cur.lastrowid


def list_models_from_registry() -> List[Dict]:
    """Return all model registry records ordered by most recent first."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, model_name, algorithm, created_at, dataset_name, "
            "x_cols, y_cols, avg_r2, avg_rmse, avg_mae, file_path "
            "FROM model_registry ORDER BY created_at DESC"
        ).fetchall()
    result = []
    for r in rows:
        result.append({
            "id":           r[0],
            "model_name":   r[1],
            "algorithm":    r[2],
            "created_at":   r[3],
            "dataset_name": r[4],
            "x_cols":       json.loads(r[5]) if r[5] else [],
            "y_cols":       json.loads(r[6]) if r[6] else [],
            "avg_r2":       r[7],
            "avg_rmse":     r[8],
            "avg_mae":      r[9],
            "file_path":    r[10],
        })
    return result


def delete_model_from_registry(model_id: int) -> None:
    """Remove a model registry entry by id."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM model_registry WHERE id = ?", (model_id,))
        conn.commit()
