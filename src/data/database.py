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
import sqlite3
from typing import List, Optional, Tuple

import pandas as pd

from config.settings import DB_PATH

# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create the datasets table if it does not already exist."""
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
        conn.commit()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


def save_dataset_to_db(name: str, df: pd.DataFrame) -> None:
    """
    Upsert a DataFrame into the database.

    If a dataset with the same *name* already exists it is replaced
    (INSERT OR REPLACE semantics).
    """
    blob = df.to_parquet()
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

    Returns ``None`` if the name is not found.
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT data FROM datasets WHERE name = ?", (name,)
        ).fetchone()
    if row:
        return pd.read_parquet(io.BytesIO(row[0]))
    return None


def delete_dataset_from_db(name: str) -> None:
    """Remove a dataset record (and its Parquet blob) by name."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM datasets WHERE name = ?", (name,))
        conn.commit()
