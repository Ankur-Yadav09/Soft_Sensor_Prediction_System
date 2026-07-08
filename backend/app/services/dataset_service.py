"""
backend/app/services/dataset_service.py
==========================================
Thin adapter over src.data.database — every function is parameterized by an
explicit dataset `name` (the existing UNIQUE key in the `datasets` table).
There is no "active dataset" concept here: nothing is cached in this
process between requests. See §2 ("Identifiers") of the migration plan.
"""
from __future__ import annotations

import io
import json

import pandas as pd
from fastapi import HTTPException, UploadFile

from src.data.database import (
    delete_dataset_from_db,
    list_datasets_from_db,
    load_dataset_from_db,
    save_dataset_to_db,
)
from backend.app.schemas.datasets import DatasetPreview, DatasetSummary


def _row_to_summary(row: tuple) -> DatasetSummary:
    name, uploaded_at, rows, cols = row
    return DatasetSummary(name=name, uploaded_at=uploaded_at, rows=rows, cols=cols)


def _find_summary(name: str) -> DatasetSummary:
    for row in list_datasets_from_db():
        if row[0] == name:
            return _row_to_summary(row)
    raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found.")


async def upload_dataset(file: UploadFile) -> DatasetSummary:
    file_bytes = await file.read()
    filename = file.filename or "upload"
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not parse '{filename}': {exc}"
        ) from exc

    save_dataset_to_db(filename, df)  # src.data.database — unchanged
    return _find_summary(filename)


def list_datasets() -> list[DatasetSummary]:
    return [_row_to_summary(row) for row in list_datasets_from_db()]


def get_dataset_preview(name: str) -> DatasetPreview:
    _find_summary(name)  # 404 if unknown, before touching the (possibly large) blob
    df = load_dataset_from_db(name)  # src.data.database — unchanged
    if df is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Could not load '{name}' — the stored data is incompatible "
                "with the current PyArrow version. Please delete this entry "
                "and re-upload the file."
            ),
        )
    head = df.head()
    # NaN/NaT/numpy scalars aren't directly JSON-serializable — round-trip
    # through pandas' own JSON encoder, which already handles all of this.
    head_records = json.loads(head.to_json(orient="records", date_format="iso"))
    return DatasetPreview(
        name=name,
        shape=list(df.shape),
        columns=df.columns.tolist(),
        head=head_records,
    )


def delete_dataset(name: str) -> None:
    _find_summary(name)  # 404 if unknown
    delete_dataset_from_db(name)  # src.data.database — unchanged
