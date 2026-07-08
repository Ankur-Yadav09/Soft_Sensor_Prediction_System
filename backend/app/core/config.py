"""
backend/app/core/config.py
============================
Re-exports values from the existing config.settings — the single source of
truth for paths/thresholds/defaults. Nothing here is redefined; this module
only adds backend-specific settings (CORS origins) that have no equivalent
in the Streamlit app.
"""
from __future__ import annotations

from config.settings import DB_PATH, MODEL_DIR  # noqa: F401  (re-exported for backend use)

# Vite's default dev server port.
CORS_ORIGINS: list[str] = ["http://localhost:5173"]
