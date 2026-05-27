"""
app.py — Soft Sensor Prediction System
=======================================
Thin Streamlit entry point.

Responsibilities
----------------
1. Configure the Streamlit page (title, layout, CSS theme)
2. Bootstrap the SQLite database schema
3. Initialise Streamlit session state
4. Render the sidebar navigation
5. Dispatch to the selected page module

All business logic lives in the src/ package:
    src/data/          — database and preprocessing
    src/models/        — IndustrialDAE architecture
    src/training/      — training loop
    src/evaluation/    — metrics computation
    src/simulation/    — What-If sensitivity analysis
    src/persistence/   — model save / load / list
    src/ui/            — layout, session, components, pages
"""

from src.data.database import init_db
from src.ui.layout import configure_page, render_sidebar
from src.ui.session import init_session_state

# ---- Page config & CSS theme (must be the first Streamlit call) ----
configure_page()

# ---- Bootstrap database schema ----
init_db()

# ---- Initialise session-state keys ----
init_session_state()

# ---- Sidebar navigation ----
selected = render_sidebar()

# ---- Page dispatch ----
from src.ui.pages import (  # noqa: E402  (imports after configure_page is intentional)
    comparison,
    history,
    overview,
    predict,
    preprocess,
    train,
    upload,
    what_if,
)

_PAGE_MAP = {
    "Overview":    overview.render,
    "Upload Data": upload.render,
    "Preprocess":  preprocess.render,
    "Train Model": train.render,
    "Predict":     predict.render,
    "What-If":     what_if.render,
    "History":     history.render,
    "Comparison":  comparison.render,
}

_PAGE_MAP[selected]()
