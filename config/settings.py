"""
config/settings.py
==================
Single source of truth for all constants, paths, and defaults used across
the Soft Sensor Prediction System.

No module at a lower layer (data, models, training, ui) should hard-code
any value that appears here.  All imports run top-down:
    config.settings  ←  src.*  ←  app.py
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
DB_PATH: str = "dashboard.db"
MODEL_DIR: str = "saved_models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Model architecture — encoder/decoder/predictor hidden sizes
# ---------------------------------------------------------------------------
ENCODER_HIDDEN_1: int = 128
ENCODER_HIDDEN_2: int = 64
DECODER_HIDDEN_1: int = 64
DECODER_HIDDEN_2: int = 128
PREDICTOR_HIDDEN_1: int = 32
PREDICTOR_HIDDEN_2: int = 16

# ---------------------------------------------------------------------------
# Training defaults (reflected in the Streamlit widgets as initial values)
# ---------------------------------------------------------------------------
DEFAULT_LATENT_DIM: int = 15
DEFAULT_DROPOUT_RATE: float = 0.2
DEFAULT_MASKING_RATIO: float = 0.10
DEFAULT_EPOCHS: int = 150
DEFAULT_LR: float = 0.001
DEFAULT_WEIGHT_TO_PRED: float = 5.0
DEFAULT_BATCH_SIZE: int = 128

# Auto-train
AUTO_TRAIN_MAX_EPOCHS: int = 2000
AUTO_TRAIN_TARGET_R2: float = 0.85

# ---------------------------------------------------------------------------
# What-If simulator
# ---------------------------------------------------------------------------
MAX_SWEEP_POINTS: int = 500
TREND_EPSILON: float = 1e-5

# ---------------------------------------------------------------------------
# Evaluation / grading thresholds
# ---------------------------------------------------------------------------
R2_EXCELLENT: float = 0.90
R2_GOOD: float = 0.75

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
PAGE_TITLE: str = "Multi X-Y | Industrial DAE"
PAGE_LAYOUT: str = "wide"
SIDEBAR_STATE: str = "expanded"

NAVIGATION_OPTIONS: list = [
    "Overview",
    "Upload Data",
    "Preprocess",
    "Train Model",
    "Predict",
    "What-If",
    "History",
    "Comparison",
]
NAVIGATION_ICONS: list = [
    "graph-up",
    "upload",
    "gear",
    "diagram-3",
    "graph-up-arrow",
    "magic",
    "clock-history",
    "bar-chart",
]

# ---------------------------------------------------------------------------
# CSS theme (complete premium industrial theme)
# ---------------------------------------------------------------------------
THEME_CSS: str = """
<style>
    /* Modern Industrial Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@400;600;800&display=swap');

    :root {
        --primary: #4da6ff;
        --secondary: #2b6cb0;
        --bg-dark: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --accent: #10b981;
    }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                    0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        width: 100% !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1),
                    0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
    }

    .stDataFrame, .stTable {
        background-color: var(--card-bg) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 10px !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    .st-emotion-cache-16idsys p {
        color: #94a3b8 !important;
    }

    /* Custom Cards */
    .status-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1rem;
    }
</style>
"""
