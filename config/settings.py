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
DEFAULT_LATENT_DIM: int = 5
DEFAULT_DROPOUT_RATE: float = 0.2
DEFAULT_MASKING_RATIO: float = 0.10
DEFAULT_EPOCHS: int = 150
DEFAULT_LR: float = 0.001
DEFAULT_WEIGHT_TO_PRED: float = 5.0
DEFAULT_BATCH_SIZE: int = 128

# Auto-train
AUTO_TRAIN_MAX_EPOCHS: int = 1000
AUTO_TRAIN_TARGET_R2: float = 0.80

# Early stopping & LR scheduling
DEFAULT_EARLY_STOP_PATIENCE: int = 20   # epochs without val improvement before stopping
DEFAULT_LR_PATIENCE: int = 10           # epochs without improvement before LR is halved

# ---------------------------------------------------------------------------
# What-If simulator
# ---------------------------------------------------------------------------
MAX_SWEEP_POINTS: int = 500
TREND_EPSILON: float = 1e-5

# ---------------------------------------------------------------------------
# Feature Selection Scoring
# ---------------------------------------------------------------------------
# Component weights (sum = 1.0)
# FQ removed — missing/variance handled upstream in preprocessing; VIF enforced via gate.
FS_WEIGHT_SELECTION_FREQ:       float = 0.30
FS_WEIGHT_PREDICTIVE_STRENGTH:  float = 0.50
FS_WEIGHT_FEATURE_QUALITY:      float = 0.00   # unused — kept for import compatibility
FS_WEIGHT_STABILITY:            float = 0.20

# Predictive Strength sub-weights — 5 active scoring methods (must sum to 1.0)
# Permutation Importance: robust model-agnostic signal, highest weight.
# Mutual Information: captures non-linear feature-target dependencies.
# Target Correlation: fast, reliable linear measure.
# mRMR: rewards relevance while penalising redundancy with already-selected features.
# Elastic Net: regularisation-based coefficient, sparse and stable.
FS_PS_CORR_WEIGHT:  float = 0.20   # Target Correlation
FS_PS_MI_WEIGHT:    float = 0.25   # Mutual Information
FS_PS_PERM_WEIGHT:  float = 0.30   # Permutation Importance
FS_PS_MRMR_WEIGHT:  float = 0.15   # mRMR
FS_PS_EN_WEIGHT:    float = 0.10   # Elastic Net

# Multi-Y threshold scaling: each extra Y target softens PS recommendation
# thresholds by this fraction (capped at 4 extra targets = 32% max softening).
FS_MULTI_Y_PS_SCALE: float = 0.08

# Recommendation thresholds
# Lowered by ~10-12 pts to compensate for FQ removal from FinalScore (~15pt average contribution).
# FQ quality gates removed from logic; VIF still enforced as a hard gate for Highly Recommended.
FS_HIGHLY_REC_MIN_FINAL:          float = 70.0
FS_HIGHLY_REC_MIN_PRED_STRENGTH:  float = 65.0
FS_HIGHLY_REC_MIN_QUALITY:        float = 60.0   # unused in logic; kept for import compatibility
FS_HIGHLY_REC_MAX_VIF:            float = 10.0

FS_RECOMMENDED_MIN_FINAL:         float = 50.0
FS_RECOMMENDED_MIN_PRED_STRENGTH: float = 45.0
FS_RECOMMENDED_MIN_QUALITY:       float = 40.0   # unused in logic; kept for import compatibility

FS_CONSIDER_MIN_FINAL:            float = 35.0

FS_WEAK_MAX_PRED_STRENGTH:        float = 30.0
FS_WEAK_MAX_QUALITY:               float = 20.0   # unused in logic; kept for import compatibility

# Stability bootstrap
FS_STABILITY_RUNS:       int   = 20
FS_STABILITY_SAMPLE_FRAC: float = 0.80
FS_STABILITY_MAX_ROWS:   int   = 3000

# ---------------------------------------------------------------------------
# Evaluation / grading thresholds
# ---------------------------------------------------------------------------
R2_EXCELLENT: float = 0.85
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
    "Preprocessing",
    "Feature Selection",
    "Train Model",
    "Predict",
]
NAVIGATION_ICONS: list = [
    "graph-up",
    "upload",
    "gear",
    "funnel",
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
        background: linear-gradient(180deg, #0f2a52 0%, #163660 50%, #0f2a52 100%) !important;
        border-right: 1px solid rgba(77, 166, 255, 0.20) !important;
    }

    /* Strip all white backgrounds from Streamlit sidebar wrappers */
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stSidebar"] [data-testid="stElementContainer"],
    [data-testid="stSidebar"] iframe,
    [data-testid="stSidebar"] ul,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] .nav {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Active nav icon turns bright white */
    [data-testid="stSidebar"] [aria-selected="true"] svg,
    [data-testid="stSidebar"] [aria-selected="true"] i {
        color: #ffffff !important;
        opacity: 1 !important;
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
