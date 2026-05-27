"""
src/ui/session.py
==================
Centralised session-state initialisation for the Streamlit application.

Call ``init_session_state()`` once near the top of app.py.  All 14
session keys are defined here; their default values are the single source
of truth for the session contract.

Keys
----
df              — the currently active DataFrame (None until uploaded)
data_history    — {filename: DataFrame} in-session dataset cache
x_cols          — list of selected input feature column names
y_cols          — list of selected target column names
X_train         — scaled training feature array (numpy)
X_test          — scaled test feature array (numpy)
y_train         — scaled training target array (numpy)
y_test          — scaled test target array (numpy)
y_test_raw      — unscaled test target DataFrame (for metric computation)
scaler_x        — fitted StandardScaler for X
scaler_y        — fitted StandardScaler for Y
model           — trained IndustrialDAE instance (or None)
model_trained   — bool flag, True when a model is ready for inference
history         — list of training run metadata dicts
sim_history     — list of What-If simulation run dicts
loaded_sim      — the last loaded simulation scenario dict (or None)
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Default values — modify here to change the startup defaults app-wide
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "df":            None,
    "data_history":  {},
    "x_cols":        [],
    "y_cols":        [],
    "X_train":       None,
    "X_test":        None,
    "y_train":       None,
    "y_test":        None,
    "y_test_raw":    None,
    "scaler_x":      None,
    "scaler_y":      None,
    "model":         None,
    "model_trained": False,
    "history":       [],
    "sim_history":   [],
    "loaded_sim":    None,
}


def init_session_state() -> None:
    """
    Initialise every session key with its default value if not already set.

    Idempotent: safe to call on every Streamlit rerun.
    """
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
