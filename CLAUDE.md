# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. There are no test files in this project.

## Architecture Overview

This is a Streamlit-based industrial ML dashboard for training and deploying **soft sensor models** — models that predict KPI targets (Y columns) from process sensor data (X columns). The primary model is a **Denoising Autoencoder (DAE)** built in PyTorch; alternative models include Random Forest, XGBoost, LightGBM, and LSTM.

### Layer Structure

**`app.py`** — Thin dispatcher. Bootstraps DB, session state, sidebar, and routes to page modules via `_PAGE_MAP`. No business logic lives here.

**`config/settings.py`** — Single source of truth for all constants: file paths (`DB_PATH`, `MODEL_DIR`), model hyperparameter defaults, training thresholds (`AUTO_TRAIN_TARGET_R2 = 0.85`, `R2_EXCELLENT = 0.90`), and UI page definitions.

**`src/data/`**
- `database.py` — SQLite CRUD for dataset versioning. DataFrames are stored as Parquet BLOBs in `dashboard.db`. Table schema: `id, name, upload_time, num_rows, num_cols, data`.
- `preprocessing.py` — Data cleaning pipeline: type coercion, per-feature stats, imputation (mean/median/ffill/bfill/zero), outlier treatment (IQR capping, Winsorization, Z-score removal), domain filters, then `split_and_scale()` which applies `StandardScaler` and returns `X_train, X_test, y_train, y_test`.

**`src/feature_selection/`**
- `auto_selector.py` — 12-method consensus engine. Methods span supervised stats (correlation, F-test, mutual info), tree importance (RF, XGBoost, LightGBM), regularization (Lasso, Elastic Net), wrappers (RFE, SFS, SBS), and PCA loadings. Confidence score = 60% × selection frequency + 40% × normalized score. Features are categorized as Highly Recommended / Recommended / Optional / Remove.
- `selector.py` — Legacy 3-method selector (kept for backward compatibility).

**`src/models/`**
- `architecture.py` — `IndustrialDAE` (PyTorch). Encoder (input → 128 → 64 → latent), Decoder (latent → 64 → 128 → input), Predictor (latent → 32 → 16 → output). Loss = MSE(reconstruction) + `weight_to_pred` × Huber(prediction).
- `wrappers.py` — Unified `predict_scaled(X_scaled) → np.ndarray` interface for DAEWrapper, SklearnWrapper, and LSTMWrapper. All code above this layer interacts with wrappers, never raw models.

**`src/training/`**
- `trainer.py` — DAE training loop. **No Streamlit imports** — progress is reported via callbacks so the UI can call this from any context. Supports auto-train mode (trains until R² > 0.85, max 2000 epochs), early stopping, and `ReduceLROnPlateau` scheduling. Returns `(model, loss_history)`.
- `train_sklearn.py` / `train_lstm.py` — Analogous trainers for sklearn models and LSTM.

**`src/evaluation/metrics.py`** — Computes RMSE, MAE, R², MAPE per target. `grade_r2(value)` returns label + emoji (Excellent ≥ 0.90, Good ≥ 0.75).

**`src/persistence/model_store.py`** — Saves/loads all model types to `saved_models/<name>/`. Each run directory contains `model.pth` or `model.pkl`, `scaler_x.pkl`, `scaler_y.pkl`, `columns.pkl` (`{x_cols, y_cols}`), and `metadata.pkl`.

**`src/simulation/what_if.py`** — Sensitivity analysis: sweeps one or more input features across a range while holding others at their mean, then labels trends (Increasing / Decreasing / Constant) per output KPI.

**`src/ui/`**
- `session.py` — Defines all 13 Streamlit session state keys with defaults (`df`, `x_cols`, `y_cols`, `X_train/test`, `y_train/test`, `scaler_x/y`, `model`, `model_trained`, `history`, `sim_history`).
- `layout.py` — `configure_page()` injects CSS theme; `render_sidebar()` builds navigation.
- `components.py` — Shared UI widgets: `render_kpi_cards()`, `render_system_status()`, `render_loss_curves()`.
- `pages/` — One module per navigation page (overview, upload, preprocess, feature_selection, train, predict, what_if, history, comparison).

### Key Conventions

- **No Streamlit in core modules.** Training and evaluation code (`src/training/`, `src/evaluation/`, `src/models/`) must remain framework-agnostic. Progress reporting uses callbacks.
- **All config values come from `config/settings.py`.** Never hardcode thresholds, paths, or hyperparameter defaults anywhere else.
- **Session state keys are defined once** in `src/ui/session.py`. Pages read/write from `st.session_state` using those keys.
- **Wrappers as the abstraction boundary.** Pages and evaluation code always call `wrapper.predict_scaled()`, never framework-specific model internals.
- **Dataset persistence is SQLite + Parquet.** New datasets must go through `database.py` APIs, not direct filesystem writes.

### User Workflow (page order)

Upload → Preprocess → Feature Selection → Train → Predict → What-If → History → Comparison
