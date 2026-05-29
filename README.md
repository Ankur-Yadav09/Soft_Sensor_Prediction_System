---
title: Industrial DAE Dashboard
emoji: 🏭
colorFrom: blue
colorTo: slate
sdk: streamlit
app_file: app.py
pinned: false
---

# Soft Sensor Prediction System

An end-to-end industrial soft sensor platform built on a multi-task Denoising Autoencoder (DAE). Upload process data, clean it, select features intelligently, train a model, evaluate predictions, and run what-if simulations — all from a single dashboard.

---

## Features

### Data Management
- **Excel Upload** — Upload `.xlsx` datasets; stored in SQLite for instant switching between versions.
- **Dataset Versioning** — Every uploaded file is persisted; reload any historical dataset at any time.

### Data Understanding
- **Feature-Level Analysis** — Select any column to see data type, record count, missing values, duplicates, outlier count, min/max/mean/median/std, skewness, kurtosis, and distribution summary.
- **Interactive Charts** — Histogram and box plot per feature rendered with Plotly.

### Basic Preprocessing
- **Remove Records** — Drop rows with missing values and/or duplicate records; live count preview before applying.
- **Missing Value Imputation** — Mean, Median, Mode, Forward Fill, Backward Fill, or Custom Value per selected column.
- **Outlier Treatment** — Six methods: IQR Capping, Z-Score Capping, Winsorization, Custom IQR Multiplier, Remove Outliers (IQR), Remove Outliers (Z-Score).
- **Domain Filters** — Per-tag min/max clipping to enforce physical process bounds.
- **Before / After Summary** — Record counts and full action log after every cleaning step.

### Intelligent Auto Feature Selection (12-Method Consensus Engine)
Automatically ranks and recommends the best input features (X) for any set of target variables (Y), with full explainability.

| Category | Methods |
|---|---|
| Supervised | Target Correlation, F-Test (ANOVA), Mutual Information |
| Feature Importance | Random Forest, XGBoost, LightGBM |
| Intrinsic / Regularisation | Lasso Regression (MultiTask), Elastic Net (MultiTask) |
| Wrapper | Recursive Feature Elimination (RFE), Sequential Forward Selection, Sequential Backward Selection |
| Dimensionality Reduction | PCA Loadings Analysis |

**Consensus Engine**
- Each method independently selects top-K features.
- `Confidence Score = 60% × Selection Frequency + 40% × Avg Normalised Score`
- Features are categorised as **Highly Recommended**, **Recommended**, **Optional**, or **Remove**.

**Outputs**
- Correlation matrix heatmap (feature–feature)
- Feature–target correlation heatmap
- VIF (Variance Inflation Factor) report and chart
- Per-method selection summary
- Per-feature reasoning: correlation score, statistical significance (p-value), VIF, regularisation selection, model contribution, and business interpretation.

**Multi-target support** — when multiple Y targets are selected, all methods average scores across targets (or use native multi-output implementations) to recommend features that are predictive for the entire target set.

### Model Training
- **IndustrialDAE Architecture** — Encoder → Latent Space → Decoder (reconstruction) + Predictor (KPI output).
- **Configurable Hyperparameters** — Latent dimension, dropout, masking ratio, learning rate, batch size, loss weight.
- **Auto-Train Mode** — Trains until avg R² > 0.85 (up to 2000 epochs) with early stopping.
- **Live Loss Curves** — Reconstruction and prediction loss plotted in real time.

### Prediction & Evaluation
- Inference on the held-out test set (20% split).
- Per-target metrics: **RMSE**, **MAE**, **R²**, **MAPE**.
- Visual outputs: actual vs. predicted line chart, scatter plot with 45° reference, residual histogram.

### What-If Sensitivity Analysis
- **Single-feature sweep** — Vary one sensor across a range while holding all others constant.
- **Multi-feature sweep** — Sweep multiple features independently and compare their effect on KPI outputs.
- Trend labelling: Increasing / Decreasing / Constant per target.

### History & Comparison
- Full training run history with hyperparameters and metrics.
- Cross-run comparison charts to identify the best performing configuration.

---

## Project Structure

```
Soft_Sensor_Prediction_System/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── config/
│   └── settings.py                 # All constants and defaults
├── src/
│   ├── data/
│   │   ├── database.py             # SQLite dataset versioning
│   │   └── preprocessing.py        # Imputation, outlier, scaling pipeline
│   ├── feature_selection/
│   │   ├── selector.py             # Original 3-method selector
│   │   └── auto_selector.py        # 12-method consensus engine
│   ├── models/
│   │   └── architecture.py         # IndustrialDAE (PyTorch)
│   ├── training/
│   │   └── trainer.py              # Training loop with auto-train
│   ├── evaluation/
│   │   └── metrics.py              # RMSE, MAE, R², MAPE
│   ├── persistence/
│   │   └── model_store.py          # Save / load model checkpoints
│   ├── simulation/
│   │   └── what_if.py              # Sensitivity analysis engine
│   └── ui/
│       ├── layout.py               # Page config, sidebar navigation
│       ├── session.py              # Session state initialisation
│       ├── components.py           # Reusable Streamlit components
│       └── pages/
│           ├── overview.py
│           ├── upload.py
│           ├── preprocess.py       # Data understanding, preprocessing, feature selection
│           ├── train.py
│           ├── predict.py
│           ├── what_if.py
│           ├── history.py
│           └── comparison.py
└── saved_models/                   # Trained model artifacts
```

---

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment on Hugging Face Spaces

1. Create a new Space and select **Streamlit** as the SDK.
2. Upload all project files.
3. The app will automatically build and deploy.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pandas` / `numpy` | Data manipulation |
| `torch` | PyTorch — DAE model |
| `scikit-learn` | Preprocessing, feature selection, metrics |
| `xgboost` | XGBoost feature importance |
| `lightgbm` | LightGBM feature importance |
| `shap` | SHAP explainability (optional) |
| `plotly` | Interactive visualisations |
| `matplotlib` | Loss curve plots |
| `pyarrow` | Parquet serialisation for dataset storage |
| `openpyxl` | Excel file reading |
