---
name: project-auto-feature-selection
description: Comprehensive Auto Feature Selection module added to Soft Sensor Prediction System — 12 methods, consensus voting, per-feature reasoning
metadata:
  type: project
---

Added a comprehensive Intelligent Auto Feature Selection engine to replace the old 3-method selector.

**Why:** User requested a production-grade feature selection system with 12 methods, consensus voting, per-feature explainability, and a rich Streamlit UI dashboard.

**How to apply:** When working on the feature selection or preprocessing parts of this codebase, the new module is the source of truth.

## Key Files Added/Changed
- `src/feature_selection/auto_selector.py` — NEW: 12-method engine + consensus + reasoning
- `src/ui/pages/preprocess.py` — UPDATED: replaced `_render_auto_selection` with `_render_intelligent_feature_selection` (5-tab dashboard)
- `requirements.txt` — added xgboost, lightgbm, shap

## Methods Implemented (auto_selector.py)
Supervised: Target Correlation, F-Test ANOVA, Mutual Information
Feature Importance: Random Forest, XGBoost (requires xgboost), LightGBM (requires lightgbm)
Intrinsic: Lasso (MultiTaskLassoCV for multi-output), Elastic Net (MultiTaskElasticNetCV)
Wrapper: RFE (LinearRegression base), Sequential Forward Selection, Sequential Backward Selection
Dimensionality: PCA Loadings Analysis

## Consensus Algorithm
ConfidenceScore = 60% × SelectionFrequency + 40% × AvgNormScore
- >= 70%: Highly Recommended
- >= 45%: Recommended
- >= 20%: Optional
- < 20%: Remove

## Performance Limits
- SFS forward: skipped if n_features > 50
- SFS backward: skipped if n_features > 30
- VIF: skipped if n_features > 80
- Wrapper methods sample up to 5,000 rows

## Public API
```python
from src.feature_selection.auto_selector import run_auto_feature_selection
result = run_auto_feature_selection(X_df, y_df, top_k=10, enabled_methods=None, progress_callback=cb)
# result.consensus_df, result.recommended_features, result.per_feature_reasoning, etc.
```
