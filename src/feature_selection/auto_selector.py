"""
src/feature_selection/auto_selector.py
=======================================
Intelligent Auto Feature Selection Engine — 12 methods, consensus voting,
per-feature reasoning, VIF analysis, and final ranked recommendations.

Categories
----------
Supervised      : Target Correlation, F-Test, Mutual Information
Feature Importance: Random Forest, XGBoost*, LightGBM*
Intrinsic       : Lasso, Elastic Net
Wrapper         : RFE, Sequential Forward Selection, Sequential Backward Selection*
Dimensionality  : PCA Loadings

* Optional / conditional based on availability or dataset size.

Public API
----------
run_auto_feature_selection(X_df, y_df, top_k, enabled_methods,
                            corr_threshold, vif_threshold)
    -> AutoSelectionResult
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance as _sklearn_perm_importance
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    MultiTaskElasticNetCV,
    MultiTaskLassoCV,
    Ridge,
)
from sklearn.preprocessing import StandardScaler

from config.settings import (
    FS_HIGHLY_REC_MAX_VIF,
    FS_HIGHLY_REC_MIN_FINAL,
    FS_HIGHLY_REC_MIN_PRED_STRENGTH,
    FS_HIGHLY_REC_MIN_QUALITY,
    FS_MULTI_Y_PS_SCALE,
    FS_PS_CORR_WEIGHT,
    FS_PS_MI_WEIGHT,
    FS_PS_MRMR_WEIGHT,
    FS_PS_PERM_WEIGHT,
    FS_PS_SHAP_WEIGHT,
    FS_PS_XGB_WEIGHT,
    FS_RECOMMENDED_MIN_FINAL,
    FS_RECOMMENDED_MIN_PRED_STRENGTH,
    FS_RECOMMENDED_MIN_QUALITY,
    FS_CONSIDER_MIN_FINAL,
    FS_SHAP_MAX_ROWS,
    FS_STABILITY_MAX_ROWS,
    FS_STABILITY_RUNS,
    FS_STABILITY_SAMPLE_FRAC,
    FS_WEAK_MAX_PRED_STRENGTH,
    FS_WEAK_MAX_QUALITY,
    FS_WEIGHT_FEATURE_QUALITY,
    FS_WEIGHT_PREDICTIVE_STRENGTH,
    FS_WEIGHT_SELECTION_FREQ,
    FS_WEIGHT_STABILITY,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional packages
# ---------------------------------------------------------------------------
try:
    from sklearn.feature_selection import SequentialFeatureSelector as _SFS
    _SFS_AVAILABLE = True
except ImportError:
    _SFS_AVAILABLE = False

try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VIF_HIGH     = 10.0
_VIF_MODERATE =  5.0

_MAX_ROWS_WRAPPER    = 5_000   # rows sampled for RFE / SFS
_MAX_FEATURES_SFS    =   50    # skip SFS-forward if more features
_MAX_FEATURES_SFS_BK =   30    # skip SFS-backward if more features
_MAX_FEATURES_VIF    =   80    # skip VIF if more features
_MAX_FEATURES_NEW    =  100    # skip permutation/SHAP if more features

# Available method IDs (used as keys throughout)
ALL_METHOD_IDS = [
    "target_correlation",
    "f_test",
    "mutual_information",
    "mrmr",
    "rf_importance",
    "xgboost_importance",
    "lightgbm_importance",
    "permutation_importance",
    "shap_importance",
    "lasso",
    "elasticnet",
    "rfe",
    "sfs_forward",
    "sfs_backward",
    "pca_analysis",
]

# The 10 methods that contribute to Selection Frequency, Predictive Strength,
# and Final Score. The remaining 5 always run but are informational only.
_SCORING_METHOD_IDS: frozenset = frozenset([
    "target_correlation",
    "f_test",
    "mutual_information",
    "mrmr",
    "xgboost_importance",
    "permutation_importance",
    "shap_importance",
    "lasso",
    "elasticnet",
    "rfe",
])

# The 5 informational methods — always run automatically, shown for context,
# but excluded from SelectionFreq, PredictiveStrength, and FinalScore.
INFORMATIONAL_METHOD_IDS: frozenset = frozenset([
    "rf_importance",
    "lightgbm_importance",
    "sfs_forward",
    "sfs_backward",
    "pca_analysis",
])

METHOD_LABELS: Dict[str, str] = {
    "target_correlation":    "Target Correlation",
    "f_test":                "F-Test (ANOVA)",
    "mutual_information":    "Mutual Information",
    "mrmr":                  "mRMR",
    "rf_importance":         "Random Forest Importance (Informational)",
    "xgboost_importance":    "XGBoost Importance",
    "lightgbm_importance":   "LightGBM Importance (Informational)",
    "permutation_importance":"Permutation Importance",
    "shap_importance":       "SHAP Importance",
    "lasso":                 "Lasso Regression",
    "elasticnet":            "Elastic Net",
    "rfe":                   "Recursive Feature Elimination",
    "sfs_forward":           "Sequential Forward Selection (Informational)",
    "sfs_backward":          "Sequential Backward Selection (Informational)",
    "pca_analysis":          "PCA Loadings Analysis (Informational)",
}

METHOD_CATEGORIES: Dict[str, str] = {
    "target_correlation":    "Supervised",
    "f_test":                "Supervised",
    "mutual_information":    "Supervised",
    "mrmr":                  "Advanced Filter",
    "rf_importance":         "Feature Importance",
    "xgboost_importance":    "Feature Importance",
    "lightgbm_importance":   "Feature Importance",
    "permutation_importance":"Feature Importance",
    "shap_importance":       "Feature Importance",
    "lasso":                 "Intrinsic",
    "elasticnet":            "Intrinsic",
    "rfe":                   "Wrapper",
    "sfs_forward":           "Wrapper",
    "sfs_backward":          "Wrapper",
    "pca_analysis":          "Dimensionality Reduction",
}

# Predictive Strength method source IDs and their config weights.
# Only the 6 independent methods below contribute; RF and LGB are excluded.
_PS_METHOD_WEIGHTS: Dict[str, float] = {
    "target_correlation":    FS_PS_CORR_WEIGHT,
    "mutual_information":    FS_PS_MI_WEIGHT,
    "xgboost_importance":    FS_PS_XGB_WEIGHT,
    "permutation_importance":FS_PS_PERM_WEIGHT,
    "shap_importance":       FS_PS_SHAP_WEIGHT,
    "mrmr":                  FS_PS_MRMR_WEIGHT,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    name: str
    method_id: str
    category: str
    selected_features: List[str]
    all_scores: Dict[str, float]     # normalised 0–1 for ALL features
    raw_scores: Dict[str, float]     # original scale
    top_k: int
    notes: str = ""
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    per_target_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # per_target_scores[feature][y_col] = raw score for that Y target.
    # Empty dict when the method does not produce per-target breakdowns.


@dataclass
class AutoSelectionResult:
    method_results: List[MethodResult]
    consensus_df: pd.DataFrame           # ranked feature table
    correlation_matrix: pd.DataFrame     # X–X Pearson correlations
    corr_with_target: pd.DataFrame       # X–Y Pearson correlations
    vif_df: pd.DataFrame
    dataset_info: Dict[str, Any]
    recommended_features: List[str]      # Highly Recommended + Recommended
    optional_features: List[str]
    features_to_remove: List[str]
    per_feature_reasoning: Dict[str, str]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda s: s.fillna(s.mean()) if s.notna().any() else s.fillna(0))


def _drop_constant_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.std() > 0]


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax == vmin:
        return {k: 0.5 for k in scores}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in scores.items()}


def _to_2d(y: np.ndarray) -> np.ndarray:
    return y.reshape(-1, 1) if y.ndim == 1 else y


def _avg_y(y_2d: np.ndarray) -> np.ndarray:
    return y_2d.mean(axis=1)


def _sample(X: np.ndarray, y: np.ndarray, max_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), max_rows, replace=False)
    return X[idx], y[idx]


def _failed(method_id: str, names: List[str], top_k: int, err: str) -> MethodResult:
    zero = {f: 0.0 for f in names}
    return MethodResult(
        name=METHOD_LABELS[method_id],
        method_id=method_id,
        category=METHOD_CATEGORIES[method_id],
        selected_features=[],
        all_scores=zero,
        raw_scores=zero,
        top_k=top_k,
        notes=f"Failed: {err}",
        success=False,
    )


# ---------------------------------------------------------------------------
# Average Rank helper (informational only — does not affect scoring)
# ---------------------------------------------------------------------------

def _compute_avg_rank(
    features: List[str],
    scoring_results: List[MethodResult],
) -> Dict[str, float]:
    """
    For each feature compute its average rank across all successful scoring methods.
    Rank 1 = highest raw score within that method. Lower = consistently top ranked.
    """
    feature_ranks: Dict[str, List[int]] = {feat: [] for feat in features}
    for r in scoring_results:
        if not r.success:
            continue
        sorted_feats = sorted(features, key=lambda f: r.raw_scores.get(f, 0.0), reverse=True)
        for rank, feat in enumerate(sorted_feats, start=1):
            feature_ranks[feat].append(rank)
    return {
        feat: round(float(np.mean(ranks)), 2) if ranks else float(len(features))
        for feat, ranks in feature_ranks.items()
    }


# ---------------------------------------------------------------------------
# Non-voting structural analyses
# ---------------------------------------------------------------------------

def _compute_correlation_matrix(X_clean: pd.DataFrame) -> pd.DataFrame:
    return X_clean.corr(method="pearson").fillna(0)


def _compute_vif(X_clean: pd.DataFrame) -> pd.DataFrame:
    n_rows, n_feat = X_clean.shape
    names = X_clean.columns.tolist()

    if n_feat > _MAX_FEATURES_VIF:
        return pd.DataFrame({
            "Feature":   names,
            "VIF":       [np.nan] * n_feat,
            "VIF_Level": ["Skipped (> 80 features)"] * n_feat,
        })

    X = X_clean.values.astype(float)
    records = []
    use_ridge = n_feat >= n_rows * 0.5

    for i, feat in enumerate(names):
        y_i = X[:, i]
        others = np.delete(X, i, axis=1)

        if others.shape[1] == 0:
            records.append({"Feature": feat, "VIF": 1.0, "VIF_Level": "Low"})
            continue

        try:
            if use_ridge:
                pred = Ridge(alpha=1.0).fit(others, y_i).predict(others)
            else:
                X_int = np.column_stack([np.ones(n_rows), others])
                beta = np.linalg.lstsq(X_int, y_i, rcond=None)[0]
                pred = X_int @ beta

            ss_res = np.sum((y_i - pred) ** 2)
            ss_tot = np.sum((y_i - y_i.mean()) ** 2)
            r2 = max(0.0, min(1 - ss_res / (ss_tot + 1e-12), 0.9999))
            vif = round(min(1.0 / (1.0 - r2), 9999.0), 2)
        except Exception:
            vif = 9999.0

        level = "High" if vif > _VIF_HIGH else "Moderate" if vif > _VIF_MODERATE else "Low"
        records.append({"Feature": feat, "VIF": vif, "VIF_Level": level})

    return (
        pd.DataFrame(records)
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def _compute_target_correlations(X_clean: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
    y_f = _safe_fill(y_df)
    rows = []
    for x_col in X_clean.columns:
        row: Dict[str, Any] = {"Feature": x_col}
        for y_col in y_f.columns:
            r = X_clean[x_col].corr(y_f[y_col])
            row[y_col] = round(float(r) if not np.isnan(r) else 0.0, 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Feature")


# ---------------------------------------------------------------------------
# Helper: build a MethodResult from score dict
# ---------------------------------------------------------------------------

def _build_result(
    method_id: str,
    scores_raw: Dict[str, float],
    names: List[str],
    top_k: int,
    notes: str = "",
    metadata: Optional[Dict] = None,
    per_target_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> MethodResult:
    scores_norm = _normalize_scores(scores_raw)
    selected = sorted(scores_raw, key=lambda f: scores_raw[f], reverse=True)[:top_k]
    return MethodResult(
        name=METHOD_LABELS[method_id],
        method_id=method_id,
        category=METHOD_CATEGORIES[method_id],
        selected_features=selected,
        all_scores=scores_norm,
        raw_scores=scores_raw,
        top_k=top_k,
        notes=notes,
        success=True,
        metadata=metadata or {},
        per_target_scores=per_target_scores or {},
    )


# ---------------------------------------------------------------------------
# Method 1 – Target Correlation (Supervised)
# ---------------------------------------------------------------------------

def _m_target_correlation(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int,
    y_names: Optional[List[str]] = None,
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        n_t = y2.shape[1]
        y_cols = y_names if y_names and len(y_names) == n_t else [f"Y{j+1}" for j in range(n_t)]
        raw: Dict[str, float] = {}
        signs: Dict[str, str] = {}
        pts: Dict[str, Dict[str, float]] = {}
        for i, feat in enumerate(names):
            cors = [float(np.corrcoef(X[:, i], y2[:, j])[0, 1]) for j in range(n_t)]
            cors = [0.0 if np.isnan(c) else c for c in cors]
            raw[feat] = float(np.mean([abs(c) for c in cors]))
            pos = sum(1 for c in cors if c >= 0)
            signs[feat] = "positive" if pos >= n_t / 2 else "negative"
            pts[feat] = {y_cols[j]: round(abs(cors[j]), 5) for j in range(n_t)}
        return _build_result(
            "target_correlation", raw, names, top_k,
            notes=f"Avg |Pearson r| with {n_t} target(s)",
            metadata={"signs": signs},
            per_target_scores=pts,
        )
    except Exception as e:
        return _failed("target_correlation", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 2 – F-Test ANOVA (Supervised)
# ---------------------------------------------------------------------------

def _m_f_test(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int,
    y_names: Optional[List[str]] = None,
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        n_t = y2.shape[1]
        y_cols = y_names if y_names and len(y_names) == n_t else [f"Y{j+1}" for j in range(n_t)]
        f_matrix, p_matrix = [], []
        for j in range(n_t):
            fv, pv = f_regression(X, y2[:, j])
            f_matrix.append(np.nan_to_num(fv, nan=0.0))
            p_matrix.append(np.nan_to_num(pv, nan=1.0))
        avg_f = np.mean(f_matrix, axis=0)
        avg_p = np.mean(p_matrix, axis=0)
        raw = {feat: float(avg_f[i]) for i, feat in enumerate(names)}
        p_vals = {feat: float(avg_p[i]) for i, feat in enumerate(names)}
        pts: Dict[str, Dict[str, float]] = {
            feat: {y_cols[j]: round(float(f_matrix[j][i]), 4) for j in range(n_t)}
            for i, feat in enumerate(names)
        }
        return _build_result(
            "f_test", raw, names, top_k,
            notes=f"Avg F-statistic over {n_t} target(s)",
            metadata={"p_values": p_vals},
            per_target_scores=pts,
        )
    except Exception as e:
        return _failed("f_test", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 3 – Mutual Information (Supervised)
# ---------------------------------------------------------------------------

def _m_mutual_information(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int,
    y_names: Optional[List[str]] = None,
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        n_t = y2.shape[1]
        y_cols = y_names if y_names and len(y_names) == n_t else [f"Y{j+1}" for j in range(n_t)]
        mi_matrix = [
            mutual_info_regression(X, y2[:, j], random_state=42)
            for j in range(n_t)
        ]
        avg_mi = np.mean(mi_matrix, axis=0)
        raw = {feat: float(avg_mi[i]) for i, feat in enumerate(names)}
        pts: Dict[str, Dict[str, float]] = {
            feat: {y_cols[j]: round(float(mi_matrix[j][i]), 5) for j in range(n_t)}
            for i, feat in enumerate(names)
        }
        return _build_result(
            "mutual_information", raw, names, top_k,
            notes=f"Avg MI score over {n_t} target(s)",
            per_target_scores=pts,
        )
    except Exception as e:
        return _failed("mutual_information", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 4 – Random Forest Importance
# ---------------------------------------------------------------------------

def _m_rf_importance(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        # RF natively supports multi-output
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        )
        rf.fit(X, y2 if y2.shape[1] > 1 else y2.ravel())
        imp = rf.feature_importances_
        raw = {feat: float(imp[i]) for i, feat in enumerate(names)}
        return _build_result(
            "rf_importance", raw, names, top_k,
            notes="Mean Decrease Impurity (MDI)",
        )
    except Exception as e:
        return _failed("rf_importance", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 5 – XGBoost Importance (optional)
# ---------------------------------------------------------------------------

def _m_xgboost_importance(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int,
    y_names: Optional[List[str]] = None,
) -> MethodResult:
    if not _XGBOOST_AVAILABLE:
        return _failed("xgboost_importance", names, top_k, "xgboost not installed")
    try:
        y2 = _to_2d(y)
        n_t = y2.shape[1]
        y_cols = y_names if y_names and len(y_names) == n_t else [f"Y{j+1}" for j in range(n_t)]
        imps = []
        for j in range(n_t):
            m = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, random_state=42, verbosity=0
            )
            m.fit(X, y2[:, j])
            imps.append(m.feature_importances_)
        avg_imp = np.mean(imps, axis=0)
        raw = {feat: float(avg_imp[i]) for i, feat in enumerate(names)}
        pts: Dict[str, Dict[str, float]] = {
            feat: {y_cols[j]: round(float(imps[j][i]), 5) for j in range(n_t)}
            for i, feat in enumerate(names)
        }
        return _build_result(
            "xgboost_importance", raw, names, top_k,
            notes="Gain-based importance (avg over targets)",
            per_target_scores=pts,
        )
    except Exception as e:
        return _failed("xgboost_importance", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 6 – LightGBM Importance (optional)
# ---------------------------------------------------------------------------

def _m_lightgbm_importance(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int,
    y_names: Optional[List[str]] = None,
) -> MethodResult:
    if not _LIGHTGBM_AVAILABLE:
        return _failed("lightgbm_importance", names, top_k, "lightgbm not installed")
    try:
        y2 = _to_2d(y)
        n_t = y2.shape[1]
        y_cols = y_names if y_names and len(y_names) == n_t else [f"Y{j+1}" for j in range(n_t)]
        imps = []
        for j in range(n_t):
            m = lgb.LGBMRegressor(
                n_estimators=100, num_leaves=31, random_state=42, verbose=-1
            )
            m.fit(X, y2[:, j])
            imps.append(m.feature_importances_.astype(float))
        avg_imp = np.mean(imps, axis=0)
        raw = {feat: float(avg_imp[i]) for i, feat in enumerate(names)}
        pts: Dict[str, Dict[str, float]] = {
            feat: {y_cols[j]: round(float(imps[j][i]), 4) for j in range(n_t)}
            for i, feat in enumerate(names)
        }
        return _build_result(
            "lightgbm_importance", raw, names, top_k,
            notes="Split-count importance (avg over targets)",
            per_target_scores=pts,
        )
    except Exception as e:
        return _failed("lightgbm_importance", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 7 – Lasso (Intrinsic)
# ---------------------------------------------------------------------------

def _m_lasso(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int,
    y_names: Optional[List[str]] = None,
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        n_t = y2.shape[1]
        y_cols = y_names if y_names and len(y_names) == n_t else [f"Y{j+1}" for j in range(n_t)]
        cv = min(3, max(2, len(X) // 50))
        pts: Dict[str, Dict[str, float]] = {}

        if n_t > 1:
            m = MultiTaskLassoCV(cv=cv, random_state=42, max_iter=2000)
            m.fit(Xs, y2)
            coef_matrix = np.abs(m.coef_)          # shape (n_targets, n_features)
            coefs = coef_matrix.mean(axis=0)
            pts = {
                feat: {y_cols[j]: round(float(coef_matrix[j, i]), 5) for j in range(n_t)}
                for i, feat in enumerate(names)
            }
        else:
            m = LassoCV(cv=cv, random_state=42, max_iter=2000)
            m.fit(Xs, y2.ravel())
            coefs = np.abs(m.coef_)
            pts = {feat: {y_cols[0]: round(float(coefs[i]), 5)} for i, feat in enumerate(names)}

        raw = {feat: float(coefs[i]) for i, feat in enumerate(names)}
        selected_mask = {feat: coefs[i] > 1e-8 for i, feat in enumerate(names)}
        return _build_result(
            "lasso", raw, names, top_k,
            notes=f"Alpha={getattr(m, 'alpha_', '?'):.4f} (CV-selected)",
            metadata={"selected_mask": selected_mask},
            per_target_scores=pts,
        )
    except Exception as e:
        return _failed("lasso", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 8 – Elastic Net (Intrinsic)
# ---------------------------------------------------------------------------

def _m_elasticnet(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int,
    y_names: Optional[List[str]] = None,
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        n_t = y2.shape[1]
        y_cols = y_names if y_names and len(y_names) == n_t else [f"Y{j+1}" for j in range(n_t)]
        cv = min(3, max(2, len(X) // 50))
        pts: Dict[str, Dict[str, float]] = {}

        if n_t > 1:
            m = MultiTaskElasticNetCV(cv=cv, random_state=42, max_iter=2000)
            m.fit(Xs, y2)
            coef_matrix = np.abs(m.coef_)          # shape (n_targets, n_features)
            coefs = coef_matrix.mean(axis=0)
            pts = {
                feat: {y_cols[j]: round(float(coef_matrix[j, i]), 5) for j in range(n_t)}
                for i, feat in enumerate(names)
            }
        else:
            m = ElasticNetCV(cv=cv, random_state=42, max_iter=2000)
            m.fit(Xs, y2.ravel())
            coefs = np.abs(m.coef_)
            pts = {feat: {y_cols[0]: round(float(coefs[i]), 5)} for i, feat in enumerate(names)}

        raw = {feat: float(coefs[i]) for i, feat in enumerate(names)}
        selected_mask = {feat: coefs[i] > 1e-8 for i, feat in enumerate(names)}
        return _build_result(
            "elasticnet", raw, names, top_k,
            notes=f"Alpha={getattr(m, 'alpha_', '?'):.4f} (CV-selected)",
            metadata={"selected_mask": selected_mask},
            per_target_scores=pts,
        )
    except Exception as e:
        return _failed("elasticnet", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 9 – Recursive Feature Elimination (Wrapper)
# ---------------------------------------------------------------------------

def _m_rfe(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        Xs, ys = _sample(X, y2, _MAX_ROWS_WRAPPER)
        scaler = StandardScaler()
        Xss = scaler.fit_transform(Xs)
        k_sel = min(top_k, len(names))

        rfe = RFE(LinearRegression(), n_features_to_select=k_sel, step=1)
        rfe.fit(Xss, ys if ys.shape[1] > 1 else ys.ravel())

        # ranking_: 1 = selected, higher = eliminated earlier
        max_rank = int(rfe.ranking_.max())
        raw = {feat: float(max_rank - rfe.ranking_[i] + 1) for i, feat in enumerate(names)}
        selected_names = [names[i] for i, s in enumerate(rfe.support_) if s]
        return _build_result(
            "rfe", raw, names, top_k,
            notes=f"LinearRegression base, k={k_sel}",
            metadata={"support": {names[i]: bool(rfe.support_[i]) for i in range(len(names))}},
        )
    except Exception as e:
        return _failed("rfe", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 10 – Sequential Forward Selection (Wrapper)
# ---------------------------------------------------------------------------

def _m_sfs_forward(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    if not _SFS_AVAILABLE:
        return _failed("sfs_forward", names, top_k, "SequentialFeatureSelector not available (upgrade sklearn)")
    if len(names) > _MAX_FEATURES_SFS:
        return _failed("sfs_forward", names, top_k, f"Skipped: > {_MAX_FEATURES_SFS} features (performance limit)")
    try:
        y2 = _to_2d(y)
        Xs, ys = _sample(X, y2, _MAX_ROWS_WRAPPER)
        scaler = StandardScaler()
        Xss = scaler.fit_transform(Xs)
        k_sel = min(top_k, len(names) - 1)
        cv = min(3, max(2, len(Xss) // 50))

        sfs = _SFS(
            LinearRegression(), n_features_to_select=k_sel,
            direction="forward", scoring="r2", cv=cv,
        )
        sfs.fit(Xss, ys.ravel() if ys.shape[1] == 1 else ys)

        selected_mask = sfs.get_support()
        # Score selected features by their correlation with avg(y) (tiebreaker)
        y_avg = ys.mean(axis=1)
        raw: Dict[str, float] = {}
        for i, feat in enumerate(names):
            if selected_mask[i]:
                r = float(np.corrcoef(Xss[:, i], y_avg)[0, 1])
                raw[feat] = abs(r) if not np.isnan(r) else 0.0
            else:
                raw[feat] = 0.0
        return _build_result(
            "sfs_forward", raw, names, top_k,
            notes=f"Forward greedy, k={k_sel}, cv={cv}",
            metadata={"support": {names[i]: bool(selected_mask[i]) for i in range(len(names))}},
        )
    except Exception as e:
        return _failed("sfs_forward", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 11 – Sequential Backward Selection (Wrapper, conditional)
# ---------------------------------------------------------------------------

def _m_sfs_backward(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    if not _SFS_AVAILABLE:
        return _failed("sfs_backward", names, top_k, "SequentialFeatureSelector not available")
    if len(names) > _MAX_FEATURES_SFS_BK:
        return _failed("sfs_backward", names, top_k, f"Skipped: > {_MAX_FEATURES_SFS_BK} features (performance limit)")
    try:
        y2 = _to_2d(y)
        Xs, ys = _sample(X, y2, _MAX_ROWS_WRAPPER)
        scaler = StandardScaler()
        Xss = scaler.fit_transform(Xs)
        k_sel = min(top_k, len(names) - 1)
        cv = min(3, max(2, len(Xss) // 50))

        sfs = _SFS(
            LinearRegression(), n_features_to_select=k_sel,
            direction="backward", scoring="r2", cv=cv,
        )
        sfs.fit(Xss, ys.ravel() if ys.shape[1] == 1 else ys)

        selected_mask = sfs.get_support()
        y_avg = ys.mean(axis=1)
        raw: Dict[str, float] = {}
        for i, feat in enumerate(names):
            if selected_mask[i]:
                r = float(np.corrcoef(Xss[:, i], y_avg)[0, 1])
                raw[feat] = abs(r) if not np.isnan(r) else 0.0
            else:
                raw[feat] = 0.0
        return _build_result(
            "sfs_backward", raw, names, top_k,
            notes=f"Backward greedy, k={k_sel}, cv={cv}",
            metadata={"support": {names[i]: bool(selected_mask[i]) for i in range(len(names))}},
        )
    except Exception as e:
        return _failed("sfs_backward", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 12 – PCA Loadings Analysis (Dimensionality Reduction)
# ---------------------------------------------------------------------------

def _m_pca_analysis(
    X: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        n_comp = min(X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(Xs)

        cum_var = np.cumsum(pca.explained_variance_ratio_)
        # Components explaining first 95 % of variance
        n95 = int(np.searchsorted(cum_var, 0.95)) + 1
        n95 = min(n95, n_comp)

        loadings = np.abs(pca.components_[:n95])  # (n95, n_features)
        ev_ratio = pca.explained_variance_ratio_[:n95]

        # Weighted loading: sum of |loading| * explained_ratio for each feature
        weighted = (loadings * ev_ratio[:, None]).sum(axis=0)
        raw = {feat: float(weighted[i]) for i, feat in enumerate(names)}
        return _build_result(
            "pca_analysis", raw, names, top_k,
            notes=f"{n95} components explain {cum_var[n95-1]*100:.1f}% variance",
        )
    except Exception as e:
        return _failed("pca_analysis", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 13 – mRMR (Advanced Filter)
# ---------------------------------------------------------------------------

def _m_mrmr(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        y_avg = _avg_y(y2)

        # Relevance: average MI with each target
        mi_matrix = [
            mutual_info_regression(X, y2[:, j], random_state=42)
            for j in range(y2.shape[1])
        ]
        relevance = np.mean(mi_matrix, axis=0)  # shape (n_features,)

        # Greedy mRMR selection
        n_feat = len(names)
        selected_idx: List[int] = []
        remaining_idx = list(range(n_feat))

        corr_matrix = np.corrcoef(X.T)  # (n_feat, n_feat)

        for _ in range(min(top_k, n_feat)):
            if not remaining_idx:
                break
            if not selected_idx:
                best = int(np.argmax([relevance[i] for i in remaining_idx]))
                best_idx = remaining_idx[best]
            else:
                scores = []
                for i in remaining_idx:
                    redundancy = float(np.mean([abs(corr_matrix[i, s]) for s in selected_idx]))
                    scores.append(relevance[i] - redundancy)
                best = int(np.argmax(scores))
                best_idx = remaining_idx[best]
            selected_idx.append(best_idx)
            remaining_idx.remove(best_idx)

        # Score: 1st selected gets highest score, decreasing by rank
        raw: Dict[str, float] = {feat: 0.0 for feat in names}
        for rank, idx in enumerate(selected_idx):
            raw[names[idx]] = float(top_k - rank)

        return _build_result(
            "mrmr", raw, names, top_k,
            notes=f"Greedy mRMR, relevance–redundancy, {y2.shape[1]} target(s)",
        )
    except Exception as e:
        return _failed("mrmr", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 14 – Permutation Importance (Feature Importance)
# ---------------------------------------------------------------------------

def _m_permutation_importance(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        Xs, ys = _sample(X, y2, _MAX_ROWS_WRAPPER)
        y_avg = _avg_y(ys)

        rf = RandomForestRegressor(
            n_estimators=50, max_features=0.5, random_state=42, n_jobs=-1
        )
        rf.fit(Xs, y_avg)

        perm = _sklearn_perm_importance(rf, Xs, y_avg, n_repeats=5, random_state=42)
        importances = perm.importances_mean
        importances = np.maximum(importances, 0.0)

        raw = {feat: float(importances[i]) for i, feat in enumerate(names)}
        return _build_result(
            "permutation_importance", raw, names, top_k,
            notes="RF base model, 5 repeats, mean importance",
        )
    except Exception as e:
        return _failed("permutation_importance", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 15 – SHAP Importance (Feature Importance)
# ---------------------------------------------------------------------------

def _m_shap_importance(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    if not _SHAP_AVAILABLE:
        return _failed("shap_importance", names, top_k, "shap not installed")
    try:
        y2 = _to_2d(y)
        Xs, ys = _sample(X, y2, _MAX_ROWS_WRAPPER)
        y_avg = _avg_y(ys)

        rf = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        rf.fit(Xs, y_avg)

        # Cap rows for SHAP computation
        n_shap = min(len(Xs), FS_SHAP_MAX_ROWS)
        rng = np.random.default_rng(42)
        shap_idx = rng.choice(len(Xs), n_shap, replace=False) if len(Xs) > n_shap else np.arange(len(Xs))
        X_shap = Xs[shap_idx]

        explainer = _shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_shap)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        raw = {feat: float(mean_abs_shap[i]) for i, feat in enumerate(names)}
        return _build_result(
            "shap_importance", raw, names, top_k,
            notes=f"TreeExplainer on RF, {n_shap} rows sampled",
        )
    except Exception as e:
        return _failed("shap_importance", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _compute_predictive_strength(
    features: List[str],
    method_results: List[MethodResult],
) -> Dict[str, float]:
    """Weighted combination of predictive method scores → 0–100 per feature."""
    result_by_id = {r.method_id: r for r in method_results if r.success}

    # Redistribute weight from missing/failed methods proportionally
    active_weights: Dict[str, float] = {}
    for mid, w in _PS_METHOD_WEIGHTS.items():
        if mid in result_by_id:
            active_weights[mid] = w

    total_w = sum(active_weights.values())
    if total_w == 0:
        return {f: 50.0 for f in features}

    ps: Dict[str, float] = {f: 0.0 for f in features}
    for mid, w in active_weights.items():
        norm_w = w / total_w
        scores = result_by_id[mid].all_scores  # already normalized 0–1
        for feat in features:
            ps[feat] += norm_w * scores.get(feat, 0.0)

    return {f: float(np.clip(ps[f] * 100, 0, 100)) for f in features}


def _compute_feature_quality(
    features: List[str],
    X_clean: pd.DataFrame,
    vif_df: pd.DataFrame,
    missing_pct_per_col: Dict[str, float],
) -> Dict[str, float]:
    """
    Feature Quality Score (0-100)

    Components:
    - VIF Score (50%)
    - Missing Value Score (30%)
    - Variance Score (20%)
    """

    vif_lookup = {}
    if not vif_df.empty and "VIF" in vif_df.columns:
        vif_lookup = dict(zip(vif_df["Feature"], vif_df["VIF"]))

    stds = X_clean.std()

    result = {}

    for feat in features:

        # =========================
        # 1. VIF SCORE (50%)
        # =========================
        vif = vif_lookup.get(feat, np.nan)

        if np.isnan(vif):
            vif_score = 70.0  # Neutral

        elif vif <= 5:
            vif_score = 100.0

        elif vif <= 10:
            vif_score = 80.0

        elif vif <= 20:
            vif_score = 50.0

        elif vif <= 30:
            vif_score = 20.0

        else:
            vif_score = 0.0

        # =========================
        # 2. MISSING SCORE (30%)
        # =========================
        miss_pct = missing_pct_per_col.get(feat, 0.0)

        # Convert to percentage if stored as fraction
        if miss_pct <= 1:
            miss_pct *= 100

        if miss_pct <= 1:
            miss_score = 100.0

        elif miss_pct <= 5:
            miss_score = 90.0

        elif miss_pct <= 10:
            miss_score = 75.0

        elif miss_pct <= 20:
            miss_score = 50.0

        elif miss_pct <= 30:
            miss_score = 25.0

        else:
            miss_score = 0.0

        # =========================
        # 3. VARIANCE SCORE (20%)
        # =========================
        std_val = float(stds.get(feat, 0.0))

        if std_val <= 0:
            var_score = 0.0

        elif std_val < 0.001:
            var_score = 20.0

        elif std_val < 0.01:
            var_score = 50.0

        elif std_val < 0.05:
            var_score = 80.0

        else:
            var_score = 100.0

        # =========================
        # FINAL QUALITY SCORE
        # =========================
        quality_score = (
            0.50 * vif_score +
            0.30 * miss_score +
            0.20 * var_score
        )

        result[feat] = round(quality_score, 2)

    return result


def _compute_stability_score(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    features: List[str],
    top_k: int,
    n_runs: int = FS_STABILITY_RUNS,
) -> Dict[str, float]:
    """
    Bootstrap Stability Score (0-100)

    Measures how consistently a feature is selected across
    multiple bootstrap samples.

    Improvements:
    - 6-method ensemble
    - 60% consensus threshold
    - Rank-weighted selection
    """

    try:
        n_total = len(X_df)

        sample_size = int(
            min(n_total, FS_STABILITY_MAX_ROWS)
            * FS_STABILITY_SAMPLE_FRAC
        )
        sample_size = max(sample_size, 20)

        X_clean = _drop_constant_cols(_safe_fill(X_df))
        y_filled = _safe_fill(y_df)

        names = X_clean.columns.tolist()

        X_vals = X_clean.values.astype(float)
        y_vals = y_filled.values.astype(float)

        y_2d = _to_2d(y_vals)

        k = min(top_k, len(names))

        stability_points = {f: 0.0 for f in features}

        rng = np.random.default_rng(42)

        for _ in range(n_runs):

            idx = rng.choice(
                n_total,
                sample_size,
                replace=True
            )

            Xb = X_vals[idx]
            yb = y_2d[idx]

            method_results = []

            methods = [
                lambda: _m_target_correlation(Xb, yb, names, k),
                lambda: _m_mutual_information(Xb, yb, names, k),
                lambda: _m_rf_importance(Xb, yb, names, k),
            ]

            for fn in methods:
                try:
                    res = fn()

                    if (
                        res is not None
                        and getattr(res, "success", False)
                    ):
                        method_results.append(
                            res.selected_features[:k]
                        )

                except Exception:
                    continue

            if len(method_results) < 3:
                continue

            required_votes = math.ceil(
                len(method_results) * 0.60
            )

            feature_points_this_run = {}

            for ranked_features in method_results:

                for rank, feat in enumerate(ranked_features):

                    if feat not in feature_points_this_run:
                        feature_points_this_run[feat] = {
                            "votes": 0,
                            "score": 0.0,
                        }

                    feature_points_this_run[feat]["votes"] += 1

                    # Rank weighting
                    rank_weight = (
                        (k - rank) / k
                    )

                    feature_points_this_run[feat]["score"] += rank_weight

            for feat, info in feature_points_this_run.items():

                if feat not in stability_points:
                    continue

                if info["votes"] >= required_votes:

                    normalized_score = (
                        info["score"]
                        / len(method_results)
                    )

                    stability_points[feat] += normalized_score

        result = {}

        for feat in features:

            score = (
                stability_points.get(feat, 0.0)
                / n_runs
            ) * 100

            result[feat] = round(
                float(np.clip(score, 0, 100)),
                2,
            )

        return result

    except Exception:

        return {
            f: 50.0
            for f in features
        }
        
        
def _assign_recommendation(
    final: float, pred_strength: float, quality: float, vif: Optional[float], correlation: float,
    n_targets: int = 1,
) -> str:
    """Multi-condition recommendation assignment with quality gate.

    PS thresholds scale down gently with additional Y targets to compensate
    for score compression from cross-target averaging (FS_MULTI_Y_PS_SCALE).
    FS_WEAK_MAX_PRED_STRENGTH is intentionally not scaled — truly weak stays weak.
    """
    scale = 1.0 - FS_MULTI_Y_PS_SCALE * min(max(n_targets - 1, 0), 4)
    effective_highly_rec_ps  = FS_HIGHLY_REC_MIN_PRED_STRENGTH  * scale
    effective_recommended_ps = FS_RECOMMENDED_MIN_PRED_STRENGTH * scale

    vif_val = np.inf if vif is None or np.isnan(vif) else float(vif)
    if abs(correlation) < 0.05 and pred_strength < 50:
        return "Weak Feature"

    if (pred_strength < FS_WEAK_MAX_PRED_STRENGTH or quality < FS_WEAK_MAX_QUALITY):
        return "Weak Feature"

    if (
        final >= FS_HIGHLY_REC_MIN_FINAL
        and pred_strength >= effective_highly_rec_ps
        and quality >= FS_HIGHLY_REC_MIN_QUALITY
        and vif_val < FS_HIGHLY_REC_MAX_VIF
    ):
        return "Highly Recommended"

    if (final >= FS_RECOMMENDED_MIN_FINAL and pred_strength >= effective_recommended_ps and quality >= FS_RECOMMENDED_MIN_QUALITY):
        return "Recommended"

    if final >= FS_CONSIDER_MIN_FINAL:
        return "Consider"

    return "Weak Feature"


# ---------------------------------------------------------------------------
# Consensus aggregation
# ---------------------------------------------------------------------------

def _aggregate_consensus(
    method_results: List[MethodResult],
    all_features: List[str],
    top_k: int,
    vif_df: pd.DataFrame,
    corr_with_target: pd.DataFrame,
    f_test_result: Optional[MethodResult],
    lasso_result: Optional[MethodResult],
    en_result: Optional[MethodResult],
    X_df: Optional[pd.DataFrame] = None,
    y_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    successful      = [r for r in method_results if r.success]
    scoring_results = [r for r in successful if r.method_id in _SCORING_METHOD_IDS]
    n_scoring       = len(scoring_results)
    if n_scoring == 0:
        return pd.DataFrame()

    n_targets = corr_with_target.shape[1] if not corr_with_target.empty else 1

    # Build VIF lookup
    vif_lookup: Dict[str, float] = {}
    if not vif_df.empty and "VIF" in vif_df.columns:
        vif_lookup = dict(zip(vif_df["Feature"], vif_df["VIF"]))

    # Build avg |corr with target| lookup
    avg_corr_lookup: Dict[str, float] = {}
    if not corr_with_target.empty:
        avg_corr_lookup = corr_with_target.abs().mean(axis=1).to_dict()

    # Build p-value lookup from F-test
    p_lookup: Dict[str, float] = {}
    if f_test_result and f_test_result.success:
        p_lookup = f_test_result.metadata.get("p_values", {})

    # --- New scoring components ---
    ps_scores      = _compute_predictive_strength(all_features, method_results)
    avg_rank_scores = _compute_avg_rank(all_features, scoring_results)

    # Missing % per feature for quality score
    missing_pct: Dict[str, float] = {}
    if X_df is not None:
        for col in all_features:
            if col in X_df.columns:
                missing_pct[col] = float(X_df[col].isnull().mean())

    x_for_quality = X_df[all_features] if (X_df is not None and all(f in X_df.columns for f in all_features)) else pd.DataFrame(columns=all_features)
    fq_scores   = _compute_feature_quality(all_features, x_for_quality, vif_df, missing_pct)

    stab_scores: Dict[str, float] = {}
    if X_df is not None and y_df is not None:
        stab_scores = _compute_stability_score(X_df, y_df, all_features, top_k)
    else:
        stab_scores = {f: 50.0 for f in all_features}

    rows = []
    for feat in all_features:
        sel_count   = sum(1 for r in scoring_results if feat in r.selected_features)
        norm_scores = [r.all_scores.get(feat, 0.0) for r in scoring_results]
        avg_norm    = float(np.mean(norm_scores)) if norm_scores else 0.0

        freq       = sel_count / n_scoring
        freq_score = freq * 100.0

        ps   = ps_scores.get(feat, 0.0)
        fq   = fq_scores.get(feat, 70.0)
        stab = stab_scores.get(feat, 50.0)

        # Prevent weak features from benefiting too much from consensus
        adjusted_freq_score = freq_score * (max(ps, 25.0) / 100.0)

        final_score = round(
            FS_WEIGHT_SELECTION_FREQ * adjusted_freq_score
            + FS_WEIGHT_PREDICTIVE_STRENGTH * ps
            + FS_WEIGHT_FEATURE_QUALITY     * fq
            + FS_WEIGHT_STABILITY           * stab,
            1,
        )

        vif = vif_lookup.get(feat, np.nan)
        avg_corr = avg_corr_lookup.get(feat, np.nan)
        p_val = p_lookup.get(feat, np.nan)

        recommendation = _assign_recommendation(
            final_score, ps, fq,
            vif if not np.isnan(vif) else None,
            avg_corr if not np.isnan(avg_corr) else 0.0,
            n_targets=n_targets,
        )

        # Lasso / ElasticNet selection flags
        lasso_sel = (
            lasso_result.metadata.get("selected_mask", {}).get(feat, False)
            if lasso_result and lasso_result.success else None
        )
        en_sel = (
            en_result.metadata.get("selected_mask", {}).get(feat, False)
            if en_result and en_result.success else None
        )

        rows.append({
            "Feature":              feat,
            "SelectionCount":       sel_count,
            "TotalMethods":         n_scoring,
            "SelectionFreq":        round(freq * 100, 1),
            "PredictiveStrength":   round(ps, 1),
            "FeatureQuality":       round(fq, 1),
            "StabilityScore":       round(stab, 1),
            "FinalScore":           final_score,
            "ConfidenceScore":      final_score,   # kept for backward compat
            "AvgNormScore":         round(avg_norm, 4),
            "AvgRank":              avg_rank_scores.get(feat, float(len(all_features))),
            "CorrWithTarget":       round(float(avg_corr), 4) if not np.isnan(avg_corr) else None,
            "VIF":                  round(float(vif), 2) if not np.isnan(vif) else None,
            "PValue":               round(float(p_val), 4) if not np.isnan(p_val) else None,
            "LassoSelected":        lasso_sel,
            "ElasticNetSelected":   en_sel,
            "Recommendation":       recommendation,
        })

    df = pd.DataFrame(rows).sort_values(
        ["FinalScore", "PredictiveStrength"], ascending=[False, False]
    ).reset_index(drop=True)
    df.index = range(1, len(df) + 1)
    df.index.name = "Rank"
    return df


# ---------------------------------------------------------------------------
# Per-feature reasoning generation
# ---------------------------------------------------------------------------

def _corr_strength(r: float) -> str:
    a = abs(r)
    if a >= 0.70: return "very strong"
    if a >= 0.50: return "strong"
    if a >= 0.30: return "moderate"
    if a >= 0.10: return "weak"
    return "very weak"


def _generate_reasoning(
    feat: str,
    row: pd.Series,
    method_results: List[MethodResult],
    corr_with_target: pd.DataFrame,
    vif_df: pd.DataFrame,
    f_result: Optional[MethodResult],
    rf_result: Optional[MethodResult],
) -> str:
    lines: List[str] = []

    rec        = row.get("Recommendation", "")
    final      = row.get("FinalScore", row.get("ConfidenceScore", 0))
    ps         = row.get("PredictiveStrength", 0)
    fq         = row.get("FeatureQuality", 0)
    stab       = row.get("StabilityScore", 0)
    freq_pct   = row.get("SelectionFreq", 0)
    n_sel      = int(row.get("SelectionCount", 0))
    n_tot      = int(row.get("TotalMethods", 1))
    avg_corr   = float(row.get("CorrWithTarget", 0) or 0)
    vif        = row.get("VIF")
    avg_rank   = row.get("AvgRank")

    # Score card table
    lines.append(f"**{feat}** — _{rec}_")
    lines.append("")
    lines.append("| Score Component | Value |")
    lines.append("|---|---|")
    lines.append(f"| **Final Score** | **{final:.1f}** |")
    lines.append(f"| Predictive Strength | {ps:.1f} |")
    lines.append(f"| Feature Quality | {fq:.1f} |")
    lines.append(f"| Stability Score | {stab:.1f} |")
    lines.append(f"| Selection Frequency | {freq_pct:.1f}% ({n_sel}/{n_tot} independent methods) |")
    if avg_rank is not None:
        lines.append(f"| Average Rank *(informational)* | {avg_rank:.1f} |")
    lines.append("")

    # Reason tags
    reason_lines: List[str] = []

    # Selection frequency reasons
    if freq_pct >= 75:
        reason_lines.append(f"✅ Selected by {n_sel} of {n_tot} independent methods (high consensus)")
    elif freq_pct >= 50:
        reason_lines.append(f"🔵 Selected by {n_sel} of {n_tot} independent methods (moderate consensus)")
    else:
        reason_lines.append(f"⚠️ Selected by only {n_sel} of {n_tot} independent methods (low consensus)")
        
    if freq_pct > 60 and ps < 40:
        reason_lines.append("⚠️ Weak evidence despite high selection frequency")

    # Average rank interpretation (informational)
    if avg_rank is not None:
        if avg_rank <= 3:
            reason_lines.append(f"✅ Consistently top ranked across methods (Avg Rank: {avg_rank:.1f})")
        elif avg_rank <= 7:
            reason_lines.append(f"🔵 Moderately ranked across methods (Avg Rank: {avg_rank:.1f})")
        else:
            reason_lines.append(f"⚠️ Lower ranked across methods (Avg Rank: {avg_rank:.1f})")

    # Predictive strength reasons
    if ps >= 70:
        if abs(avg_corr) < 0.2:
            reason_lines.append(
                "ℹ️ Predictive signal appears primarily non-linear rather than linear"
            )
        reason_lines.append(f"✅ High predictive power (Strength: {ps:.1f})")
    elif ps >= 50:
        reason_lines.append(f"🔵 Moderate predictive power (Strength: {ps:.1f})")
    else:
        reason_lines.append(f"🔴 Low predictive power (Strength: {ps:.1f})")

    # SHAP contribution
    shap_result = next((r for r in method_results if r.method_id == "shap_importance" and r.success), None)
    if shap_result:
        shap_norm = shap_result.all_scores.get(feat, 0.0)
        if shap_norm > 0.6:
            reason_lines.append("✅ Strong SHAP contribution")
        elif shap_norm > 0.3:
            reason_lines.append("🔵 Moderate SHAP contribution")

    # Permutation importance contribution
    perm_result = next((r for r in method_results if r.method_id == "permutation_importance" and r.success), None)
    if perm_result:
        perm_norm = perm_result.all_scores.get(feat, 0.0)
        if perm_norm > 0.6:
            reason_lines.append("✅ Strong permutation importance")
        elif perm_norm > 0.3:
            reason_lines.append("🔵 Moderate permutation importance")

    # mRMR redundancy
    mrmr_result = next((r for r in method_results if r.method_id == "mrmr" and r.success), None)
    if mrmr_result:
        mrmr_norm = mrmr_result.all_scores.get(feat, 0.0)
        if mrmr_norm > 0.6:
            reason_lines.append("✅ Low redundancy detected by mRMR")

    # VIF reasons
    if vif is not None:
        if vif > _VIF_HIGH:
            reason_lines.append(f"🔴 High multicollinearity detected (VIF = {vif:.1f})")
        elif vif > _VIF_MODERATE:
            reason_lines.append(f"⚠️ Moderate multicollinearity (VIF = {vif:.1f})")
        else:
            reason_lines.append(f"✅ Low multicollinearity (VIF = {vif:.1f})")

    # Correlation with target
    if abs(avg_corr) < 0.1:
        reason_lines.append("🔴 Low correlation with target")
    elif abs(avg_corr) >= 0.5:
        reason_lines.append(f"✅ Strong correlation with target (|r| = {avg_corr:.3f})")

    if fq >= 80:
        reason_lines.append(
            f"✅ Excellent feature quality (Score: {fq:.1f})"
        )

    elif fq < 40:
        reason_lines.append(
            f"⚠️ Poor feature quality (Score: {fq:.1f})"
        )
    # Stability
    if stab >= 75:
        reason_lines.append(f"✅ Stable across bootstrap runs ({stab:.0f}%)")
    elif stab < 40:
        reason_lines.append(f"⚠️ Unstable across bootstrap runs ({stab:.0f}%)")

    if reason_lines:
        lines.append("**Why this recommendation:**")
        lines.extend([f"- {r}" for r in reason_lines])
        lines.append("")

    # Correlation with each target
    if not corr_with_target.empty and feat in corr_with_target.index:
        cors = corr_with_target.loc[feat]
        parts = [f"`{col}`: r={val:+.3f} ({_corr_strength(val)})" for col, val in cors.items()]
        lines.append("**Correlation with target(s):** " + " | ".join(parts))

    # F-test significance
    p_val = row.get("PValue")
    if p_val is not None:
        sig = "✅ Statistically significant (p < 0.05)" if p_val < 0.05 else "⚠️ Not statistically significant (p ≥ 0.05)"
        lines.append(f"**Statistical Significance:** p = {p_val:.4f} — {sig}")

    # RF importance
    if rf_result and rf_result.success:
        rf_pct = rf_result.raw_scores.get(feat, 0.0) * 100
        lines.append(f"**RF Importance:** {rf_pct:.2f}% of total impurity reduction")

    # Lasso / Elastic Net selection
    ls = row.get("LassoSelected")
    en = row.get("ElasticNetSelected")
    reg_parts = []
    if ls is not None:
        reg_parts.append(f"Lasso: {'✅ Selected' if ls else '❌ Eliminated'}")
    if en is not None:
        reg_parts.append(f"Elastic Net: {'✅ Selected' if en else '❌ Eliminated'}")
    if reg_parts:
        lines.append("**Regularisation:** " + " | ".join(reg_parts))

    # Methods that selected / rejected this feature (scoring methods only)
    sel_by  = [r.name for r in method_results if r.success and r.method_id in _SCORING_METHOD_IDS and feat in r.selected_features]
    not_by  = [r.name for r in method_results if r.success and r.method_id in _SCORING_METHOD_IDS and feat not in r.selected_features]
    info_by = [r.name for r in method_results if r.success and r.method_id not in _SCORING_METHOD_IDS]
    if sel_by:
        lines.append(f"**Selected by:** {', '.join(sel_by)}")
    if not_by:
        lines.append(f"**Not selected by:** {', '.join(not_by)}")
    if info_by:
        lines.append(f"**Informational (not scored):** {', '.join(info_by)}")

    # Business interpretation
    lines.append("")
    lines.append("**Business Interpretation:**")
    
    if rec == "Highly Recommended":

        if abs(avg_corr) < 0.3 and ps >= 70:
            lines.append(
                "This feature exhibits weak linear correlation with the target but "
                "strong predictive value through model-based methods (SHAP, permutation "
                "importance, tree-based importance, etc.). "
                "Include it as a primary input for the soft sensor model."
            )

        elif abs(avg_corr) >= 0.3:
            lines.append(
                "This feature demonstrates both statistical and model-based predictive "
                "strength and should be considered a primary input for the soft sensor model."
            )

        else:
            lines.append(
                "This feature is consistently identified as important across multiple "
                "independent methods and should be retained in the model."
            )
        
    elif rec == "Recommended":
        lines.append(
            f"This feature contributes meaningful predictive information "
            f"(selected by {n_sel}/{n_tot} methods). "
            "Recommended as a supporting input feature."
        )
        
    elif rec == "Consider":
        lines.append(
            "Marginal predictive value. Include only if domain knowledge strongly "
            "supports its relevance, or if the model underfits without it."
        )
    else:
        if ps < 30:
            lines.append(
                "Weak predictive signal detected across statistical and model-based methods. "
                "Removing this feature is unlikely to reduce model performance."
            )
        else:
            lines.append(
                "This feature did not meet the quality and recommendation thresholds "
                "required for inclusion in the final feature set."
            )

        if vif is not None and vif > _VIF_HIGH:
            lines.append(
                "The high VIF confirms this feature is largely redundant — "
                "its information is already captured by other features."
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset information summary
# ---------------------------------------------------------------------------

def _analyze_dataset_info(
    X_df: pd.DataFrame, y_df: pd.DataFrame, X_clean: pd.DataFrame
) -> Dict[str, Any]:
    dropped_const = [c for c in X_df.columns if c not in X_clean.columns]
    return {
        "n_rows":            len(X_df),
        "n_raw_features":    len(X_df.columns),
        "n_clean_features":  len(X_clean.columns),
        "n_targets":         len(y_df.columns),
        "target_names":      y_df.columns.tolist(),
        "constant_features": dropped_const,
        "missing_pct_x":     round(X_df.isnull().mean().mean() * 100, 2),
        "missing_pct_y":     round(y_df.isnull().mean().mean() * 100, 2),
        "xgboost_available": _XGBOOST_AVAILABLE,
        "lightgbm_available": _LIGHTGBM_AVAILABLE,
        "shap_available":    _SHAP_AVAILABLE,
        "sfs_available":     _SFS_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_auto_feature_selection(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    top_k: int = 10,
    enabled_methods: Optional[List[str]] = None,
    corr_threshold: float = 0.85,
    vif_threshold: float = 10.0,
    progress_callback=None,
) -> AutoSelectionResult:
    """
    Run the comprehensive auto feature selection pipeline.

    Parameters
    ----------
    X_df              : input feature DataFrame (raw, may have NaN)
    y_df              : target DataFrame (raw, may have NaN)
    top_k             : number of top features each method selects
    enabled_methods   : list of method IDs to run (None = auto-select)
    corr_threshold    : pairwise X-X correlation threshold for redundancy flag
    vif_threshold     : VIF threshold for multicollinearity flag (informational)
    progress_callback : optional callable(step: str) for progress reporting

    Returns
    -------
    AutoSelectionResult
    """

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    # ---- 1. Prepare clean arrays ----------------------------------------
    X_clean = _drop_constant_cols(_safe_fill(X_df))
    y_filled = _safe_fill(y_df)
    names: List[str] = X_clean.columns.tolist()
    top_k = min(top_k, len(names))

    X_vals = X_clean.values.astype(float)
    y_vals = y_filled.values.astype(float)
    y_2d   = _to_2d(y_vals)

    # ---- 2. Dataset info -------------------------------------------------
    info = _analyze_dataset_info(X_df, y_df, X_clean)

    # ---- 3. Structural analyses (non-voting) ----------------------------
    _progress("Computing correlation matrix…")
    corr_matrix = _compute_correlation_matrix(X_clean)

    _progress("Computing VIF (multicollinearity)…")
    vif_df = _compute_vif(X_clean)

    _progress("Computing target correlations…")
    corr_with_target = _compute_target_correlations(X_clean, y_df)

    # ---- 4. Determine which methods to run ------------------------------
    if enabled_methods is None:
        # Auto-select the 10 independent scoring methods
        enabled_methods = [
            "target_correlation", "f_test", "mutual_information",
            "mrmr", "lasso", "elasticnet", "rfe",
        ]
        if _XGBOOST_AVAILABLE:
            enabled_methods.append("xgboost_importance")
        if len(names) <= _MAX_FEATURES_NEW:
            enabled_methods.append("permutation_importance")
            if _SHAP_AVAILABLE:
                enabled_methods.append("shap_importance")

    y_names: List[str] = y_filled.columns.tolist()

    # Method dispatcher
    method_dispatch = {
        "target_correlation":   lambda: _m_target_correlation(X_vals, y_2d, names, top_k, y_names),
        "f_test":               lambda: _m_f_test(X_vals, y_2d, names, top_k, y_names),
        "mutual_information":   lambda: _m_mutual_information(X_vals, y_2d, names, top_k, y_names),
        "mrmr":                 lambda: _m_mrmr(X_vals, y_2d, names, top_k),
        "rf_importance":        lambda: _m_rf_importance(X_vals, y_2d, names, top_k),
        "xgboost_importance":   lambda: _m_xgboost_importance(X_vals, y_2d, names, top_k, y_names),
        "lightgbm_importance":  lambda: _m_lightgbm_importance(X_vals, y_2d, names, top_k, y_names),
        "permutation_importance": lambda: _m_permutation_importance(X_vals, y_2d, names, top_k),
        "shap_importance":      lambda: _m_shap_importance(X_vals, y_2d, names, top_k),
        "lasso":                lambda: _m_lasso(X_vals, y_2d, names, top_k, y_names),
        "elasticnet":           lambda: _m_elasticnet(X_vals, y_2d, names, top_k, y_names),
        "rfe":                  lambda: _m_rfe(X_vals, y_2d, names, top_k),
        "sfs_forward":          lambda: _m_sfs_forward(X_vals, y_2d, names, top_k),
        "sfs_backward":         lambda: _m_sfs_backward(X_vals, y_2d, names, top_k),
        "pca_analysis":         lambda: _m_pca_analysis(X_vals, names, top_k),
    }

    # ---- 5. Run selected methods ----------------------------------------
    method_results: List[MethodResult] = []
    for mid in enabled_methods:
        if mid not in method_dispatch:
            continue
        _progress(f"Running {METHOD_LABELS.get(mid, mid)}…")
        try:
            result = method_dispatch[mid]()
        except Exception as exc:
            result = _failed(mid, names, top_k, str(exc))
        method_results.append(result)

    # ---- 6. Consensus ---------------------------------------------------
    _progress("Aggregating consensus scores…")
    f_result  = next((r for r in method_results if r.method_id == "f_test"), None)
    rf_result = next((r for r in method_results if r.method_id == "rf_importance"), None)
    ls_result = next((r for r in method_results if r.method_id == "lasso"), None)
    en_result = next((r for r in method_results if r.method_id == "elasticnet"), None)

    _progress("Computing feature quality and stability scores…")
    consensus_df = _aggregate_consensus(
        method_results, names, top_k,
        vif_df, corr_with_target, f_result, ls_result, en_result,
        X_df=X_df, y_df=y_df,
    )

    # ---- 7. Categorise features -----------------------------------------
    recommended  = consensus_df[consensus_df["Recommendation"].isin(
        ["Highly Recommended", "Recommended"])]["Feature"].tolist()
    optional     = consensus_df[consensus_df["Recommendation"] == "Consider"]["Feature"].tolist()
    to_remove    = consensus_df[consensus_df["Recommendation"] == "Weak Feature"]["Feature"].tolist()

    # ---- 8. Generate per-feature reasoning ------------------------------
    _progress("Generating feature reasoning…")
    reasoning: Dict[str, str] = {}
    for _, row in consensus_df.reset_index().iterrows():
        feat = row["Feature"]
        reasoning[feat] = _generate_reasoning(
            feat, row, method_results,
            corr_with_target, vif_df,
            f_result, rf_result,
        )

    _progress("Done.")
    return AutoSelectionResult(
        method_results=method_results,
        consensus_df=consensus_df,
        correlation_matrix=corr_matrix,
        corr_with_target=corr_with_target,
        vif_df=vif_df,
        dataset_info=info,
        recommended_features=recommended,
        optional_features=optional,
        features_to_remove=to_remove,
        per_feature_reasoning=reasoning,
    )
