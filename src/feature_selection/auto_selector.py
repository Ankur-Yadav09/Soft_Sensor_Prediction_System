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

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, f_regression, mutual_info_regression
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    MultiTaskElasticNetCV,
    MultiTaskLassoCV,
    Ridge,
)
from sklearn.preprocessing import StandardScaler

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
_HIGH_CONF = 0.70   # >= 70 %  → Highly Recommended
_MED_CONF  = 0.45   # >= 45 %  → Recommended
_LOW_CONF  = 0.20   # >= 20 %  → Optional
                    # <  20 %  → Remove

_VIF_HIGH     = 10.0
_VIF_MODERATE =  5.0

_MAX_ROWS_WRAPPER    = 5_000   # rows sampled for RFE / SFS
_MAX_FEATURES_SFS    =   50    # skip SFS-forward if more features
_MAX_FEATURES_SFS_BK =   30    # skip SFS-backward if more features
_MAX_FEATURES_VIF    =   80    # skip VIF if more features

# Available method IDs (used as keys throughout)
ALL_METHOD_IDS = [
    "target_correlation",
    "f_test",
    "mutual_information",
    "rf_importance",
    "xgboost_importance",
    "lightgbm_importance",
    "lasso",
    "elasticnet",
    "rfe",
    "sfs_forward",
    "sfs_backward",
    "pca_analysis",
]

METHOD_LABELS: Dict[str, str] = {
    "target_correlation":   "Target Correlation",
    "f_test":               "F-Test (ANOVA)",
    "mutual_information":   "Mutual Information",
    "rf_importance":        "Random Forest Importance",
    "xgboost_importance":   "XGBoost Importance",
    "lightgbm_importance":  "LightGBM Importance",
    "lasso":                "Lasso Regression",
    "elasticnet":           "Elastic Net",
    "rfe":                  "Recursive Feature Elimination",
    "sfs_forward":          "Sequential Forward Selection",
    "sfs_backward":         "Sequential Backward Selection",
    "pca_analysis":         "PCA Loadings Analysis",
}

METHOD_CATEGORIES: Dict[str, str] = {
    "target_correlation":   "Supervised",
    "f_test":               "Supervised",
    "mutual_information":   "Supervised",
    "rf_importance":        "Feature Importance",
    "xgboost_importance":   "Feature Importance",
    "lightgbm_importance":  "Feature Importance",
    "lasso":                "Intrinsic",
    "elasticnet":           "Intrinsic",
    "rfe":                  "Wrapper",
    "sfs_forward":          "Wrapper",
    "sfs_backward":         "Wrapper",
    "pca_analysis":         "Dimensionality Reduction",
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
    )


# ---------------------------------------------------------------------------
# Method 1 – Target Correlation (Supervised)
# ---------------------------------------------------------------------------

def _m_target_correlation(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        n_t = y2.shape[1]
        raw: Dict[str, float] = {}
        signs: Dict[str, str] = {}
        for i, feat in enumerate(names):
            cors = [float(np.corrcoef(X[:, i], y2[:, j])[0, 1]) for j in range(n_t)]
            cors = [0.0 if np.isnan(c) else c for c in cors]
            raw[feat] = float(np.mean([abs(c) for c in cors]))
            pos = sum(1 for c in cors if c >= 0)
            signs[feat] = "positive" if pos >= n_t / 2 else "negative"
        return _build_result(
            "target_correlation", raw, names, top_k,
            notes=f"Avg |Pearson r| with {n_t} target(s)",
            metadata={"signs": signs},
        )
    except Exception as e:
        return _failed("target_correlation", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 2 – F-Test ANOVA (Supervised)
# ---------------------------------------------------------------------------

def _m_f_test(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        n_t = y2.shape[1]
        f_matrix, p_matrix = [], []
        for j in range(n_t):
            fv, pv = f_regression(X, y2[:, j])
            f_matrix.append(np.nan_to_num(fv, nan=0.0))
            p_matrix.append(np.nan_to_num(pv, nan=1.0))
        avg_f = np.mean(f_matrix, axis=0)
        avg_p = np.mean(p_matrix, axis=0)
        raw = {feat: float(avg_f[i]) for i, feat in enumerate(names)}
        p_vals = {feat: float(avg_p[i]) for i, feat in enumerate(names)}
        return _build_result(
            "f_test", raw, names, top_k,
            notes=f"Avg F-statistic over {n_t} target(s)",
            metadata={"p_values": p_vals},
        )
    except Exception as e:
        return _failed("f_test", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 3 – Mutual Information (Supervised)
# ---------------------------------------------------------------------------

def _m_mutual_information(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        n_t = y2.shape[1]
        mi_matrix = [
            mutual_info_regression(X, y2[:, j], random_state=42)
            for j in range(n_t)
        ]
        avg_mi = np.mean(mi_matrix, axis=0)
        raw = {feat: float(avg_mi[i]) for i, feat in enumerate(names)}
        return _build_result(
            "mutual_information", raw, names, top_k,
            notes=f"Avg MI score over {n_t} target(s)",
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
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    if not _XGBOOST_AVAILABLE:
        return _failed("xgboost_importance", names, top_k, "xgboost not installed")
    try:
        y2 = _to_2d(y)
        imps = []
        for j in range(y2.shape[1]):
            m = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, random_state=42, verbosity=0
            )
            m.fit(X, y2[:, j])
            imps.append(m.feature_importances_)
        avg_imp = np.mean(imps, axis=0)
        raw = {feat: float(avg_imp[i]) for i, feat in enumerate(names)}
        return _build_result(
            "xgboost_importance", raw, names, top_k,
            notes="Gain-based importance (avg over targets)",
        )
    except Exception as e:
        return _failed("xgboost_importance", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 6 – LightGBM Importance (optional)
# ---------------------------------------------------------------------------

def _m_lightgbm_importance(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    if not _LIGHTGBM_AVAILABLE:
        return _failed("lightgbm_importance", names, top_k, "lightgbm not installed")
    try:
        y2 = _to_2d(y)
        imps = []
        for j in range(y2.shape[1]):
            m = lgb.LGBMRegressor(
                n_estimators=100, num_leaves=31, random_state=42, verbose=-1
            )
            m.fit(X, y2[:, j])
            imps.append(m.feature_importances_.astype(float))
        avg_imp = np.mean(imps, axis=0)
        raw = {feat: float(avg_imp[i]) for i, feat in enumerate(names)}
        return _build_result(
            "lightgbm_importance", raw, names, top_k,
            notes="Split-count importance (avg over targets)",
        )
    except Exception as e:
        return _failed("lightgbm_importance", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 7 – Lasso (Intrinsic)
# ---------------------------------------------------------------------------

def _m_lasso(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        n_t = y2.shape[1]
        cv = min(3, max(2, len(X) // 50))

        if n_t > 1:
            m = MultiTaskLassoCV(cv=cv, random_state=42, max_iter=2000)
            m.fit(Xs, y2)
            coefs = np.abs(m.coef_).mean(axis=0)  # (n_targets, n_features) → avg
        else:
            m = LassoCV(cv=cv, random_state=42, max_iter=2000)
            m.fit(Xs, y2.ravel())
            coefs = np.abs(m.coef_)

        raw = {feat: float(coefs[i]) for i, feat in enumerate(names)}
        selected_mask = {feat: coefs[i] > 1e-8 for i, feat in enumerate(names)}
        return _build_result(
            "lasso", raw, names, top_k,
            notes=f"Alpha={getattr(m, 'alpha_', '?'):.4f} (CV-selected)",
            metadata={"selected_mask": selected_mask},
        )
    except Exception as e:
        return _failed("lasso", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 8 – Elastic Net (Intrinsic)
# ---------------------------------------------------------------------------

def _m_elasticnet(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        n_t = y2.shape[1]
        cv = min(3, max(2, len(X) // 50))

        if n_t > 1:
            m = MultiTaskElasticNetCV(cv=cv, random_state=42, max_iter=2000)
            m.fit(Xs, y2)
            coefs = np.abs(m.coef_).mean(axis=0)
        else:
            m = ElasticNetCV(cv=cv, random_state=42, max_iter=2000)
            m.fit(Xs, y2.ravel())
            coefs = np.abs(m.coef_)

        raw = {feat: float(coefs[i]) for i, feat in enumerate(names)}
        selected_mask = {feat: coefs[i] > 1e-8 for i, feat in enumerate(names)}
        return _build_result(
            "elasticnet", raw, names, top_k,
            notes=f"Alpha={getattr(m, 'alpha_', '?'):.4f} (CV-selected)",
            metadata={"selected_mask": selected_mask},
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
) -> pd.DataFrame:

    successful = [r for r in method_results if r.success]
    n_methods = len(successful)
    if n_methods == 0:
        return pd.DataFrame()

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

    rows = []
    for feat in all_features:
        sel_count = sum(1 for r in successful if feat in r.selected_features)
        norm_scores = [r.all_scores.get(feat, 0.0) for r in successful]
        avg_norm = float(np.mean(norm_scores))

        freq = sel_count / n_methods
        confidence = round((0.6 * freq + 0.4 * avg_norm) * 100, 1)

        if confidence >= _HIGH_CONF * 100:
            recommendation = "Highly Recommended"
        elif confidence >= _MED_CONF * 100:
            recommendation = "Recommended"
        elif confidence >= _LOW_CONF * 100:
            recommendation = "Optional"
        else:
            recommendation = "Remove"

        vif = vif_lookup.get(feat, np.nan)
        avg_corr = avg_corr_lookup.get(feat, np.nan)
        p_val = p_lookup.get(feat, np.nan)

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
            "Feature":          feat,
            "SelectionCount":   sel_count,
            "TotalMethods":     n_methods,
            "SelectionFreq":    round(freq * 100, 1),
            "ConfidenceScore":  confidence,
            "AvgNormScore":     round(avg_norm, 4),
            "CorrWithTarget":   round(float(avg_corr), 4) if not np.isnan(avg_corr) else None,
            "VIF":              round(float(vif), 2) if not np.isnan(vif) else None,
            "PValue":           round(float(p_val), 4) if not np.isnan(p_val) else None,
            "LassoSelected":    lasso_sel,
            "ElasticNetSelected": en_sel,
            "Recommendation":   recommendation,
        })

    df = pd.DataFrame(rows).sort_values(
        ["ConfidenceScore", "AvgNormScore"], ascending=[False, False]
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

    rec   = row.get("Recommendation", "")
    conf  = row.get("ConfidenceScore", 0)
    n_sel = int(row.get("SelectionCount", 0))
    n_tot = int(row.get("TotalMethods", 1))

    lines.append(f"**{feat}** — _{rec}_  |  Confidence: **{conf}%**  ({n_sel}/{n_tot} methods)")
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

    # VIF
    vif = row.get("VIF")
    if vif is not None:
        if vif > _VIF_HIGH:
            vif_note = f"🔴 High multicollinearity (VIF = {vif:.1f}) — collinear with other features"
        elif vif > _VIF_MODERATE:
            vif_note = f"🟡 Moderate multicollinearity (VIF = {vif:.1f})"
        else:
            vif_note = f"🟢 Low multicollinearity (VIF = {vif:.1f})"
        lines.append(f"**Multicollinearity:** {vif_note}")

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

    # Methods that selected / rejected this feature
    sel_by = [r.name for r in method_results if r.success and feat in r.selected_features]
    not_by = [r.name for r in method_results if r.success and feat not in r.selected_features]
    if sel_by:
        lines.append(f"**Selected by:** {', '.join(sel_by)}")
    if not_by:
        lines.append(f"**Not selected by:** {', '.join(not_by)}")

    # Business interpretation
    lines.append("")
    lines.append("**Business Interpretation:**")
    avg_corr = row.get("CorrWithTarget", 0) or 0
    if rec == "Highly Recommended":
        lines.append(
            f"This feature shows {_corr_strength(avg_corr)} predictive signal and "
            "is consistently identified as important across multiple independent methods. "
            "Include it as a primary input for the soft sensor model."
        )
    elif rec == "Recommended":
        lines.append(
            f"This feature contributes meaningful predictive information "
            f"(selected by {n_sel}/{n_tot} methods). "
            "Recommended as a supporting input feature."
        )
    elif rec == "Optional":
        lines.append(
            "Marginal predictive value. Include only if domain knowledge strongly "
            "supports its relevance, or if the model underfits without it."
        )
    else:
        lines.append(
            "Minimal predictive signal detected. Removing this feature is unlikely "
            "to reduce model accuracy and will simplify the model."
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
        # Auto-select based on dataset size
        enabled_methods = [
            "target_correlation", "f_test", "mutual_information",
            "rf_importance", "lasso", "elasticnet", "rfe", "pca_analysis",
        ]
        if len(names) <= _MAX_FEATURES_SFS:
            enabled_methods.append("sfs_forward")
        if len(names) <= _MAX_FEATURES_SFS_BK:
            enabled_methods.append("sfs_backward")
        if _XGBOOST_AVAILABLE:
            enabled_methods.append("xgboost_importance")
        if _LIGHTGBM_AVAILABLE:
            enabled_methods.append("lightgbm_importance")

    # Method dispatcher
    method_dispatch = {
        "target_correlation":  lambda: _m_target_correlation(X_vals, y_2d, names, top_k),
        "f_test":              lambda: _m_f_test(X_vals, y_2d, names, top_k),
        "mutual_information":  lambda: _m_mutual_information(X_vals, y_2d, names, top_k),
        "rf_importance":       lambda: _m_rf_importance(X_vals, y_2d, names, top_k),
        "xgboost_importance":  lambda: _m_xgboost_importance(X_vals, y_2d, names, top_k),
        "lightgbm_importance": lambda: _m_lightgbm_importance(X_vals, y_2d, names, top_k),
        "lasso":               lambda: _m_lasso(X_vals, y_2d, names, top_k),
        "elasticnet":          lambda: _m_elasticnet(X_vals, y_2d, names, top_k),
        "rfe":                 lambda: _m_rfe(X_vals, y_2d, names, top_k),
        "sfs_forward":         lambda: _m_sfs_forward(X_vals, y_2d, names, top_k),
        "sfs_backward":        lambda: _m_sfs_backward(X_vals, y_2d, names, top_k),
        "pca_analysis":        lambda: _m_pca_analysis(X_vals, names, top_k),
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

    consensus_df = _aggregate_consensus(
        method_results, names, top_k,
        vif_df, corr_with_target, f_result, ls_result, en_result,
    )

    # ---- 7. Categorise features -----------------------------------------
    recommended  = consensus_df[consensus_df["Recommendation"].isin(
        ["Highly Recommended", "Recommended"])]["Feature"].tolist()
    optional     = consensus_df[consensus_df["Recommendation"] == "Optional"]["Feature"].tolist()
    to_remove    = consensus_df[consensus_df["Recommendation"] == "Remove"]["Feature"].tolist()

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
