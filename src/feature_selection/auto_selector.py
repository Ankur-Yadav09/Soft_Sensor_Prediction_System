"""
src/feature_selection/auto_selector.py
=======================================
Intelligent Auto Feature Selection Engine — 5 core scoring methods, consensus
voting, per-feature reasoning, VIF analysis, and final ranked recommendations.

Core Scoring Methods (contribute to SelectionFreq, PredictiveStrength, FinalScore)
------------------------------------------------------------------------------------
Supervised      : Target Correlation, Mutual Information
Advanced Filter : mRMR (Maximum Relevance Minimum Redundancy)
Feature Importance: Permutation Importance
Intrinsic       : Elastic Net

Public API
----------
run_auto_feature_selection(X_df, y_df, top_k, enabled_methods,
                            corr_threshold, vif_threshold)
    -> AutoSelectionResult

run_per_target_auto_selection(X_df, y_df, top_k, enabled_methods,
                               corr_threshold, vif_threshold)
    -> PerTargetSelectionResult   # preferred for multi-Y datasets
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance as _sklearn_perm_importance
from sklearn.linear_model import (
    ElasticNetCV,
    MultiTaskElasticNetCV,
    Ridge,
)
from sklearn.preprocessing import StandardScaler

from config.settings import (
    FS_HIGHLY_REC_MAX_VIF,
    FS_HIGHLY_REC_MIN_FINAL,
    FS_HIGHLY_REC_MIN_PRED_STRENGTH,
    FS_MULTI_Y_PS_SCALE,
    # Predictive Strength sub-weights for the 5 core scoring methods
    FS_PS_CORR_WEIGHT,   # Target Correlation
    FS_PS_MI_WEIGHT,     # Mutual Information
    FS_PS_PERM_WEIGHT,   # Permutation Importance
    FS_PS_MRMR_WEIGHT,   # mRMR
    FS_PS_EN_WEIGHT,     # Elastic Net
    FS_RECOMMENDED_MIN_FINAL,
    FS_RECOMMENDED_MIN_PRED_STRENGTH,
    FS_CONSIDER_MIN_FINAL,
    FS_STABILITY_MAX_ROWS,
    FS_STABILITY_RUNS,
    FS_STABILITY_SAMPLE_FRAC,
    FS_WEAK_MAX_PRED_STRENGTH,
    FS_WEIGHT_FEATURE_QUALITY,
    FS_WEIGHT_PREDICTIVE_STRENGTH,
    FS_WEIGHT_SELECTION_FREQ,
    FS_WEIGHT_STABILITY,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VIF_HIGH     = 10.0
_VIF_MODERATE =  5.0

_MAX_ROWS_WRAPPER = 5_000   # rows sampled for permutation importance
_MAX_FEATURES_VIF =    80   # skip VIF computation if more features (performance)
_MAX_FEATURES_NEW =   100   # skip permutation importance if more features (performance)

# The 5 core scoring methods that drive SelectionFrequency, PredictiveStrength,
# and FinalScore. Only these method IDs are valid in the enabled_methods list.
_SCORING_METHOD_IDS: frozenset = frozenset([
    "target_correlation",       # Supervised: direct linear signal with each target
    "mutual_information",       # Supervised: captures non-linear target dependencies
    "mrmr",                     # Filter: maximum relevance, minimum redundancy
    "permutation_importance",   # Model-based: RF permutation drop in R²
    "elasticnet",               # Intrinsic: L1+L2 regularisation coefficient magnitude
])

METHOD_LABELS: Dict[str, str] = {
    "target_correlation":     "Target Correlation",
    "mutual_information":     "Mutual Information",
    "mrmr":                   "mRMR",
    "permutation_importance": "Permutation Importance",
    "elasticnet":             "Elastic Net",
}

METHOD_CATEGORIES: Dict[str, str] = {
    "target_correlation":     "Supervised",
    "mutual_information":     "Supervised",
    "mrmr":                   "Advanced Filter",
    "permutation_importance": "Feature Importance",
    "elasticnet":             "Intrinsic",
}

# Per-method contribution weights for the Predictive Strength composite score.
# Only the 5 core scoring methods are listed here; any method absent from this
# dict is silently excluded from the PS calculation even if it ran successfully.
# Weights must sum to 1.0 — redistribution is handled automatically at runtime
# if a method fails (see _compute_predictive_strength).
_PS_METHOD_WEIGHTS: Dict[str, float] = {
    "target_correlation":     FS_PS_CORR_WEIGHT,   # 0.20 — direct linear signal
    "mutual_information":     FS_PS_MI_WEIGHT,      # 0.25 — non-linear dependency
    "permutation_importance": FS_PS_PERM_WEIGHT,    # 0.30 — model-agnostic drop score
    "mrmr":                   FS_PS_MRMR_WEIGHT,    # 0.15 — relevance minus redundancy
    "elasticnet":             FS_PS_EN_WEIGHT,      # 0.10 — regularised coefficient
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


@dataclass
class PerTargetSelectionResult:
    """Result of running feature selection independently for each Y target.

    For multi-Y datasets this is the PRIMARY result object — it replaces the
    combined-average AutoSelectionResult.  All consensus fields are derived by
    aggregating per-target scores rather than averaging Y columns before scoring.
    """
    # --- per-target breakdown (unchanged) ---
    target_results: Dict[str, AutoSelectionResult]
    # union of Highly Recommended + Recommended across ALL targets (sorted by coverage desc)
    union_features: List[str]
    # union of Consider features NOT already in union_features
    optional_union: List[str]
    # feature → list of Y target names it was recommended for
    feature_target_map: Dict[str, List[str]]

    # --- aggregated result fields (mirrors AutoSelectionResult) ---
    consensus_df: pd.DataFrame              # coverage-based aggregated ranking
    recommended_features: List[str]         # HR + Rec from aggregated consensus
    optional_features: List[str]            # Consider from aggregated consensus
    features_to_remove: List[str]           # Weak Feature from aggregated consensus
    per_feature_reasoning: Dict[str, str]   # reasoning keyed by feature name
    method_results: List[MethodResult]      # per-method results averaged across targets
    corr_with_target: pd.DataFrame          # all Y columns concat'd from per-target results
    vif_df: pd.DataFrame                    # X–X VIF (same for all targets)
    dataset_info: Dict[str, Any]            # merged info with n_targets, target_names
    correlation_matrix: pd.DataFrame        # X–X Pearson (same for all targets)


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
# Method 2 – Mutual Information (Supervised)
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
# Method 3 – Elastic Net (Intrinsic)
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
# Method 4 – mRMR (Advanced Filter)
# ---------------------------------------------------------------------------

def _m_mrmr(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        y2 = _to_2d(y)

        # Relevance: average MI with each target
        mi_matrix = [
            mutual_info_regression(X, y2[:, j], random_state=42)
            for j in range(y2.shape[1])
        ]
        relevance = np.mean(mi_matrix, axis=0)  # shape (n_features,)

        # Precompute MI between every pair of X features for the redundancy term.
        # Using MI (not Pearson) catches nonlinear redundancy — two sensors that
        # are functionally equivalent but not linearly correlated are correctly
        # penalised. MI(Xi, Xj) is symmetric so we compute upper-triangle only.
        n_feat = len(names)
        mi_xx = np.zeros((n_feat, n_feat))
        for i in range(n_feat):
            mi_row = mutual_info_regression(X, X[:, i], random_state=42)
            mi_xx[i] = mi_row
            mi_xx[:, i] = mi_row  # symmetric

        # Greedy mRMR selection
        selected_idx: List[int] = []
        remaining_idx = list(range(n_feat))

        for _ in range(min(top_k, n_feat)):
            if not remaining_idx:
                break
            if not selected_idx:
                best = int(np.argmax([relevance[i] for i in remaining_idx]))
                best_idx = remaining_idx[best]
            else:
                scores = []
                for i in remaining_idx:
                    # Redundancy = mean MI between candidate and already-selected features
                    redundancy = float(np.mean([mi_xx[i, s] for s in selected_idx]))
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
            notes=f"Greedy mRMR, MI relevance + MI redundancy, {y2.shape[1]} target(s)",
        )
    except Exception as e:
        return _failed("mrmr", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Method 5 – Permutation Importance (Feature Importance)
# ---------------------------------------------------------------------------

def _m_permutation_importance(
    X: np.ndarray, y: np.ndarray, names: List[str], top_k: int
) -> MethodResult:
    try:
        y2 = _to_2d(y)
        Xs, ys = _sample(X, y2, _MAX_ROWS_WRAPPER)
        n_targets = ys.shape[1]

        # Train one RF per Y target and average importances so that a feature
        # which is a specialist for one target is not diluted by the others.
        # Averaging Y before fitting would compress a high-importance specialist
        # feature's signal by 1/n_targets.
        all_importances = []
        for j in range(n_targets):
            rf_j = RandomForestRegressor(
                n_estimators=50, max_features=0.5, random_state=42, n_jobs=-1
            )
            rf_j.fit(Xs, ys[:, j])
            perm_j = _sklearn_perm_importance(rf_j, Xs, ys[:, j], n_repeats=5, random_state=42)
            all_importances.append(np.maximum(perm_j.importances_mean, 0.0))

        importances = np.mean(all_importances, axis=0)

        raw = {feat: float(importances[i]) for i, feat in enumerate(names)}
        return _build_result(
            "permutation_importance", raw, names, top_k,
            notes=f"RF per target ({n_targets}), 5 repeats, mean importance",
        )
    except Exception as e:
        return _failed("permutation_importance", names, top_k, str(e))


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _compute_predictive_strength(
    features: List[str],
    method_results: List[MethodResult],
) -> Dict[str, float]:
    """Weighted combination of 5 core scoring method scores → 0–100 per feature.

    Uses _PS_METHOD_WEIGHTS to combine normalised scores from:
      Target Correlation, Mutual Information, Permutation Importance, mRMR,
      and Elastic Net.

    Weight redistribution: if any method failed or was not run, its weight is
    spread proportionally across the remaining active methods so the output
    remains on the 0–100 scale regardless of which methods succeeded.
    """
    result_by_id = {r.method_id: r for r in method_results if r.success}

    # Collect weights only for methods that actually produced results.
    # Methods absent from _PS_METHOD_WEIGHTS (e.g. informational methods)
    # are automatically excluded even if present in method_results.
    active_weights: Dict[str, float] = {}
    for mid, w in _PS_METHOD_WEIGHTS.items():
        if mid in result_by_id:
            active_weights[mid] = w

    total_w = sum(active_weights.values())
    if total_w == 0:
        # No core scoring method succeeded — return neutral mid-point
        return {f: 50.0 for f in features}

    ps: Dict[str, float] = {f: 0.0 for f in features}
    for mid, w in active_weights.items():
        # Normalise weight so active methods always sum to 1.0
        norm_w = w / total_w
        scores = result_by_id[mid].all_scores  # normalised 0–1 per feature
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
    multiple bootstrap samples using 3 diverse methods:
      - Target Correlation (linear association)
      - Mutual Information (nonlinear association)
      - Elastic Net (regularised necessity)

    Diversity matters: Corr + MI + EN span three different selection
    philosophies so their consensus reflects genuine robustness, not
    just shared correlation-with-target signal.
    60% consensus threshold (≥2/3 votes) + rank-weighted scoring.
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

            # Three bootstrap methods chosen for signal diversity:
            #   Corr  — linear association (fast, stable)
            #   MI    — nonlinear association (information-theoretic)
            #   EN    — regularised necessity (different selection philosophy)
            # Replacing RF Importance (which agreed with Corr+MI most of the time)
            # with Elastic Net gives genuinely independent votes and reduces
            # inflation of stability scores for merely correlated features.
            methods = [
                lambda: _m_target_correlation(Xb, yb, names, k),
                lambda: _m_mutual_information(Xb, yb, names, k),
                lambda: _m_elasticnet(Xb, yb, names, k),
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
    """Multi-condition recommendation assignment.

    The `quality` parameter (Feature Quality score) is accepted for call-site
    compatibility — _aggregate_from_per_target_results includes FQ in FinalScore —
    but is not used as a gate here because missing/variance issues are handled
    upstream in preprocessing.
    VIF is retained as the sole data-health gate for Highly Recommended.
    PS thresholds scale down gently with additional Y targets (FS_MULTI_Y_PS_SCALE).
    FS_WEAK_MAX_PRED_STRENGTH is intentionally not scaled — truly weak stays weak.
    """
    scale = 1.0 - FS_MULTI_Y_PS_SCALE * min(max(n_targets - 1, 0), 4)
    effective_highly_rec_ps  = FS_HIGHLY_REC_MIN_PRED_STRENGTH * scale
    effective_recommended_ps = FS_RECOMMENDED_MIN_PRED_STRENGTH * scale

    vif_val = np.inf if vif is None or np.isnan(vif) else float(vif)

    # Minimum signal floor — no target correlation and very low PS → Weak Feature
    if abs(correlation) < 0.05 and pred_strength < 50:
        return "Weak Feature"

    # PS floor — truly weak predictive signal
    if pred_strength < FS_WEAK_MAX_PRED_STRENGTH:
        return "Weak Feature"

    if (
        final >= FS_HIGHLY_REC_MIN_FINAL
        and pred_strength >= effective_highly_rec_ps
        and vif_val < FS_HIGHLY_REC_MAX_VIF
    ):
        return "Highly Recommended"

    if final >= FS_RECOMMENDED_MIN_FINAL and pred_strength >= effective_recommended_ps:
        return "Recommended"

    if final >= FS_CONSIDER_MIN_FINAL:
        return "Consider"

    return "Weak Feature"


def _dedup_multicollinear(
    consensus_df: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    corr_threshold: float,
) -> pd.DataFrame:
    """Post-scoring deduplication pass.

    For each pair of Recommended/Highly Recommended features whose X–X
    Pearson |r| exceeds corr_threshold, keep the one with the higher
    FinalScore and downgrade the other to 'Consider'.  Processes pairs
    greedily from highest |r| to lowest so the most redundant pairs are
    resolved first.  A downgraded feature is stamped with a
    'MulticollinearWith' column value for display and reasoning.
    """
    if correlation_matrix.empty or corr_threshold >= 1.0:
        return consensus_df

    keep_recs = {"Recommended", "Highly Recommended"}
    df = consensus_df.copy()

    rec_feats = set(df.loc[df["Recommendation"].isin(keep_recs), "Feature"].tolist())
    feat_list = [f for f in correlation_matrix.columns if f in rec_feats]

    pairs: List[tuple] = []
    for i in range(len(feat_list)):
        for j in range(i + 1, len(feat_list)):
            a, b = feat_list[i], feat_list[j]
            if a in correlation_matrix.index and b in correlation_matrix.columns:
                r = abs(float(correlation_matrix.loc[a, b]))
                if r > corr_threshold:
                    pairs.append((r, a, b))

    pairs.sort(reverse=True)

    downgraded: set = set()
    for r_val, a, b in pairs:
        if a in downgraded or b in downgraded:
            continue
        rec_a = df.loc[df["Feature"] == a, "Recommendation"].values
        rec_b = df.loc[df["Feature"] == b, "Recommendation"].values
        if not (
            len(rec_a) and rec_a[0] in keep_recs
            and len(rec_b) and rec_b[0] in keep_recs
        ):
            continue

        score_a = float(df.loc[df["Feature"] == a, "FinalScore"].values[0])
        score_b = float(df.loc[df["Feature"] == b, "FinalScore"].values[0])
        winner = a if score_a >= score_b else b
        loser  = b if score_a >= score_b else a

        df.loc[df["Feature"] == loser, "Recommendation"] = "Consider"
        df.loc[df["Feature"] == loser, "MulticollinearWith"] = (
            f"{winner} (|r|={r_val:.4f})"
        )
        downgraded.add(loser)

    return df


# ---------------------------------------------------------------------------
# Consensus aggregation
# ---------------------------------------------------------------------------

def _aggregate_consensus(
    method_results: List[MethodResult],
    all_features: List[str],
    top_k: int,
    vif_df: pd.DataFrame,
    corr_with_target: pd.DataFrame,
    en_result: Optional[MethodResult],
    X_df: Optional[pd.DataFrame] = None,
    y_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    # Separate scoring methods (5 core) from informational-only methods.
    # Only scoring_results contribute to SelectionFreq and FinalScore.
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

    # Compute the four score components
    ps_scores       = _compute_predictive_strength(all_features, method_results)
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

        # Selection frequency: fraction of the 5 core methods that included this
        # feature in their top-k selections, expressed as 0–100.
        freq       = sel_count / n_scoring
        freq_score = freq * 100.0

        ps   = ps_scores.get(feat, 0.0)
        fq   = fq_scores.get(feat, 70.0)   # still computed for VIF gate; not in FinalScore
        stab = stab_scores.get(feat, 50.0)

        # Dampen selection-frequency bonus for features with very low predictive
        # strength so a weak feature cannot reach a high FinalScore purely through
        # appearing in many method top-k lists.
        adjusted_freq_score = freq_score * (max(ps, 25.0) / 100.0)

        # FinalScore = SelectionFreq (30%) + PredictiveStrength (50%) + Stability (20%)
        # FQ is excluded: missing/variance are handled in preprocessing upstream.
        # VIF still enforces quality as a hard gate in _assign_recommendation().
        final_score = round(
            FS_WEIGHT_SELECTION_FREQ       * adjusted_freq_score
            + FS_WEIGHT_PREDICTIVE_STRENGTH * ps
            + FS_WEIGHT_STABILITY           * stab,
            1,
        )

        vif      = vif_lookup.get(feat, np.nan)
        avg_corr = avg_corr_lookup.get(feat, np.nan)

        recommendation = _assign_recommendation(
            final_score, ps, fq,
            vif if not np.isnan(vif) else None,
            avg_corr if not np.isnan(avg_corr) else 0.0,
            n_targets=n_targets,
        )

        # Elastic Net selection flag (binary: coefficient > threshold)
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
            "StabilityScore":       round(stab, 1),
            "FinalScore":           final_score,
            "ConfidenceScore":      final_score,   # kept for backward compat
            "AvgNormScore":         round(avg_norm, 4),
            "AvgRank":              avg_rank_scores.get(feat, float(len(all_features))),
            "CorrWithTarget":       round(float(avg_corr), 4) if not np.isnan(avg_corr) else None,
            "VIF":                  round(float(vif), 2) if not np.isnan(vif) else None,
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

    # Elastic Net selection flag
    en = row.get("ElasticNetSelected")
    if en is not None:
        lines.append(f"**Regularisation (Elastic Net):** {'✅ Selected' if en else '❌ Eliminated'}")

    # Methods that selected / rejected this feature across the 5 core scoring methods
    sel_by = [r.name for r in method_results if r.success and r.method_id in _SCORING_METHOD_IDS and feat in r.selected_features]
    not_by = [r.name for r in method_results if r.success and r.method_id in _SCORING_METHOD_IDS and feat not in r.selected_features]
    if sel_by:
        lines.append(f"**Selected by:** {', '.join(sel_by)}")
    if not_by:
        lines.append(f"**Not selected by:** {', '.join(not_by)}")

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
    _apply_dedup: bool = True,
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
    info["vif_skipped"] = len(names) > _MAX_FEATURES_VIF

    # ---- 3. Structural analyses (non-voting) ----------------------------
    _progress("Computing correlation matrix…")
    corr_matrix = _compute_correlation_matrix(X_clean)

    _progress("Computing VIF (multicollinearity)…")
    vif_df = _compute_vif(X_clean)

    _progress("Computing target correlations…")
    corr_with_target = _compute_target_correlations(X_clean, y_df)

    # ---- 4. Determine which methods to run ------------------------------
    if enabled_methods is None:
        # Default: run only the 5 core scoring methods.
        # Permutation importance is always included since it is always a
        # scoring method (the _MAX_FEATURES_NEW guard is retained for very
        # large feature sets where permutation becomes slow, but in practice
        # it runs for all reasonable dataset sizes).
        enabled_methods = [
            "target_correlation",    # supervised: Pearson correlation with target(s)
            "mutual_information",    # supervised: information-theoretic MI score
            "mrmr",                  # filter: maximum relevance – minimum redundancy
            "elasticnet",            # intrinsic: L1+L2 regularised regression
        ]
        # Permutation importance is model-based and scales with n_features;
        # skip only when the feature space is very large.
        if len(names) <= _MAX_FEATURES_NEW:
            enabled_methods.append("permutation_importance")

    info["permutation_skipped"] = "permutation_importance" not in enabled_methods

    y_names: List[str] = y_filled.columns.tolist()

    # Method dispatch table — the 5 core scoring methods only.
    method_dispatch = {
        "target_correlation":     lambda: _m_target_correlation(X_vals, y_2d, names, top_k, y_names),
        "mutual_information":     lambda: _m_mutual_information(X_vals, y_2d, names, top_k, y_names),
        "mrmr":                   lambda: _m_mrmr(X_vals, y_2d, names, top_k),
        "permutation_importance": lambda: _m_permutation_importance(X_vals, y_2d, names, top_k),
        "elasticnet":             lambda: _m_elasticnet(X_vals, y_2d, names, top_k, y_names),
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
    en_result = next((r for r in method_results if r.method_id == "elasticnet"), None)

    _progress("Computing feature quality and stability scores…")
    consensus_df = _aggregate_consensus(
        method_results, names, top_k,
        vif_df, corr_with_target, en_result,
        X_df=X_df, y_df=y_df,
    )
    if _apply_dedup:
        consensus_df = _dedup_multicollinear(consensus_df, corr_matrix, corr_threshold)

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
        )
        if _apply_dedup:
            mc = row.get("MulticollinearWith") if hasattr(row, "get") else None
            if mc and pd.notna(mc):
                reasoning[feat] += (
                    f" Downgraded from Recommended: highly correlated with {mc}"
                    " — include only one of this pair."
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


# ---------------------------------------------------------------------------
# Per-target aggregation helpers
# ---------------------------------------------------------------------------

def _get_ps(cdf: pd.DataFrame, feat: str) -> float:
    """Safely read PredictiveStrength for a feature from a consensus DataFrame."""
    if cdf.empty or "Feature" not in cdf.columns:
        return 0.0
    match = cdf.loc[cdf["Feature"] == feat, "PredictiveStrength"]
    return float(match.values[0]) if len(match) > 0 else 0.0


def _build_aggregate_method_results(
    target_results: Dict[str, "AutoSelectionResult"],
    scope_features: List[str],
) -> List[MethodResult]:
    """Build one MethodResult per core method by averaging scores across all per-target runs.

    Used exclusively by the ranking matrix heatmap (tab3) — does not affect FinalScore.
    """
    n_targets = len(target_results)
    aggregate: List[MethodResult] = []

    for mid in list(_SCORING_METHOD_IDS):
        # Collect successful per-target results for this method
        per_target: List[MethodResult] = []
        for res in target_results.values():
            mr = next((r for r in res.method_results if r.method_id == mid and r.success), None)
            if mr is not None:
                per_target.append(mr)

        if not per_target:
            continue

        # Average all_scores and raw_scores across targets (nan-safe mean)
        avg_all: Dict[str, float] = {}
        avg_raw: Dict[str, float] = {}
        for feat in scope_features:
            vals_all = [r.all_scores.get(feat, 0.0) for r in per_target]
            vals_raw = [r.raw_scores.get(feat, 0.0) for r in per_target]
            avg_all[feat] = float(np.nanmean(vals_all))
            avg_raw[feat] = float(np.nanmean(vals_raw))

        # A feature is "selected" if it appears in top-k of ≥50% of successful target runs
        threshold = max(1, len(per_target) / 2)
        sel_counts = {feat: sum(1 for r in per_target if feat in r.selected_features)
                      for feat in scope_features}
        selected = [f for f, c in sel_counts.items() if c >= threshold]
        selected.sort(key=lambda f: avg_all.get(f, 0.0), reverse=True)

        first = per_target[0]
        aggregate.append(MethodResult(
            name=first.name,
            method_id=mid,
            category=first.category,
            selected_features=selected,
            all_scores=avg_all,
            raw_scores=avg_raw,
            top_k=first.top_k,
            notes=f"Aggregated across {len(per_target)}/{n_targets} targets",
            success=True,
            metadata={},
            per_target_scores={},
        ))

    return aggregate


def _aggregate_from_per_target_results(
    target_results: Dict[str, "AutoSelectionResult"],
    scope_features: List[str],
    feature_target_map: Dict[str, List[str]],
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Build a coverage-based consensus DataFrame from per-target AutoSelectionResult objects.

    Selection Frequency is replaced by Coverage Ratio (fraction of Y targets that
    recommended the feature).  Predictive Strength is the nanmean of per-target PS.
    Feature Quality and Stability Score are computed once on the full X/Y datasets.
    The Final Score formula and recommendation thresholds are unchanged.
    """
    n_targets = len(target_results)
    y_cols = list(target_results.keys())

    # --- shared quality / stability scores (computed once for all features) ---
    vif_df: pd.DataFrame = pd.DataFrame()
    any_res = next(iter(target_results.values()))
    vif_df = any_res.vif_df

    missing_pct: Dict[str, float] = {
        col: float(X_df[col].isnull().mean())
        for col in scope_features if col in X_df.columns
    }
    x_for_quality = (
        X_df[scope_features]
        if all(f in X_df.columns for f in scope_features)
        else pd.DataFrame(columns=scope_features)
    )
    fq_scores  = _compute_feature_quality(scope_features, x_for_quality, vif_df, missing_pct)
    stab_scores = _compute_stability_score(X_df, y_df, scope_features, top_k)

    # VIF and avg corr lookups from any representative per-target result
    vif_lookup: Dict[str, float] = {}
    if not vif_df.empty and "VIF" in vif_df.columns:
        vif_lookup = dict(zip(vif_df["Feature"], vif_df["VIF"]))

    # Combined corr_with_target for avg corr display
    corr_frames = [res.corr_with_target for res in target_results.values()]
    combined_corr = pd.concat(corr_frames, axis=1) if corr_frames else pd.DataFrame()
    avg_corr_lookup: Dict[str, float] = {}
    if not combined_corr.empty:
        avg_corr_lookup = combined_corr.abs().mean(axis=1).to_dict()

    rows = []
    for feat in scope_features:
        # --- Coverage (replaces method-based SelectionFreq) ---
        coverage_count = len(feature_target_map.get(feat, []))
        coverage_ratio = coverage_count / n_targets
        coverage_pct   = coverage_ratio * 100.0

        # --- Predictive Strength: nanmean over SELECTED targets only ---
        # Only targets that recommended this feature (HR or Rec) contribute to PS.
        # Targets that rejected the feature are excluded — otherwise their low PS
        # dilutes the true predictive signal for the targets that actually need it.
        # For optional_union (Consider-only) features, feature_target_map still lists
        # the targets that gave them Consider, so we use those.
        relevant_targets = feature_target_map.get(feat, list(target_results.keys()))
        ps_vals: List[float] = [
            _get_ps(target_results[y].consensus_df, feat) for y in relevant_targets
        ]
        ps = float(np.nanmean(ps_vals)) if ps_vals else 0.0

        # --- Per-target AvgRank (informational) ---
        rank_vals: List[float] = []
        for res in target_results.values():
            cdf = res.consensus_df
            if not cdf.empty and "Feature" in cdf.columns and "AvgRank" in cdf.columns:
                match = cdf.loc[cdf["Feature"] == feat, "AvgRank"]
                if len(match) > 0:
                    rank_vals.append(float(match.values[0]))
        avg_rank = float(np.nanmean(rank_vals)) if rank_vals else float(len(scope_features))

        fq   = fq_scores.get(feat, 70.0)
        stab = stab_scores.get(feat, 50.0)

        # --- FinalScore (same damped formula as _aggregate_consensus) ---
        adjusted_freq = coverage_pct * (max(ps, 25.0) / 100.0)
        final_score = round(
            FS_WEIGHT_SELECTION_FREQ       * adjusted_freq
            + FS_WEIGHT_PREDICTIVE_STRENGTH * ps
            + FS_WEIGHT_FEATURE_QUALITY     * fq
            + FS_WEIGHT_STABILITY           * stab,
            1,
        )

        vif      = vif_lookup.get(feat, np.nan)
        avg_corr = avg_corr_lookup.get(feat, np.nan)

        recommendation = _assign_recommendation(
            final_score, ps, fq,
            vif if not np.isnan(vif) else None,
            avg_corr if not np.isnan(avg_corr) else 0.0,
            n_targets=n_targets,
        )

        # ElasticNetSelected = True if EN selected feature in ANY target's run
        en_sel: Optional[bool] = None
        for res in target_results.values():
            en_r = next((r for r in res.method_results if r.method_id == "elasticnet" and r.success), None)
            if en_r is not None:
                if en_r.metadata.get("selected_mask", {}).get(feat, False):
                    en_sel = True
                    break
                else:
                    en_sel = False  # saw elasticnet but not selected; may be overridden by another target

        rows.append({
            "Feature":          feat,
            # Coverage columns (new, explicit)
            "CoverageCount":    coverage_count,
            "CoverageRatio":    round(coverage_ratio, 4),
            "CoveragePercent":  round(coverage_pct, 1),
            # Legacy column names kept so existing UI code works unchanged
            "SelectionCount":   coverage_count,
            "TotalMethods":     n_targets,
            "SelectionFreq":    round(coverage_pct, 1),
            "PredictiveStrength": round(ps, 1),
            "FeatureQuality":   round(fq, 1),
            "StabilityScore":   round(stab, 1),
            "FinalScore":       final_score,
            "ConfidenceScore":  final_score,
            "AvgRank":          round(avg_rank, 1),
            "CorrWithTarget":   round(float(avg_corr), 4) if not np.isnan(avg_corr) else None,
            "VIF":              round(float(vif), 2) if not np.isnan(vif) else None,
            "ElasticNetSelected": en_sel,
            "Recommendation":   recommendation,
        })

    df = pd.DataFrame(rows).sort_values(
        ["FinalScore", "PredictiveStrength"], ascending=[False, False]
    ).reset_index(drop=True)
    df.index = range(1, len(df) + 1)
    df.index.name = "Rank"
    return df


# ---------------------------------------------------------------------------
# Per-target orchestration
# ---------------------------------------------------------------------------

def run_per_target_auto_selection(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    top_k: int = 10,
    enabled_methods: Optional[List[str]] = None,
    corr_threshold: float = 0.85,
    vif_threshold: float = 10.0,
    progress_callback=None,
) -> PerTargetSelectionResult:
    """
    Run the full feature selection pipeline independently for each Y target,
    then build a union-based final feature pool.

    For single-Y datasets this is equivalent to one regular run.
    For multi-Y datasets each target is evaluated in isolation, preventing
    target-specific features from being averaged away.

    Parameters
    ----------
    X_df, y_df            : same contract as run_auto_feature_selection
    top_k                 : top-K passed to each per-target run
    enabled_methods       : method list passed to each per-target run
    corr_threshold        : passed to each per-target run
    vif_threshold         : passed to each per-target run
    progress_callback     : optional callable(step: str) for outer progress

    Returns
    -------
    PerTargetSelectionResult
    """
    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    y_cols = y_df.columns.tolist()
    n_targets = len(y_cols)
    target_results: Dict[str, AutoSelectionResult] = {}

    for i, y_col in enumerate(y_cols):
        _progress(f"[{i + 1}/{n_targets}] Feature selection for target '{y_col}'…")
        result = run_auto_feature_selection(
            X_df=X_df,
            y_df=y_df[[y_col]],
            top_k=top_k,
            enabled_methods=enabled_methods,
            corr_threshold=corr_threshold,
            vif_threshold=vif_threshold,
            progress_callback=None,  # suppress per-step messages inside each sub-run
            _apply_dedup=False,      # dedup runs once on the aggregate, not per-target
        )
        target_results[y_col] = result

    # Build feature → [target names] map from Highly Recommended + Recommended
    feature_target_map: Dict[str, List[str]] = {}
    for y_col, res in target_results.items():
        for feat in res.recommended_features:
            feature_target_map.setdefault(feat, []).append(y_col)

    # Consider features not already in the recommended union
    optional_map: Dict[str, List[str]] = {}
    for y_col, res in target_results.items():
        for feat in res.optional_features:
            if feat not in feature_target_map:
                optional_map.setdefault(feat, []).append(y_col)
                feature_target_map.setdefault(feat, []).append(y_col)

    # Sort by coverage (features recommended by the most targets first)
    union_features = sorted(
        [f for f in feature_target_map if f not in optional_map],
        key=lambda f: len(feature_target_map[f]),
        reverse=True,
    )
    optional_union = sorted(
        optional_map.keys(),
        key=lambda f: len(optional_map[f]),
        reverse=True,
    )

    # Build aggregated consensus from per-target results
    _progress("Aggregating per-target results…")
    scope = list(dict.fromkeys(union_features + optional_union))  # ordered, deduped

    consensus_df = _aggregate_from_per_target_results(
        target_results, scope, feature_target_map, X_df, y_df, top_k
    )
    consensus_df = _dedup_multicollinear(
        consensus_df,
        next(iter(target_results.values())).correlation_matrix,
        corr_threshold,
    )

    # Derive recommendation buckets from the aggregated consensus
    recommended_features = consensus_df.loc[
        consensus_df["Recommendation"].isin(["Highly Recommended", "Recommended"]),
        "Feature",
    ].tolist()
    optional_features = consensus_df.loc[
        consensus_df["Recommendation"] == "Consider", "Feature"
    ].tolist()
    features_to_remove = consensus_df.loc[
        consensus_df["Recommendation"] == "Weak Feature", "Feature"
    ].tolist()

    # Combined corr_with_target: all Y columns together
    corr_frames = [res.corr_with_target for res in target_results.values() if not res.corr_with_target.empty]
    combined_corr = pd.concat(corr_frames, axis=1) if corr_frames else pd.DataFrame()

    # Aggregate method results (averaged across targets) for ranking matrix display
    agg_method_results = _build_aggregate_method_results(target_results, scope)

    # Per-feature reasoning: use method_results from the best-PS target for each feature
    per_feature_reasoning: Dict[str, str] = {}
    any_res = next(iter(target_results.values()))
    for feat in scope:
        best_y = max(
            target_results,
            key=lambda y: _get_ps(target_results[y].consensus_df, feat),
        )
        best_res = target_results[best_y]
        row = consensus_df.loc[consensus_df["Feature"] == feat].squeeze()
        per_feature_reasoning[feat] = _generate_reasoning(
            feat, row,
            best_res.method_results,
            combined_corr,
            best_res.vif_df,
        )
        mc = row.get("MulticollinearWith") if hasattr(row, "get") else None
        if mc and pd.notna(mc):
            per_feature_reasoning[feat] += (
                f" Downgraded from Recommended: highly correlated with {mc}"
                " — include only one of this pair."
            )

    _progress("Per-target selection complete.")
    return PerTargetSelectionResult(
        target_results=target_results,
        union_features=union_features,
        optional_union=optional_union,
        feature_target_map=feature_target_map,
        consensus_df=consensus_df,
        recommended_features=recommended_features,
        optional_features=optional_features,
        features_to_remove=features_to_remove,
        per_feature_reasoning=per_feature_reasoning,
        method_results=agg_method_results,
        corr_with_target=combined_corr,
        vif_df=any_res.vif_df,
        dataset_info={**any_res.dataset_info, "n_targets": n_targets, "target_names": y_cols},
        correlation_matrix=any_res.correlation_matrix,
    )
