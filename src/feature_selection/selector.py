"""
src/feature_selection/selector.py
===================================
Automated feature selection for soft sensor modelling.

Three complementary methods are provided, each with a different inductive
bias, so they can be used in combination or individually:

1. Mutual Information (MI)
   Captures *any* statistical dependency (linear or non-linear) between
   each X feature and the target Y variables.  Based on
   ``sklearn.feature_selection.mutual_info_regression``.

2. Random Forest Importance (RF)
   Trains a multi-output Random Forest on all X features and uses the
   mean decrease in impurity (MDI) averaged across all target columns as
   the importance score.  Robust to feature scale and highly non-linear
   relationships.  Analogous to tree-SHAP values in ranking behaviour.

3. Correlation Filtering
   Ranks features by their average absolute Pearson correlation with the
   Y targets, then applies a collinearity filter: if two top-ranked
   features are too correlated with *each other* (|r| > threshold), the
   lower-ranked one is dropped.  Ensures the final set is diverse.

All three methods handle multi-output Y by averaging scores across targets.

Public API
----------
run_feature_selection(X_df, y_df, method, k, corr_threshold=0.85)
    → list[dict]     # ranked selection results with reasoning

Each result dict contains:
    Rank      : int
    Feature   : str
    Score     : float   (method-specific scale)
    Score Label: str    (human-readable score name)
    Strength  : str     (Weak / Moderate / Strong)
    Reason    : str     (detailed explanation for the selection)
"""
from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

# Supported method keys
SelectionMethod = Literal[
    "Mutual Information",
    "Random Forest Importance",
    "Correlation Filtering",
]


# ---------------------------------------------------------------------------
# Internal: strength labelling helpers
# ---------------------------------------------------------------------------

def _mi_strength(score: float) -> str:
    if score >= 0.40:
        return "Strong"
    elif score >= 0.15:
        return "Moderate"
    return "Weak"


def _rf_strength(score: float) -> str:
    pct = score * 100
    if pct >= 10.0:
        return "Strong"
    elif pct >= 3.0:
        return "Moderate"
    return "Weak"


def _corr_strength(score: float) -> str:
    if score >= 0.60:
        return "Strong"
    elif score >= 0.35:
        return "Moderate"
    return "Weak"


# ---------------------------------------------------------------------------
# Internal: safe imputation before scoring
# ---------------------------------------------------------------------------

def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with column mean; fall back to 0 if the column is all-NaN."""
    return df.apply(lambda s: s.fillna(s.mean()) if s.notna().any() else s.fillna(0))


def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns with zero variance (constants).

    Constant features carry no predictive information and cause
    division-by-zero warnings in correlation computations.
    """
    return df.loc[:, df.std() > 0]


# ---------------------------------------------------------------------------
# Method 1 — Mutual Information
# ---------------------------------------------------------------------------

def _select_mutual_information(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    k: int,
) -> List[dict]:
    """
    Rank X features by average Mutual Information with all Y columns.

    MI is estimated using a k-nearest-neighbours approach
    (``mutual_info_regression``).  It captures *both* linear and
    non-linear dependencies, making it ideal for process data with
    complex sensor interactions.
    """
    X_df_clean = _drop_constant_columns(_safe_fill(X_df))
    X = X_df_clean.values
    y_filled = _safe_fill(y_df)

    # Compute MI for each Y column, then average
    mi_matrix = np.array(
        [
            mutual_info_regression(X, y_filled[col].values, random_state=42)
            for col in y_df.columns
        ]
    )  # shape: (n_y, n_x)
    avg_mi = mi_matrix.mean(axis=0)

    # Build ranked list (only over non-constant columns)
    ranked = sorted(
        zip(X_df_clean.columns, avg_mi), key=lambda t: t[1], reverse=True
    )

    results = []
    for rank, (feat, score) in enumerate(ranked[:k], start=1):
        strength = _mi_strength(score)

        if len(y_df.columns) == 1:
            target_str = f"target '{y_df.columns[0]}'"
        else:
            target_str = f"all {len(y_df.columns)} targets (averaged)"

        results.append(
            {
                "Rank":        rank,
                "Feature":     feat,
                "Score":       round(float(score), 4),
                "Score Label": "Avg MI Score",
                "Strength":    strength,
                "Reason": (
                    f"MI = {score:.4f} with {target_str}. "
                    f"Mutual Information captures both linear and non-linear "
                    f"statistical dependencies between sensor readings and KPIs. "
                    f"MI > 0.40 → Strong predictor; 0.15–0.40 → Moderate; "
                    f"< 0.15 → Weak. This feature is ranked **{strength}** — "
                    + (
                        "it explains substantial shared information with the process KPIs "
                        "and is highly recommended for the soft sensor."
                        if strength == "Strong"
                        else "it carries meaningful information about the KPI behaviour."
                        if strength == "Moderate"
                        else "include only if the top features alone underfit the model."
                    )
                ),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Method 2 — Random Forest Importance
# ---------------------------------------------------------------------------

def _select_rf_importance(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    k: int,
) -> List[dict]:
    """
    Rank X features by mean decrease in impurity (MDI) from a Random Forest.

    For multi-output Y, a separate RF is trained per target and the
    importances are averaged.  This mirrors tree-SHAP behaviour in ranking
    order.  RF importance is robust to non-linearity, feature scale, and
    correlated features.
    """
    X_df_clean = _drop_constant_columns(_safe_fill(X_df))
    X = X_df_clean.values
    y_filled = _safe_fill(y_df)

    importance_matrix = []
    for col in y_df.columns:
        rf = RandomForestRegressor(
            n_estimators=120,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y_filled[col].values)
        importance_matrix.append(rf.feature_importances_)

    avg_importance = np.mean(importance_matrix, axis=0)

    ranked = sorted(
        zip(X_df_clean.columns, avg_importance), key=lambda t: t[1], reverse=True
    )

    total_shown = sum(score for _, score in ranked[:k])

    results = []
    for rank, (feat, score) in enumerate(ranked[:k], start=1):
        strength   = _rf_strength(score)
        pct        = score * 100
        share_pct  = (score / total_shown * 100) if total_shown > 0 else 0

        results.append(
            {
                "Rank":        rank,
                "Feature":     feat,
                "Score":       round(float(score), 5),
                "Score Label": "RF Importance",
                "Strength":    strength,
                "Reason": (
                    f"RF Importance = {score:.5f} ({pct:.2f}% of total variance explained). "
                    f"Among the top {k} selected features, this one accounts for "
                    f"{share_pct:.1f}% of their combined importance. "
                    f"Random Forests measure how often a feature is used for optimal "
                    f"splits weighted by the number of samples affected — making this "
                    f"an interpretable proxy for SHAP tree values. "
                    f"Strength rating: **{strength}**. "
                    + (
                        "Dominant split variable — critical for KPI prediction accuracy."
                        if strength == "Strong"
                        else "Reliable predictor contributing meaningfully to tree splits."
                        if strength == "Moderate"
                        else "Minor contributor; retain if domain knowledge supports it."
                    )
                ),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Method 3 — Correlation Filtering
# ---------------------------------------------------------------------------

def _select_correlation(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    k: int,
    corr_threshold: float = 0.85,
) -> List[dict]:
    """
    Select features by average |Pearson correlation| with Y targets,
    then apply a collinearity filter to ensure diversity.

    Two features that are highly correlated with *each other* carry
    redundant information.  The filter keeps the higher-ranked one and
    skips the lower-ranked duplicate.
    """
    X_filled = _drop_constant_columns(_safe_fill(X_df))
    y_filled = _safe_fill(y_df)

    # Average |correlation| with all Y columns (constant-free X only)
    corr_with_y: Dict[str, float] = {}
    corr_signs:  Dict[str, str]   = {}
    for feat in X_filled.columns:
        cors = []
        signs = []
        for col in y_df.columns:
            r = X_filled[feat].corr(y_filled[col])
            if np.isnan(r):
                r = 0.0
            cors.append(abs(r))
            signs.append("positive" if r >= 0 else "negative")
        corr_with_y[feat]  = float(np.mean(cors))
        # Report the dominant direction
        corr_signs[feat] = "positive" if signs.count("positive") >= len(signs) / 2 else "negative"

    # Sort by score descending
    sorted_feats = sorted(corr_with_y.items(), key=lambda t: t[1], reverse=True)

    # Collinearity filter among X features
    X_corr_matrix = X_filled.corr().abs()
    selected: List[str] = []
    skipped_for: Dict[str, str] = {}

    for feat, _ in sorted_feats:
        if len(selected) >= k:
            break
        redundant = False
        for sel in selected:
            c_val = X_corr_matrix.loc[feat, sel]
            if c_val > corr_threshold:
                redundant   = True
                skipped_for[feat] = sel
                break
        if not redundant:
            selected.append(feat)

    results = []
    for rank, feat in enumerate(selected, start=1):
        score    = corr_with_y[feat]
        strength = _corr_strength(score)
        sign     = corr_signs[feat]

        results.append(
            {
                "Rank":        rank,
                "Feature":     feat,
                "Score":       round(score, 4),
                "Score Label": "Avg |Pearson r|",
                "Strength":    strength,
                "Reason": (
                    f"Average |Pearson r| = {score:.4f} ({sign} relationship with targets). "
                    f"This feature passed the collinearity filter "
                    f"(|r| < {corr_threshold} with all already-selected features), "
                    f"ensuring the selected set is diverse and non-redundant. "
                    f"Strength: **{strength}** — "
                    + (
                        f"|r| ≥ 0.60 confirms a strong linear link to the process KPIs."
                        if strength == "Strong"
                        else f"|r| between 0.35–0.60 indicates a meaningful but moderate linear signal."
                        if strength == "Moderate"
                        else f"|r| < 0.35 — weak linear signal, but retained due to low collinearity."
                    )
                ),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_feature_selection(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    method: SelectionMethod,
    k: int,
    corr_threshold: float = 0.85,
) -> List[dict]:
    """
    Run the chosen feature selection method and return ranked results.

    Parameters
    ----------
    X_df            : candidate input feature DataFrame (all numeric, no NaN required)
    y_df            : target variable DataFrame (all numeric, no NaN required)
    method          : one of the three supported method strings
    k               : number of top features to return
    corr_threshold  : collinearity rejection threshold (Correlation method only)

    Returns
    -------
    List of dicts with keys:
        Rank, Feature, Score, Score Label, Strength, Reason
    """
    if len(X_df.columns) == 0 or len(y_df.columns) == 0:
        return []

    # Clip k to available features
    k = min(k, len(X_df.columns))

    if method == "Mutual Information":
        return _select_mutual_information(X_df, y_df, k)
    elif method == "Random Forest Importance":
        return _select_rf_importance(X_df, y_df, k)
    elif method == "Correlation Filtering":
        return _select_correlation(X_df, y_df, k, corr_threshold)
    else:
        raise ValueError(f"Unknown feature selection method: {method!r}")
