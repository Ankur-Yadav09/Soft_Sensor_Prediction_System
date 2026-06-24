# Feature Selection — Complete Technical Reference

## Table of Contents

1. [Overview](#overview)
2. [Input Preparation](#input-preparation)
3. [Method 1 — Target Correlation](#method-1--target-correlation)
4. [Method 2 — Mutual Information](#method-2--mutual-information)
5. [Method 3 — mRMR](#method-3--mrmr)
6. [Method 4 — Permutation Importance](#method-4--permutation-importance)
7. [Method 5 — Elastic Net](#method-5--elastic-net)
8. [Score Normalization](#score-normalization)
9. [Predictive Strength (PS)](#predictive-strength-ps)
10. [Feature Quality (FQ)](#feature-quality-fq)
11. [Stability Score](#stability-score)
12. [Selection Frequency + Damping](#selection-frequency--damping)
13. [FinalScore Formula](#finalscore-formula)
14. [Recommendation Assignment](#recommendation-assignment)
15. [Multi-Y Aggregation Path](#multi-y-aggregation-path)
16. [Multicollinearity Deduplication](#multicollinearity-deduplication)
17. [Per-Feature Reasoning](#per-feature-reasoning)
18. [Scenario Examples](#scenario-examples)
19. [What Does NOT Change](#what-does-not-change)

---

## Overview

The feature selection engine evaluates every X sensor against every Y KPI target **independently**, then aggregates the results. For multi-Y datasets this prevents strong target-specific features from being washed out by averaging across irrelevant targets.

**Single-Y path:** `run_auto_feature_selection()` directly — 5 methods run once, consensus built once.

**Multi-Y path:** `run_per_target_auto_selection()` — 5 methods run once per Y target, results aggregated using coverage as the primary signal.

---

## Input Preparation

Before any method runs, the engine sanitizes the input data:

### Step 1 — Fill missing values (`_safe_fill`)

```
For each column:
  if any non-NaN values exist → fill NaN with column mean
  if all values are NaN      → fill with 0
```

This is a temporary fill used only inside the feature selection engine — it does NOT modify `st.session_state.df`. The user's preprocessing choices (imputation, outlier treatment) are applied before this step via the Preprocessing page.

### Step 2 — Drop constant columns (`_drop_constant_cols`)

```
Drop any column where std == 0
```

A column with zero standard deviation carries no signal — all values are identical. These are dropped before every method runs. The list of dropped columns is recorded in `dataset_info["constant_features"]` and displayed as a warning banner in the Feature Selection UI.

### Step 3 — VIF computation (`_compute_vif`)

Variance Inflation Factor measures how much a feature's variance is explained by the other features — a proxy for multicollinearity.

For each feature Xi:
```
Regress Xi on all other Xj (j ≠ i) using OLS or Ridge
R² = coefficient of determination of that regression
VIF = 1 / (1 - R²)
```

Ridge regression is used when `n_features ≥ 0.5 × n_rows` (near-collinear system). VIF is capped at 9999. Skipped entirely if `n_features > 80` (performance).

| VIF Range | Level |
|---|---|
| ≤ 5 | Low — no multicollinearity concern |
| 5 – 10 | Moderate — watch this feature |
| > 10 | High — significant multicollinearity |

### Step 4 — X–X and X–Y Correlation Matrices

**X–X:** `X_clean.corr(method="pearson")` — used for multicollinearity detection (Overview tab) and deduplication pass.

**X–Y (corr_with_target):** For each X feature and each Y target: signed Pearson r. Stored per column; used in reasoning generation and as a signal gate in `_assign_recommendation()`.

---

## Method 1 — Target Correlation

**Category:** Supervised | **PS Weight:** 20%

**What it measures:** Linear relationship between each X feature and each Y target, measured by Pearson r.

### Exact Algorithm

```
For each feature Xi (i = 1 … n_features):
  For each target Yj (j = 1 … n_targets):
    r_ij = Pearson correlation coefficient between Xi and Yj
         = corrcoef(Xi, Yj)[0, 1]
    if NaN → replace with 0.0

  raw_score[Xi] = mean( |r_i1|, |r_i2|, …, |r_in| )
  sign[Xi] = "positive" if majority of r_ij ≥ 0 else "negative"
  per_target[Xi][Yj] = |r_ij|   ← stored for Tab5 display
```

The scoring uses **absolute** value (`|r|`) because both strong positive and strong negative correlation indicate predictive power. The sign is stored separately for display.

### Example

| Feature | r with Y1 | r with Y2 | Avg |r| | Raw Score |
|---|---|---|---|---|
| Temperature | +0.92 | -0.20 | 0.56 | 0.56 |
| Pressure | +0.30 | +0.88 | 0.59 | 0.59 |
| Humidity | +0.05 | +0.05 | 0.05 | 0.05 |

After min-max normalization (see §8): scores become 0–1 relative to the best feature in this run.

### Notes

- Measures only **linear** relationship. A feature with a strong non-linear relationship (e.g., U-shaped) may score low here but high on Mutual Information.
- In Tab5 (method detail expander), the signed r values are shown per Y column. The "Avg |r|" column is what feeds into scoring.

---

## Method 2 — Mutual Information

**Category:** Supervised | **PS Weight:** 25%

**What it measures:** Non-linear statistical dependency between X and Y. MI = 0 means independence; higher means more dependency of any kind (linear or non-linear).

### Exact Algorithm

```
For each target Yj (j = 1 … n_targets):
  mi_j = mutual_info_regression(X, Yj, random_state=42)
       = array of MI scores, one per feature
       (uses sklearn's k-nearest neighbour estimator)

For each feature Xi:
  raw_score[Xi] = mean( mi_1[i], mi_2[i], …, mi_n[i] )
  per_target[Xi][Yj] = mi_j[i]   ← stored for Tab5
```

`mutual_info_regression` estimates MI using the Kozachenko–Leonenko entropy estimator with k=3 nearest neighbours by default. It handles continuous variables and does not assume any particular distribution.

### When MI > Correlation

A feature that is strongly correlated with Y in a non-linear way (exponential, quadratic, periodic) will show:
- Low Target Correlation score (r ≈ 0 for perfectly symmetric non-linear relationships)
- High MI score (still detects the dependency)

This is why MI has the second-highest weight (25%) and runs alongside Correlation.

---

## Method 3 — mRMR

**Category:** Advanced Filter | **PS Weight:** 15%

**What it measures:** Maximum Relevance Minimum Redundancy — selects features that are maximally relevant to the target while being minimally redundant with already-selected features.

### Exact Algorithm

```
Step 1 — Compute relevance
  For each target Yj: run mutual_info_regression(X, Yj)
  relevance[Xi] = mean MI across all targets

Step 2 — Build X–X Pearson correlation matrix
  corr_matrix[i, j] = Pearson r between Xi and Xj

Step 3 — Greedy selection loop (selects top_k features)
  selected = []
  remaining = [all features]

  Iteration 1: pick feature with highest relevance
    → adds to selected, removes from remaining

  Iteration 2 … top_k:
    For each candidate Xi in remaining:
      redundancy[Xi] = mean( |corr_matrix[Xi, Xs]| for Xs in selected )
      score[Xi] = relevance[Xi] - redundancy[Xi]
    → pick Xi with highest score
    → adds to selected, removes from remaining

Step 4 — Assign rank-based scores
  raw_score[Xi] = top_k - rank    if Xi in selected  (1st → top_k, 2nd → top_k-1 …)
  raw_score[Xi] = 0               if Xi not selected
```

### Why mRMR Matters

Two features can both have high MI with Y, but if they are highly correlated with each other (r=0.95), including both adds little new information. mRMR penalizes the second one for redundancy, naturally preferring a diverse, non-redundant feature set.

### Example (top_k = 5)

| Selection Round | Feature | Relevance | Redundancy | Score |
|---|---|---|---|---|
| 1 | Temperature | 0.88 | — | 0.88 |
| 2 | Pressure | 0.85 | 0.22 | 0.63 |
| 3 | Flow_Rate | 0.72 | 0.18 | 0.54 |
| 4 | pH | 0.75 | 0.31 | 0.44 |
| 5 | Vibration | 0.60 | 0.19 | 0.41 |

Raw scores: Temperature=5, Pressure=4, Flow_Rate=3, pH=2, Vibration=1, Humidity=0

---

## Method 4 — Permutation Importance

**Category:** Feature Importance | **PS Weight:** 30% (highest weight)

**What it measures:** The drop in model R² when a feature's values are randomly shuffled. A large drop means the model relies on that feature; a small drop means the feature is not critical.

### Exact Algorithm

```
Step 1 — Subsample data (if large)
  Max rows = 5000 (to keep runtime manageable)
  Sample randomly without replacement, seed=42

Step 2 — Average Y targets
  y_avg = mean(Y1, Y2, …, Yn) per row
  (multi-Y: uses averaged target for the base model)

Step 3 — Train base Random Forest
  RandomForestRegressor(
    n_estimators = 50,
    max_features = 0.5,
    random_state = 42,
    n_jobs       = -1
  )
  rf.fit(X_sampled, y_avg)

Step 4 — Compute permutation importance
  sklearn.permutation_importance(
    rf, X_sampled, y_avg,
    n_repeats   = 5,
    random_state= 42
  )
  importance[Xi] = mean drop in R² across 5 shuffles of column i
  Clip negatives to 0 (negative means shuffling helped — treat as zero signal)

Step 5 — Raw score
  raw_score[Xi] = importance[Xi]   (mean R² drop, 5 repeats)
```

### Why Permutation Has the Highest Weight (30%)

- It is **model-agnostic in principle** (uses a trained model but the scoring is post-hoc)
- It captures **actual predictive drop** — not just correlation or coefficient magnitude
- It is robust against feature scale differences (permutation doesn't care about units)
- A feature that scores high here genuinely hurts prediction when removed

### Limitation

Permutation importance can underestimate correlated features — if Xi and Xj are highly correlated, shuffling Xi may not hurt much because Xj compensates. The multicollinearity deduplication pass (Phase 6) addresses this downstream.

---

## Method 5 — Elastic Net

**Category:** Intrinsic | **PS Weight:** 10%

**What it measures:** Sparse linear regression coefficient after L1+L2 regularization. Features with non-zero coefficients are selected; coefficient magnitude indicates relative importance.

### Exact Algorithm

```
Step 1 — Standardize X
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  (Required: coefficients must be on the same scale for L1 to be fair)

Step 2 — Cross-validation folds
  cv = min(3, max(2, n_rows / 50))
  (adaptive: tiny datasets get cv=2, larger get cv=3)

Step 3 — Fit model
  Single Y:  ElasticNetCV(cv=cv, random_state=42, max_iter=2000)
  Multi-Y:   MultiTaskElasticNetCV(cv=cv, random_state=42, max_iter=2000)
  → CV selects the best alpha (regularization strength) automatically

Step 4 — Extract coefficients
  Single Y:  coefs = |model.coef_|                   shape: (n_features,)
  Multi-Y:   coefs = mean(|model.coef_|, axis=0)     shape: (n_features,)
             per_target[Xi][Yj] = |coef_matrix[j, i]|

Step 5 — Binary selection flag
  ElasticNetSelected[Xi] = True  if coefs[i] > 1e-8
                         = False otherwise
  (L1 regularization drives truly irrelevant features exactly to 0)

Step 6 — Raw score
  raw_score[Xi] = coefs[i]
```

### What L1 + L2 Does

- **L1 (Lasso):** drives irrelevant feature coefficients to exactly 0 → automatic selection
- **L2 (Ridge):** keeps correlated features stable → prevents one from dominating arbitrarily
- Elastic Net balances both, making it more stable than pure Lasso when features are correlated

### Why Elastic Net Has the Lowest Weight (10%)

Elastic Net is a linear model — it cannot detect non-linear relationships. It is included for stability and as a regularization cross-check, but its signal is intentionally weighted lower than model-based methods (Permutation, MI).

---

## Score Normalization

After each method computes its raw scores, they are normalized to [0, 1] using min-max scaling.

```
normalized[Xi] = (raw[Xi] - min_raw) / (max_raw - min_raw)

Edge case: if all features have the same raw score → every feature gets 0.5
```

This normalization is applied PER METHOD. A score of 0.8 from Mutual Information and 0.8 from Target Correlation both mean "80% of the way between the worst and best feature in THIS run" — not the same absolute value.

**Why normalize?** The raw scales are incompatible. Target Correlation produces values in [0, 1]; Mutual Information is in nats (unbounded); Permutation Importance is in R² units; Elastic Net is in coefficient magnitude. Normalization puts all methods on a common 0–1 scale before combining them into PS.

---

## Predictive Strength (PS)

PS is the weighted combination of the 5 normalized method scores. It represents "how strongly does this feature predict the target(s) according to our ensemble of methods?"

### Formula

```
PS = (w_corr × norm_corr + w_mi × norm_mi + w_perm × norm_perm
      + w_mrmr × norm_mrmr + w_en × norm_en) × 100
```

### Exact Weights

| Method | Weight | Reason |
|---|---|---|
| Permutation Importance | **30%** | Directly measures predictive drop — most informative signal |
| Mutual Information | **25%** | Captures non-linear dependencies missed by correlation |
| Target Correlation | **20%** | Fast, interpretable linear signal |
| mRMR | **15%** | Relevance-minus-redundancy — penalizes redundant features |
| Elastic Net | **10%** | Regularized linear model — lower weight (linear only) |

### Weight Redistribution on Method Failure

If a method fails (exception, insufficient data) its weight is redistributed proportionally to the remaining active methods:

```
For each active method m:
  norm_weight[m] = weight[m] / sum(weights of active methods)

PS = sum( norm_weight[m] × score[m] ) × 100
```

Example: if Elastic Net fails, the 10% is split proportionally — Permutation gets ~33.3%, MI gets ~27.8%, etc. PS stays on the 0–100 scale.

### PS Result: 0–100

- PS = 100: feature scored top-ranked in every method
- PS = 50: mid-range across methods
- PS = 0: bottom-ranked in every method

---

## Feature Quality (FQ)

FQ measures the **data health** of a feature — independent of its predictive strength. A feature can have high PS but poor FQ (e.g., highly multicollinear or mostly missing).

### Formula

```
FQ = 0.50 × VIF_Score + 0.30 × Missing_Score + 0.20 × Variance_Score
```

### VIF Score (50% of FQ)

| VIF | VIF_Score |
|---|---|
| ≤ 5 | 100 — no multicollinearity |
| 5 – 10 | 80 — moderate, acceptable |
| 10 – 20 | 50 — high, consider removing |
| 20 – 30 | 20 — very high |
| > 30 | 0 — extreme multicollinearity |
| NaN (skipped) | 70 — neutral assumption |

### Missing Value Score (30% of FQ)

| Missing % | Miss_Score |
|---|---|
| ≤ 1% | 100 — essentially complete |
| 1 – 5% | 90 |
| 5 – 10% | 75 |
| 10 – 20% | 50 |
| 20 – 30% | 25 |
| > 30% | 0 — too much missing data |

### Variance Score (20% of FQ)

| std | Var_Score |
|---|---|
| = 0 | 0 — constant, no information |
| < 0.001 | 20 — near-zero variance |
| 0.001 – 0.01 | 50 |
| 0.01 – 0.05 | 80 |
| ≥ 0.05 | 100 — healthy variance |

### FQ Example

Feature: `Pressure` — VIF=8.5, Missing=3%, std=0.12

```
VIF_Score     = 80   (VIF between 5 and 10)
Missing_Score = 90   (3% missing)
Variance_Score= 100  (std ≥ 0.05)

FQ = 0.50 × 80 + 0.30 × 90 + 0.20 × 100
   = 40 + 27 + 20
   = 87.0
```

---

## Stability Score

Stability measures how **consistently** a feature is selected across random subsets of the data. A high-stability feature is reliably important regardless of which rows are in the training set.

### Parameters (from `settings.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `FS_STABILITY_RUNS` | 20 | Number of bootstrap iterations |
| `FS_STABILITY_SAMPLE_FRAC` | 0.80 | 80% of rows sampled per run |
| `FS_STABILITY_MAX_ROWS` | 3000 | Cap on rows used for stability (performance) |

### Exact Algorithm

```
sample_size = int( min(n_rows, 3000) × 0.80 )
sample_size = max(sample_size, 20)

stability_points = {feature: 0.0 for all features}

For run = 1 … 20:

  Step 1 — Bootstrap sample
    idx = random sample WITH replacement, size = sample_size, seed varies per run
    X_boot = X[idx], y_boot = y[idx]

  Step 2 — Run 3 fast methods on bootstrap sample
    Method A: Target Correlation   → selected_A (top-k features)
    Method B: Mutual Information   → selected_B
    Method C: RF Importance        → selected_C
    (RF Importance is NOT a scoring method — used only here for stability)

  Step 3 — Require ≥ 2 votes (60% of 3 methods) to count a feature
    required_votes = ceil(3 × 0.60) = 2

  Step 4 — Compute rank-weighted score for this run
    For each method's ranked list:
      For each feature Xi at rank r (0-indexed):
        rank_weight = (top_k - r) / top_k
        add rank_weight to feature_points[Xi]
        increment votes[Xi]

  Step 5 — Credit features that met the vote threshold
    For each feature Xi:
      if votes[Xi] ≥ 2:
        normalized_score = feature_points[Xi] / 3  (divide by n_methods)
        stability_points[Xi] += normalized_score

Final StabilityScore:
  For each feature Xi:
    StabilityScore[Xi] = (stability_points[Xi] / 20) × 100
    Clipped to [0, 100]
```

### What Rank Weighting Does

A feature ranked 1st in a method contributes more than a feature ranked 10th, even if both are "selected":

```
top_k = 10, feature at rank 0 (1st): weight = (10-0)/10 = 1.0
top_k = 10, feature at rank 9 (10th): weight = (10-9)/10 = 0.1
```

This rewards consistent top ranking, not just borderline inclusion.

### Stability Fallback

If the bootstrap process fails (exception in all 3 methods), every feature gets `StabilityScore = 50.0` (neutral — no information either way).

---

## Selection Frequency + Damping

### Selection Frequency

```
SelectionCount[Xi] = number of the 5 core methods that included Xi in their top-k
SelectionFreq[Xi]  = SelectionCount / 5 × 100   (expressed as 0–100%)
```

A feature selected by all 5 methods: SelectionFreq = 100%
A feature selected by 3 of 5 methods: SelectionFreq = 60%

### Damping Factor

A weak feature can appear in many methods' top-k lists simply because it ranked, say, 8th out of 10 features in a dataset with few alternatives. To prevent high SelectionFreq from inflating FinalScore for a weak feature:

```
adjusted_SelectionFreq = SelectionFreq × max(PS, 25.0) / 100
```

| PS | Damping effect on SelectionFreq |
|---|---|
| PS = 80 | adjusted = SelectionFreq × 0.80 (mild dampening) |
| PS = 50 | adjusted = SelectionFreq × 0.50 (moderate) |
| PS = 25 | adjusted = SelectionFreq × 0.25 (strong dampening, minimum floor) |
| PS = 10 | adjusted = SelectionFreq × 0.25 (same as PS=25 — floor at 25%) |

The floor of 25 prevents a feature from being scored as if it has zero frequency even when PS is very low.

---

## FinalScore Formula

```
FinalScore = 0.30 × adjusted_SelectionFreq
           + 0.50 × PredictiveStrength
           + 0.20 × StabilityScore
```

**Feature Quality (FQ) is not part of FinalScore.** Missing values and near-zero variance are handled upstream in the Preprocessing step, so those sub-components would be flat constants for every feature. VIF still enforces quality as a hard gate inside `_assign_recommendation()` — a high-VIF feature cannot reach Highly Recommended regardless of its FinalScore.

### Component Weights

| Component | Weight | Range | Drives |
|---|---|---|---|
| adjusted_SelectionFreq | **30%** | 0–100 | Breadth — how many methods agree |
| PredictiveStrength | **50%** | 0–100 | Core signal — weighted method ensemble |
| StabilityScore | **20%** | 0–100 | Robustness — consistency across bootstrap runs |

PredictiveStrength carries the highest weight (50%) because it directly encodes the multi-method predictive ensemble. Stability rewards features that are consistently selected across data subsets, not just on the full dataset.

### FinalScore Range

| FinalScore | Typical outcome |
|---|---|
| ≥ 70 | Highly Recommended (if PS ≥ 65 and VIF < 10) |
| 50 – 69 | Recommended (if PS ≥ 45) |
| 35 – 49 | Consider |
| < 35 | Weak Feature |

---

## Recommendation Assignment

Recommendation is assigned by `_assign_recommendation()` using a **multi-gate** logic — not just FinalScore. Every gate must pass for the higher tiers.

### Exact Thresholds (from `settings.py`)

| Threshold constant | Value | Note |
|---|---|---|
| `FS_HIGHLY_REC_MIN_FINAL` | **70.0** | Was 80 before FQ removal |
| `FS_HIGHLY_REC_MIN_PRED_STRENGTH` | **65.0** | Was 70 before FQ removal |
| `FS_HIGHLY_REC_MAX_VIF` | 10.0 | Still enforced as hard gate |
| `FS_RECOMMENDED_MIN_FINAL` | **50.0** | Was 60 before FQ removal |
| `FS_RECOMMENDED_MIN_PRED_STRENGTH` | **45.0** | Was 50 before FQ removal |
| `FS_CONSIDER_MIN_FINAL` | **35.0** | Was 40 before FQ removal |
| `FS_WEAK_MAX_PRED_STRENGTH` | 30.0 | |
| `FS_HIGHLY_REC_MIN_QUALITY` | 60.0 | **Unused** — FQ removed from logic |
| `FS_RECOMMENDED_MIN_QUALITY` | 40.0 | **Unused** — FQ removed from logic |
| `FS_WEAK_MAX_QUALITY` | 20.0 | **Unused** — FQ removed from logic |

Thresholds were lowered by ~10–12 points to compensate for the removal of Feature Quality (FQ) from FinalScore. FQ was contributing an average of ~15 points (0.20 × 75 ≈ 15) to scores. Its quality sub-components (missing values, near-zero variance) are now handled upstream in Preprocessing, making the FQ contribution redundant.

### Decision Logic (evaluated in order)

```
1. EARLY WEAK FEATURE GATES (override everything):
   if |avg_corr_with_target| < 0.05 AND PS < 50:
       → "Weak Feature"   (no linear signal and low predictive power)
   if PS < 30:
       → "Weak Feature"   (minimum predictive strength floor)

2. HIGHLY RECOMMENDED (all 3 conditions must hold):
   FinalScore  ≥ 70
   PS          ≥ 65 × scale   (scale reduces for multi-Y, see below)
   VIF         < 10
   → "Highly Recommended"

3. RECOMMENDED (both conditions must hold):
   FinalScore  ≥ 50
   PS          ≥ 45 × scale
   → "Recommended"

4. CONSIDER:
   FinalScore  ≥ 35
   → "Consider"

5. DEFAULT:
   → "Weak Feature"
```

**Note:** Feature Quality (FQ) gates (`FQ ≥ 60`, `FQ ≥ 40`, `FQ < 20`) have been removed. FQ sub-components (missing values, near-zero variance) are addressed in Preprocessing. Only the VIF hard gate remains for Highly Recommended — multicollinear features cannot reach the top tier regardless of FinalScore.

### Multi-Y PS Scaling

When there are multiple Y targets, the PS thresholds for HR and Recommended are softened. This compensates for the fact that PS is averaged across targets, which naturally compresses scores when a feature is a specialist for one target out of many.

```
scale = 1.0 - 0.08 × min(n_targets - 1, 4)
```

| n_targets | scale | Effective HR PS threshold | Effective Rec PS threshold |
|---|---|---|---|
| 1 | 1.00 | 65.0 | 45.0 |
| 2 | 0.92 | 59.8 | 41.4 |
| 3 | 0.84 | 54.6 | 37.8 |
| 4 | 0.76 | 49.4 | 34.2 |
| 5+ | 0.68 | 44.2 | 30.6 (max softening) |

The Weak Feature PS floor (30) is **not scaled** — a truly weak feature stays Weak Feature regardless of how many Y targets exist.

### Why Multiple Gates Instead of Just FinalScore?

A single FinalScore threshold could be gamed by edge cases:
- High stability + low PS → FinalScore ≥ 50, but the feature doesn't predict anything useful
- The PS gate ensures there must be a genuine predictive signal (≥ 45 for Recommended, ≥ 65 for HR)
- The VIF gate on HR ensures no multicollinear feature gets the top badge (regardless of FinalScore)

FQ quality gates were removed because data quality issues (missing values, near-zero variance) are now addressed in Preprocessing before feature selection runs — making FQ gates redundant.

---

## Multi-Y Aggregation Path

For datasets with more than one Y target, `run_per_target_auto_selection()` is the primary engine. It runs the full 5-method pipeline once per Y target independently, then aggregates.

### Step-by-Step Flow

```
For each Y target (Y1, Y2, … Yn):
  Run run_auto_feature_selection(X, [Yj], _apply_dedup=False)
  → produces AutoSelectionResult_j with:
    - consensus_df_j    (PS, FQ, Stability, Recommendation for this target alone)
    - recommended_j     (HR + Recommended features for Yj)
    - optional_j        (Consider features for Yj)
    - method_results_j  (5 MethodResult objects)
    - corr_with_target_j (X vs Yj correlations)
    - vif_df_j          (X–X VIF, same across all targets)
    - correlation_matrix_j (X–X Pearson, same across all targets)

Build union_features:
  All features that were Recommended or HR in at least one target's result
  Sorted by coverage count descending

Build optional_union:
  All features that were Consider in at least one target (not already in union)

Build feature_target_map:
  feature_target_map[Xi] = [list of Yj where Xi was recommended/HR]

Aggregate per feature (for all features in union + optional):
  CoverageCount[Xi]    = len(feature_target_map[Xi])
  CoverageRatio[Xi]    = CoverageCount[Xi] / n_targets
  SelectionFreq[Xi]    = CoverageRatio × 100   ← replaces method-count frequency
  PS[Xi]               = mean( PS_j[Xi] for Yj in feature_target_map[Xi] )
                         ← only over SELECTED targets, not all targets
  FQ[Xi]               = _compute_feature_quality() on full X (same as single-Y)
  Stability[Xi]        = _compute_stability_score() on full X, full Y (same as single-Y)
  FinalScore[Xi]       = 0.30×adj_freq + 0.50×PS + 0.20×Stability
  Recommendation[Xi]   = _assign_recommendation(FinalScore, PS, VIF, corr, n_targets)

Run multicollinearity deduplication (ONCE on aggregated consensus)
```

### Why PS Uses Only Selected Targets

If Temperature predicts Y1 well (PS=91) but not Y2 or Y3 (PS=17, 20), the old approach of averaging all three gives PS=42.7, which would classify Temperature as borderline or weak. By averaging only over the targets that recommended Temperature (just Y1), PS = 91 — reflecting its true strength for the target it matters for.

### The Aggregated PerTargetSelectionResult

All downstream UI tabs read from this aggregated result as if it were an `AutoSelectionResult` — Tab2, Tab3, Tab4, Tab5 are unchanged. Tab6 additionally shows the per-target breakdown table.

---

## Multicollinearity Deduplication

After all scores are assigned, a post-scoring pass removes redundant feature pairs from the Recommended list.

### When It Runs

| Scenario | Where |
|---|---|
| Single-Y | At end of `run_auto_feature_selection()` |
| Multi-Y | At end of `run_per_target_auto_selection()` on aggregated consensus |
| Per-target sub-calls | **Disabled** (`_apply_dedup=False`) — prevents corrupting union calculation |

### Algorithm

```
Input: consensus_df, X–X correlation_matrix, corr_threshold (UI setting, default 0.85)

1. Find all Recommended/Highly Recommended features
2. Scan correlation_matrix for pairs where |r| > corr_threshold
3. Sort pairs by |r| descending (most redundant first)
4. Greedy resolution:
   For each pair (A, B) in sorted order:
     if either is already downgraded → skip
     if both still Recommended/HR:
       winner = feature with higher FinalScore
       loser  = the other
       loser.Recommendation = "Consider"
       loser.MulticollinearWith = "winner_name (|r|=X.XXXX)"
```

Downgraded features remain in the dataset as "Consider" — the user can manually include them if domain knowledge supports it. They are excluded from the Quick-Apply "Use Recommended" set.

---

## Per-Feature Reasoning

After all scores and recommendations are computed, `_generate_reasoning()` builds a human-readable explanation for each feature that appears in the Feature Selection UI (Tab2 "Why?" column).

### Content of the Reasoning Block

```
1. Score card table
   Feature name, Recommendation label
   FinalScore, PS, FQ, Stability, SelectionFreq

2. Reason tags (bullet points)
   SelectionFreq interpretation:
     ≥75% → "Selected by X/5 methods (high consensus)"
     ≥50% → "moderate consensus"
     <50%  → "low consensus"
   Average rank interpretation (informational):
     ≤3  → "consistently top ranked"
     ≤7  → "moderately ranked"
     >7   → "lower ranked"
   PS interpretation:
     ≥70 → "High predictive power"
     ≥50 → "Moderate predictive power"
     <50 → "Low predictive power"
   Permutation importance:
     norm>0.6 → "Strong permutation importance"
     norm>0.3 → "Moderate permutation importance"
   mRMR: if norm>0.6 → "Low redundancy detected by mRMR"
   VIF:
     >10 → "High multicollinearity (VIF=X)"
     >5  → "Moderate multicollinearity"
     ≤5  → "Low multicollinearity"
   Correlation with target:
     |r|<0.1 → "Low correlation with target"
     |r|≥0.5 → "Strong correlation (|r|=X)"
   FQ: ≥80 → "Excellent feature quality" / <40 → "Poor quality"
   Stability: ≥75 → "Stable" / <40 → "Unstable"

3. Correlation with each target (signed r per Y column)

4. Elastic Net: selected or eliminated

5. Which of the 5 methods selected / rejected this feature

6. Business interpretation paragraph
   Highly Recommended: include as primary input
   Recommended: supporting input
   Consider: include only if domain knowledge supports
   Weak Feature: not recommended
```

If multicollinearity deduplication downgraded a feature, an additional note is appended:
> "Downgraded from Recommended: highly correlated with {winner} (|r|=X.XXXX) — include only one of this pair."

---

## Scenario Examples

### Example Dataset

| Sensors (X) | KPI Targets (Y) |
|---|---|
| Temperature | Product_Quality |
| pH | Energy_Consumption |
| Flow_Rate | Yield_Rate |
| Pressure | |
| Vibration | |
| Humidity | |

---

### Scenario 1 — Specialist Feature (Strong for 1 Target)

> **Temperature**: PS=91 for Product_Quality, rejected by Energy and Yield

| Step | Old Approach | New Approach |
|---|---|---|
| PS used | mean(91, 20, 17) = **42.7** | mean(91) = **91.0** |
| Result | Consider ❌ | Recommended ✅ |

**Why it works:** `feature_target_map["Temperature"] = ["Product_Quality"]` — only Y1's PS feeds the aggregation.

---

### Scenario 2 — Generalist Feature (Selected by All Targets)

> **Flow_Rate**: PS=67, 81, 74 across all 3 targets

| Step | Old Approach | New Approach |
|---|---|---|
| PS used | mean(67, 81, 74) = **74.0** | mean(67, 81, 74) = **74.0** |
| Result | Recommended ✅ | Recommended ✅ |

When all targets select a feature, both approaches give identical PS. The generalist also gets SelectionFreq = 100% (3/3 coverage).

---

### Scenario 3 — Partial Specialist

> **Vibration**: PS=10 (Y1 rejected), PS=73 (Y2 selected), PS=53 (Y3 selected)

| Step | Old Approach | New Approach |
|---|---|---|
| PS used | mean(10, 73, 53) = **45.3** | mean(73, 53) = **63.0** |
| Result | Consider ⚠️ | Consider (higher score) ✅ |

Y1's irrelevant PS=10 is excluded from aggregation.

---

### Scenario 4 — Feature Rejected by All Targets

> **Humidity**: PS=5, 5, 4 — rejected by all 3 targets

| Step | Result |
|---|---|
| In union? | ❌ No — excluded at Phase 2 |
| PS computed? | Never reaches aggregation |
| Recommendation | Not shown |

---

### Scenario 5 — Single Y Target

> Dataset has only 1 Y target

| Step | What Happens |
|---|---|
| Engine | `run_auto_feature_selection` directly |
| SelectionFreq | Fraction of 5 methods that selected feature (0–100%) |
| PS | Weighted combination of 5 method normalized scores × 100 |
| Scale factor | 1.0 (no multi-Y softening) |
| Behaviour | Identical to original — completely unaffected |

---

### Scenario 6 — High Coverage, Low PS (Noisy Generalist)

> Feature selected by all 3 targets but PS=35, 30, 32 everywhere

```
SelectionFreq    = 100%
PS               = mean(35, 30, 32) = 32.3
adjusted_freq    = 100 × max(32.3, 25) / 100 = 32.3
Stability        = 75 (assumed)
FinalScore       = 0.30×32.3 + 0.50×32.3 + 0.20×75
                 = 9.7 + 16.2 + 15.0 = 40.9
Recommendation   = Consider
```

The damping factor couples SelectionFreq to PS — a weak-PS feature cannot score high on FinalScore through breadth alone.

---

### Scenario 7 — Recommendation Gate Example

> Feature: FinalScore=68, PS=70, VIF=7.5, n_targets=1

```
Early Weak gates: PS=70 ≥ 30 → pass

Highly Recommended check (n_targets=1, scale=1.0):
  FinalScore ≥ 70?   68 < 70 → FAIL
  → Not Highly Recommended

Recommended check:
  FinalScore ≥ 50?   68 ≥ 50 ✅
  PS ≥ 45?           70 ≥ 45 ✅
  → Recommended ✅
```

This feature just missed Highly Recommended because FinalScore=68 < 70. A slightly higher Stability or SelectionFreq would push it across the HR threshold.

---

### Scenario 8 — Multi-Y Threshold Softening

> Feature: FinalScore=55, PS=40, VIF=7.5, n_targets=3

```
scale = 1.0 - 0.08 × (3-1) = 0.84
effective HR PS threshold  = 65 × 0.84 = 54.6
effective Rec PS threshold = 45 × 0.84 = 37.8

Highly Recommended check:
  FinalScore ≥ 70?   55 < 70 → FAIL
  → Not Highly Recommended

Recommended check (multi-Y, scale=0.84):
  FinalScore ≥ 50?   55 ≥ 50 ✅
  PS ≥ 37.8?         40 ≥ 37.8 ✅
  → Recommended ✅

Same feature with single-Y (scale=1.0):
  Recommended check:
  FinalScore ≥ 50?   55 ≥ 50 ✅
  PS ≥ 45?           40 < 45 → FAIL
  Consider check:
  FinalScore ≥ 35?   55 ≥ 35 ✅
  → Consider
```

The softening changed the outcome: **Consider → Recommended**. This reflects a feature that is a moderate contributor to one Y target out of three — a specialist signal that the scaled threshold correctly promotes.

---

### Scenario 9 — Multicollinear Pair

> **DMCTF_Feed** (FinalScore=68) and **CHG_GAS_FLOW_TO_DRYER** (FinalScore=62), |r|=0.9746

Both independently exceeded the Recommended threshold. Deduplication pass:

| Step | Action |
|---|---|
| Pair detected | DMCTF_Feed ↔ CHG_GAS_FLOW_TO_DRYER, \|r\|=0.9746 > 0.85 |
| Winner | DMCTF_Feed — FinalScore 68 > 62 |
| Loser | CHG_GAS_FLOW_TO_DRYER → Consider |
| MulticollinearWith | "DMCTF_Feed (\|r\|=0.9746)" |

Final Tab2 result:

| Feature | Recommendation | MulticollinearWith |
|---|---|---|
| DMCTF_Feed | 🔵 Recommended | — |
| CHG_GAS_FLOW_TO_DRYER | 🟡 Consider | DMCTF_Feed (\|r\|=0.9746) |

---

## What Does NOT Change

| Component | Status |
|---|---|
| `_compute_predictive_strength()` | Unchanged — called per target independently |
| `_compute_feature_quality()` | Unchanged — VIF + missing + variance on full X |
| `_compute_stability_score()` | Unchanged — bootstrap on full X/Y |
| `_assign_recommendation()` | Unchanged — same thresholds, called before dedup |
| `_generate_reasoning()` | Unchanged — uses best-target method results; dedup appends a note |
| `_dedup_multicollinear()` | **New** — post-scoring pass only; does NOT change scores |
| Single-Y path | Unchanged — uses `run_auto_feature_selection` directly |
| All UI tabs | Unchanged — Tab2–Tab5 work via pseudo AutoSelectionResult |
| Tab2 | Enhanced — `MulticollinearWith` column for downgraded features |
| Tab6 | Enhanced — per-target PS breakdown table |
| All config thresholds | In `config/settings.py` only — never hardcoded |
