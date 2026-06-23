# Feature Selection — How It Works (With Examples)

## Overview

The feature selection engine evaluates every X sensor/input against every Y KPI target
**independently**, then aggregates the results using coverage and predictive strength.
This prevents strong target-specific features from being averaged away when multiple
Y targets are present.

---

## Example Dataset

| Sensors (X) | KPI Targets (Y) |
|---|---|
| Temperature | Product_Quality |
| pH | Energy_Consumption |
| Flow_Rate | Yield_Rate |
| Pressure | |
| Vibration | |
| Humidity | |

**6 input sensors → 3 KPI targets**

---

## Phase 1 — Per-Target Independent Scoring

For each Y target, all 5 scoring methods run independently on the full X dataset.
No target information is mixed or averaged at this stage.

### 5 Scoring Methods

| Method | What It Measures |
|---|---|
| **Target Correlation** | Linear relationship between X and Y |
| **Mutual Information** | Non-linear dependency between X and Y |
| **mRMR** | Relevance to Y minus redundancy with already-selected features |
| **Permutation Importance** | Drop in prediction accuracy when X is shuffled |
| **Elastic Net** | Regularised regression coefficient — sparse and stable |

Each method produces a score per feature (0–1 normalised). These combine into
**Predictive Strength (PS)** using fixed weights:

| Method | Weight |
|---|---|
| Permutation Importance | 30% |
| Mutual Information | 25% |
| Target Correlation | 20% |
| mRMR | 15% |
| Elastic Net | 10% |

---

### Y1 = Product_Quality

| Feature | Correlation | Mut. Info | mRMR | Permutation | ElasticNet | **PS** | **Verdict** |
|---|---|---|---|---|---|---|---|
| Temperature | 0.92 | 0.88 | 0.85 | 0.91 | ✅ | **91** | ✅ Highly Recommended |
| pH | 0.85 | 0.80 | 0.78 | 0.83 | ✅ | **82** | ✅ Recommended |
| Flow_Rate | 0.70 | 0.65 | 0.60 | 0.72 | ❌ | **67** | ✅ Recommended |
| Pressure | 0.30 | 0.28 | 0.25 | 0.31 | ❌ | **29** | ❌ Weak Feature |
| Vibration | 0.10 | 0.12 | 0.08 | 0.09 | ❌ | **10** | ❌ Weak Feature |
| Humidity | 0.05 | 0.04 | 0.06 | 0.05 | ❌ | **5** | ❌ Weak Feature |

### Y2 = Energy_Consumption

| Feature | Correlation | Mut. Info | mRMR | Permutation | ElasticNet | **PS** | **Verdict** |
|---|---|---|---|---|---|---|---|
| Pressure | 0.88 | 0.90 | 0.87 | 0.89 | ✅ | **88** | ✅ Highly Recommended |
| Flow_Rate | 0.82 | 0.79 | 0.80 | 0.84 | ✅ | **81** | ✅ Recommended |
| Vibration | 0.75 | 0.72 | 0.70 | 0.73 | ❌ | **73** | ✅ Recommended |
| Temperature | 0.20 | 0.18 | 0.22 | 0.19 | ❌ | **20** | ❌ Weak Feature |
| pH | 0.10 | 0.09 | 0.11 | 0.08 | ❌ | **10** | ❌ Weak Feature |
| Humidity | 0.05 | 0.06 | 0.04 | 0.05 | ❌ | **5** | ❌ Weak Feature |

### Y3 = Yield_Rate

| Feature | Correlation | Mut. Info | mRMR | Permutation | ElasticNet | **PS** | **Verdict** |
|---|---|---|---|---|---|---|---|
| Flow_Rate | 0.78 | 0.74 | 0.72 | 0.76 | ✅ | **74** | ✅ Highly Recommended |
| pH | 0.72 | 0.70 | 0.68 | 0.74 | ✅ | **71** | ✅ Recommended |
| Vibration | 0.55 | 0.52 | 0.50 | 0.54 | ❌ | **53** | ✅ Recommended |
| Temperature | 0.18 | 0.15 | 0.20 | 0.17 | ❌ | **17** | ❌ Weak Feature |
| Pressure | 0.22 | 0.20 | 0.18 | 0.21 | ❌ | **20** | ❌ Weak Feature |
| Humidity | 0.04 | 0.05 | 0.03 | 0.04 | ❌ | **4** | ❌ Weak Feature |

---

## Phase 2 — Union of Selected Features

All features recommended (HR or Rec) by **at least one target** enter the union.
Features rejected by every target are excluded.

| Feature | Y1 Verdict | Y2 Verdict | Y3 Verdict | **In Union?** |
|---|---|---|---|---|
| Temperature | ✅ HR | ❌ Weak | ❌ Weak | ✅ Yes |
| pH | ✅ Rec | ❌ Weak | ✅ Rec | ✅ Yes |
| Flow_Rate | ✅ Rec | ✅ Rec | ✅ HR | ✅ Yes |
| Pressure | ❌ Weak | ✅ HR | ❌ Weak | ✅ Yes |
| Vibration | ❌ Weak | ✅ Rec | ✅ Rec | ✅ Yes |
| Humidity | ❌ Weak | ❌ Weak | ❌ Weak | ❌ No |

> **Humidity** is excluded — no target found it useful.
> **Pressure** and **Temperature** are included even though they were rejected by 2/3 targets —
> their specialist signal for one target is preserved.

---

## Phase 3 — feature_target_map (Auto-Built)

The system automatically records which targets selected each feature:

| Feature | Selected By | Coverage Count | Coverage % |
|---|---|---|---|
| Temperature | Product_Quality | 1 / 3 | 33% |
| pH | Product_Quality, Yield_Rate | 2 / 3 | 67% |
| Flow_Rate | Product_Quality, Energy, Yield_Rate | 3 / 3 | 100% |
| Pressure | Energy_Consumption | 1 / 3 | 33% |
| Vibration | Energy_Consumption, Yield_Rate | 2 / 3 | 67% |

This map is built automatically by scanning each target's `recommended_features` list
in a single loop — no manual input required.

---

## Phase 4 — Predictive Strength Aggregation (Key Fix)

PS is averaged **only over the targets that selected the feature**, not all targets.

### Why This Matters

| Feature | PS(Y1) | PS(Y2) | PS(Y3) | Mean ALL (wrong) | Selected Targets | **Mean SELECTED (correct)** |
|---|---|---|---|---|---|---|
| Temperature | **91** | 20 | 17 | 42.7 ❌ | Y1 only | **91.0** ✅ |
| pH | **82** | 10 | **71** | 54.3 ❌ | Y1, Y3 | **76.5** ✅ |
| Flow_Rate | **67** | **81** | **74** | 74.0 ✅ | Y1, Y2, Y3 | **74.0** ✅ |
| Pressure | 29 | **88** | 20 | 45.7 ❌ | Y2 only | **88.0** ✅ |
| Vibration | 10 | **73** | **53** | 45.3 ❌ | Y2, Y3 | **63.0** ✅ |

> Only **Flow_Rate** (selected by all targets) gives the same result either way.
> Every specialist feature is diluted by the "mean all" approach.

---

## Phase 5 — Final Score Computation (unchanged)

Four components combine into the Final Score:

```
FinalScore = 0.25 × SelectionFreq  +  0.40 × PredictiveStrength
           + 0.20 × FeatureQuality  +  0.15 × StabilityScore
```

| Component | Source | What It Measures |
|---|---|---|
| **SelectionFreq** | Coverage % across targets | Breadth — how many targets need this feature |
| **PredictiveStrength** | Mean PS over selected targets | Strength — how well it predicts for relevant targets |
| **FeatureQuality** | VIF + Missing% + Variance | Data health — multicollinearity, completeness, variance |
| **StabilityScore** | Bootstrap resampling | Consistency across data subsets |

### Final Score Table (FQ=85, Stab=75 assumed constant)

| Feature | Coverage% | PS | FQ | Stab | **Final Score** | **Recommendation** |
|---|---|---|---|---|---|---|
| Temperature | 33% | **91.0** | 85 | 75 | **67.5** | 🔵 Recommended |
| pH | 67% | **76.5** | 85 | 75 | **65.8** | 🔵 Recommended |
| Flow_Rate | 100% | **74.0** | 85 | 75 | **72.4** | 🔵 Recommended |
| Pressure | 33% | **88.0** | 85 | 75 | **66.7** | 🔵 Recommended |
| Vibration | 67% | **63.0** | 85 | 75 | **62.0** | 🟡 Consider |

> SelectionFreq is dampened: `adjusted = Coverage% × max(PS, 25) / 100`
> This prevents a weak feature from reaching high FinalScore purely through high coverage.

---

## Phase 6 — Multicollinearity Deduplication (new)

After all scores and recommendations are assigned, a **post-scoring deduplication pass** runs on the final consensus DataFrame. This prevents two features that measure essentially the same thing from both landing in the Recommended list.

### Why This Is Needed

Every scoring method evaluates features independently. If `DMCTF_Feed` and `CHG_GAS_FLOW_TO_DRYER` are correlated at r=0.9746, both can independently score high on PS and SelectionFreq and both land as Recommended — even though including both adds no information to the model.

The Overview tab already *detects* these pairs. Phase 6 *acts* on them.

### How It Works

**Input:** The final consensus DataFrame + X–X correlation matrix + `corr_threshold` (set in UI, default 0.85)

**Step 1 — Collect pairs**

Scan the X–X correlation matrix for all pairs of Recommended / Highly Recommended features where |r| > corr_threshold. Sort pairs by |r| descending (most redundant first).

**Step 2 — Greedy resolution**

Walk the sorted list. For each pair (A, B) where both are still Recommended/HR:

| Decision | Rule |
|---|---|
| **Winner** | Feature with higher FinalScore — already accounts for PS, FQ, Stability |
| **Loser** | Downgraded from Recommended → **Consider** |
| **Column stamped** | `MulticollinearWith` = "Winner (|r|=X.XXXX)" |
| **Reasoning updated** | "Downgraded from Recommended: highly correlated with Winner (|r|=X.XXXX) — include only one of this pair." |

If a feature was already downgraded in a prior pair, it is **skipped** in later pairs — it can only be a loser once.

**Step 3 — Re-derive buckets**

`recommended_features`, `optional_features`, `features_to_remove` are re-derived from the deduped DataFrame. Quick-apply "Use Recommended" draws from this final list.

### Example — Your Dataset

| Pair | |r| | Winner | Loser → |
|---|---|---|---|
| K1301 vs Quench tower overhead temp | 0.9998 | K1301 (higher score) | Quench → Consider |
| K1301 vs Top Reflux Temp | 0.9990 | K1301 | Top Reflux → Consider |
| Quench vs Top Reflux Temp | 0.9986 | — | **Skipped** (Quench already downgraded) |
| DMCTF_Feed vs CHG_GAS_FLOW_TO_DRYER | 0.9746 | DMCTF_Feed (higher score) | CHG → Consider |

**Result in Tab2:**

| Feature | Recommendation | MulticollinearWith |
|---|---|---|
| DMCTF_Feed | 🔵 Recommended | — |
| CHG_GAS_FLOW_TO_DRYER | 🟡 Consider | DMCTF_Feed (\|r\|=0.9746) |
| K1301_1ST_STG_SUCT_TEMP | 🟢 Highly Recommended | — |
| Quench tower overhead temp | 🟡 Consider | K1301_1ST_STG_SUCT_TEMP (\|r\|=0.9998) |
| Top Reflux Temp | 🟡 Consider | K1301_1ST_STG_SUCT_TEMP (\|r\|=0.9990) |

### Where It Runs

| Scenario | Where dedup runs |
|---|---|
| **Single-Y** | Once, inside `run_auto_feature_selection()` after consensus is built |
| **Multi-Y** | Once, inside `run_per_target_auto_selection()` after the aggregate consensus is built. Per-target sub-runs do NOT run dedup — this keeps union/coverage calculation clean. |

### Key Design Decisions

| Decision | Reason |
|---|---|
| Downgrade to **Consider**, not **Remove** | The feature is still predictive — just redundant given its correlated partner. User can still manually include it. |
| Keep **higher FinalScore** as winner | FinalScore already encodes PS + FQ + Stability — it's the best composite proxy for "better representative" |
| **Greedy from highest |r| first** | Resolves the most redundant pairs first; a downgraded feature is never reconsidered |
| **Only Recommended/HR** scanned | Consider and Weak Feature are already below the bar — no need to further downgrade |
| Uses **user's corr_threshold** | Same threshold shown in the UI Overview pair table — consistent and transparent |

---

## All Scenarios Covered

### Scenario 1 — Specialist Feature (Strong for 1 Target)

> **Temperature**: PS=91 for Product_Quality, rejected by Energy and Yield

| Step | Old Approach | New Approach |
|---|---|---|
| PS used | mean(91, 20, 17) = **42.7** | mean(91) = **91.0** |
| Result | Consider ❌ | Recommended ✅ |

**Why it works:** `feature_target_map["Temperature"] = ["Product_Quality"]` — only Y1's PS is used.

---

### Scenario 2 — Generalist Feature (Selected by All Targets)

> **Flow_Rate**: PS=67, 81, 74 across all 3 targets

| Step | Old Approach | New Approach |
|---|---|---|
| PS used | mean(67, 81, 74) = **74.0** | mean(67, 81, 74) = **74.0** |
| Result | Recommended ✅ | Recommended ✅ |

**Why it works:** When all targets select a feature, both approaches give identical PS.
The generalist also gets Coverage=100%, boosting SelectionFreq.

---

### Scenario 3 — Partial Specialist (Selected by Some, Rejected by Others)

> **Vibration**: PS=10 (Y1 rejected), PS=73 (Y2 selected), PS=53 (Y3 selected)

| Step | Old Approach | New Approach |
|---|---|---|
| PS used | mean(10, 73, 53) = **45.3** | mean(73, 53) = **63.0** |
| Result | Consider ⚠️ | Consider (higher score) ✅ |

**Why it works:** Y1's irrelevant PS=10 is excluded. Only Y2 and Y3 (which found Vibration useful) contribute.

---

### Scenario 4 — Feature Rejected by All Targets

> **Humidity**: PS=5, 5, 4 — rejected by all 3 targets

| Step | Result |
|---|---|
| In union? | ❌ No — excluded at Phase 2 |
| PS computed? | Never reaches aggregation |
| Recommendation | Not shown |

**Why it works:** The union only includes features recommended by at least one target.
Universally rejected features are dropped early.

---

### Scenario 5 — Single Y Target (Unchanged Path)

> Dataset has only **1 Y target** (e.g., Product_Quality only)

| Step | What Happens |
|---|---|
| Engine used | `run_auto_feature_selection` directly — per-target path is NOT entered |
| Methods run | All 5 methods against the single Y |
| SelectionFreq | Fraction of 5 methods that selected feature (original meaning) |
| PS | Normal weighted combination of 5 method scores |
| Result | Identical to original behavior — completely unaffected |

**Why it works:** The UI branches on `len(y_cols) > 1`. Single-Y always uses the original engine.

---

### Scenario 6 — Feature in "Consider" for One Target Only

> **pH** was Consider (not Recommended) for Y2, Recommended for Y1 and Y3

| Step | Result |
|---|---|
| In union? | ✅ Yes — Y1 and Y3 recommended it |
| In feature_target_map? | `["Product_Quality", "Yield_Rate"]` |
| PS used | mean(82, 71) = **76.5** |
| Y2's PS=10 excluded? | ✅ Yes — Y2 did not recommend it |

Consider-only targets that did NOT reach Recommended are excluded from PS.

---

### Scenario 7 — High Coverage, Low PS (Noisy Generalist)

> A feature selected by all 3 targets but with weak PS everywhere: PS=35, 30, 32

| Component | Value |
|---|---|
| Coverage% | 100% |
| PS (mean selected) | 32.3 |
| adjusted_freq | 100 × max(32.3, 25) / 100 = **32.3** |
| FinalScore | 0.25×32.3 + 0.40×32.3 + 0.20×85 + 0.15×75 = **47.5** |
| Recommendation | 🟡 Consider |

**Why it works:** The damping factor `× max(PS, 25)/100` prevents high Coverage from
artificially elevating a weak feature. SelectionFreq and PS are coupled — a weak PS
caps the SelectionFreq contribution.

---

### Scenario 8 — Very High N Targets (e.g., 10 Y Variables)

> Feature A: PS=90 for 2 targets (specialist), PS≈10 for 8 targets

| Step | Old Approach | New Approach |
|---|---|---|
| Targets in PS calc | All 10 | 2 selected targets only |
| PS | mean(90,90,10,10,10,10,10,10,10,10) = **26** | mean(90,90) = **90** |
| Coverage% | 20% | 20% |
| FinalScore | ~38 → Weak ❌ | ~62 → Consider / Recommended ✅ |

The specialist is rescued regardless of how many total targets exist.

---

### Scenario 9 — Multicollinear Pair in Recommended List

> **DMCTF_Feed** (FinalScore=68) and **CHG_GAS_FLOW_TO_DRYER** (FinalScore=62), |r|=0.9746

Both features independently scored above the Recommended threshold. Without deduplication, both would land in the "Use Recommended" list even though they carry nearly identical information.

**Phase 5 output (before dedup):**

| Feature | PS | FinalScore | Recommendation |
|---|---|---|---|
| DMCTF_Feed | 72 | 68.0 | 🔵 Recommended |
| CHG_GAS_FLOW_TO_DRYER | 65 | 62.0 | 🔵 Recommended |

**Phase 6 dedup pass:**

| Step | Action |
|---|---|
| Pair detected | DMCTF_Feed ↔ CHG_GAS_FLOW_TO_DRYER, \|r\|=0.9746 > threshold(0.85) |
| Winner | DMCTF_Feed — higher FinalScore (68 > 62) |
| Loser downgraded | CHG_GAS_FLOW_TO_DRYER → Consider |
| Column stamped | `MulticollinearWith` = "DMCTF_Feed (\|r\|=0.9746)" |

**Phase 6 output (after dedup):**

| Feature | FinalScore | Recommendation | MulticollinearWith |
|---|---|---|---|
| DMCTF_Feed | 68.0 | 🔵 Recommended | — |
| CHG_GAS_FLOW_TO_DRYER | 62.0 | 🟡 Consider | DMCTF_Feed (\|r\|=0.9746) |

**Why it works:** The user can include CHG_GAS_FLOW_TO_DRYER manually if domain knowledge requires it, but the default "Use Recommended" set is clean — no redundant pair.

---

## Final Recommendation Summary

| Recommendation | Criteria |
|---|---|
| 🟢 **Highly Recommended** | FinalScore ≥ 80 AND PS ≥ 70 AND FQ ≥ 60 AND VIF ≤ 10 |
| 🔵 **Recommended** | FinalScore ≥ 60 AND PS ≥ 50 AND FQ ≥ 40 |
| 🟡 **Consider** | FinalScore ≥ 40 |
| 🔴 **Weak Feature** | FinalScore < 40 OR (PS < 30 AND FQ < 20) |

> Thresholds are softened automatically for multi-Y datasets via the `n_targets` parameter
> in `_assign_recommendation()` — each additional Y target reduces PS thresholds by 8%
> (capped at 4 extra targets = 32% max softening).

---

## What Does NOT Change

| Component | Status |
|---|---|
| `_compute_predictive_strength()` | Unchanged — called per target |
| `_compute_feature_quality()` | Unchanged — VIF + missing + variance on full X |
| `_compute_stability_score()` | Unchanged — bootstrap on full X/Y |
| `_assign_recommendation()` | Unchanged — same thresholds, called before dedup |
| `_generate_reasoning()` | Unchanged — uses best-target method results; dedup appends a note |
| `_dedup_multicollinear()` | **New** — post-scoring pass; does NOT change scores, only Recommendation + MulticollinearWith column |
| Single-Y path | Unchanged — uses `run_auto_feature_selection` directly (dedup runs once at end) |
| All UI tabs | Unchanged — tab2, tab3, tab4, tab5 work via pseudo AutoSelectionResult |
| Tab2 | Enhanced — now shows `MulticollinearWith` column for downgraded features |
| Tab6 | Enhanced — now shows per-target PS breakdown table |
