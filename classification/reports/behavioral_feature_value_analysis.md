# Behavioral Feature Value Analysis

**Date:** 2026-03-16
**Context:** During preparation of the clean repo for publication, we updated all ablation sections (4.3, 4.4, 13.2 in classification_results.md) from fixed-parameter results to grid-searched results. This revealed that behavioral features contribute far less than the original fixed-param ablation suggested.

This document captures the full investigation into where (and whether) screenome behavioral features add predictive value.

---

## 1. Background

The classification model predicts direction of CES-D change (improving / stable / worsening) using up to 39 features:
- **prior_cesd** (1 feature): current CES-D score — what a clinician already knows
- **person_mean_cesd** (1 feature): person-level trait mean CES-D from training data
- **Behavioral features** (20 base + 17 lag = 37 features): screenome-derived smartphone usage patterns

Three label operationalizations:
- **sev_crossing** (primary): crossing clinical severity boundaries (CES-D 16/24)
- **personal_sd**: exceeding person-specific SD of CES-D change (k=1.0)
- **balanced_tercile**: rank-based terciles of CES-D delta

---

## 2. Key Finding: The 2-Feature Model

A model using only **prior_cesd + person_mean_cesd** (2 features) matches or exceeds the full 39-feature model across all label types and all metrics.

### sev_crossing (primary)

| Condition | N feat | XGB AUC | XGB BalAcc | XGB Sens-W | XGB PPV-W |
|---|---|---|---|---|---|
| prior_cesd only | 1 | 0.872 | 0.764 | 0.730 | 0.229 |
| **prior_cesd + pmcesd** | **2** | **0.904** | **0.807** | **0.811** | **0.337** |
| behav + lag, no prior | 37 | 0.638 | 0.376 | 0.081 | 0.176 |
| base + lag | 38 | 0.875 | 0.735 | 0.622 | 0.232 |
| full | 39 | 0.906 | 0.834 | 0.838 | 0.356 |

### personal_sd

| Condition | N feat | Best AUC | Best BalAcc | Best Sens-W |
|---|---|---|---|---|
| prior_cesd only | 1 | 0.695 (EN/SVM) | 0.552 | 0.220 |
| **prior_cesd + pmcesd** | **2** | **0.767 (EN/SVM)** | **0.644** | **0.683** |
| behav + lag, no prior | 37 | 0.554 | 0.370 | 0.317 |
| base + lag | 38 | 0.708 (EN) | 0.530 | 0.463 |
| full | 39 | 0.759 (EN) | 0.624 | 0.634 |

### balanced_tercile

| Condition | N feat | Best AUC | Best BalAcc | Best Sens-W |
|---|---|---|---|---|
| prior_cesd only | 1 | 0.665 (EN) | 0.499 | 0.226 |
| **prior_cesd + pmcesd** | **2** | **0.735 (SVM)** | **0.552** | **0.591** |
| behav + lag, no prior | 37 | 0.593 | 0.406 | 0.350 |
| base + lag | 38 | 0.659 (XGB) | 0.489 | 0.328 |
| full | 39 | 0.732 (LGBM) | 0.557 | 0.562 |

---

## 3. Paired Bootstrap Significance Tests

1000-resample percentile bootstrap, 95% CIs. SIG = p < 0.05 and CI excludes 0.

### Transition: prior_cesd (1) → prior_cesd + pmcesd (2)
**Significant for ALL models across ALL label types.** person_mean_cesd is the sole significant additive feature.

| Label | ElasticNet | XGBoost | LightGBM | SVM |
|---|---|---|---|---|
| sev_crossing | +0.095 SIG | +0.032 SIG | +0.044 SIG | +0.025 SIG |
| personal_sd | +0.074 SIG | +0.096 SIG | +0.106 SIG | +0.073 SIG |
| balanced_tercile | +0.066 SIG | +0.077 SIG | +0.081 SIG | +0.077 SIG |

### Transition: prior_cesd + pmcesd (2) → full (39)
**Not significant for any model on any label type.** Adding 37 behavioral features to the 2-feature model adds zero significant value.

| Label | ElasticNet | XGBoost | LightGBM | SVM |
|---|---|---|---|---|
| sev_crossing | −0.025 SIG− | +0.001 ns | −0.017 SIG− | −0.058 SIG− |
| personal_sd | −0.009 ns | +0.007 ns | −0.012 ns | −0.017 ns |
| balanced_tercile | −0.014 SIG− | −0.001 ns | +0.009 ns | −0.014 SIG− |

Note: SIG− means the 39-feat model is significantly *worse* than 2-feat.

### Transition: prior_cesd + pmcesd (2) → base + lag (38)
**Dropping person_mean_cesd and adding 36 behavioral features makes the model significantly worse** across all labels.

### Transition: behav + lag (37) → full (39)
Adding prior_cesd + person_mean_cesd to behavioral features is always massively significant (+0.14 to +0.30 AUC).

---

## 4. Why Does This Happen?

### Not an overfitting artifact
We tested both per-condition grid search and fixed params (from the 39-feat model applied to all conditions). The pattern is identical either way. Val-test AUC gaps are ≤0 (test ≥ val), ruling out overfitting.

### Feature importance tells a different story than prediction
In the 39-feat XGBoost model, behavioral features account for **55.3%** of feature importance (split-based). The model IS using them — they just don't change predictions. Top behavioral features:
- `lag_mean_daily_social_ratio` (6.0%)
- `lag_mean_daily_social_screens_delta` (4.7%)
- `clip_dispersion_delta` (4.2%)
- `mean_daily_overnight_ratio_delta` (4.1%)
- `mean_daily_switches_delta` (4.1%)

### Disagreement analysis (sev_crossing, XGBoost)
The 2-feat and 39-feat models disagree on only **3 / 411** test cases (0.7%). When they disagree, 39-feat is right on 2 and 2-feat is right on 1.

### Per-person analysis
89/92 persons (97%) have BalAcc within ±5pp between models. No severity subgroup benefits consistently from behavioral features.

---

## 5. Interpretation: What IS the Value of Behavioral Features?

### What they don't provide (in this dataset):
- **No significant predictive value** above prior_cesd + person_mean_cesd for any label operationalization
- **No cold-start value**: behavioral features alone (without CESD) perform near chance (AUC 0.52–0.66)
- **No person-specific value**: no subgroup of people consistently benefits from behavioral features

### What they DO provide:

1. **Mechanistic interpretability**: Behavioral features explain *how* worsening manifests — social media withdrawal (`lag_mean_daily_social_ratio`), narrowing content variety (`clip_dispersion_delta`), disrupted sleep timing (`mean_daily_overnight_ratio_delta`), reduced app switching (`mean_daily_switches_delta`). The CESD features predict *who* worsens but not *why*. This is valuable for:
   - Intervention design (what behaviors to target)
   - Clinical face validity (explaining predictions to clinicians)
   - Generating hypotheses about depression dynamics

2. **The null result IS scientifically important**: Demonstrating that screenome behavioral digital phenotyping does not add predictive value above simple clinical anchors is a finding the field needs. Many digital phenotyping studies compare to chance baselines rather than to what a clinician already knows. This study's clinician-baseline framing sets a higher standard.

3. **Label-independent consistency**: The pattern holds across three different label operationalizations (clinical thresholds, personalized SD, rank-based), confirming it's not an artifact of how "worsening" is defined.

---

## 6. Lag Features Specifically

Lag-1 behavioral features (previous period's behavior) add nothing over base features:

| Label | Model | base(21) AUC | base+lag(38) AUC | Δ |
|---|---|---|---|---|
| sev_crossing | XGBoost | 0.876 | 0.875 | −0.001 |
| personal_sd | XGBoost | 0.693 | 0.690 | −0.003 |
| balanced_tercile | XGBoost | 0.655 | 0.659 | +0.004 |

None of these differences are significant in the paired bootstrap. The old fixed-param ablation showed lag helping (+0.015 AUC for XGB) because the fixed params happened to be suboptimal for the 21-feature set. With per-condition tuning, the apparent lag contribution vanishes.

---

## 7. Updated Ablation Narrative

The ablation sections (4.3, 4.4, 13.2 in classification_results.md) have been updated with grid-searched numbers. The narrative now correctly reflects:

- **person_mean_cesd is the sole significant additive feature** (prior_cesd → 2-feat: SIG for all models)
- **Behavioral features add marginal signal at best** (2-feat → 39-feat: non-significant)
- **The trait feature resolves the mod→sev failure mode** (0% → 88% detection for XGBoost)
- **Behavioral features without CESD anchors perform near chance** (AUC 0.53–0.66)

---

## 8. Recommended Paper Framing

1. **Lead with the clinician-baseline comparison**: Prior CESD alone achieves AUC 0.87 (sev_crossing). The relevant question is what digital phenotyping adds above this.

2. **person_mean_cesd is the key finding**: A simple trait-level severity anchor raises AUC to 0.90 and resolves the mod→sev failure mode. This is the single most impactful feature addition.

3. **Behavioral features provide mechanistic insight, not predictive lift**: The model uses behavioral features (55% of split importance) but they are redundant given the CESD anchors. Their value is in explaining the behavioral correlates of mood trajectory change — social withdrawal, sleep disruption, narrowing content variety.

4. **The null result for behavioral prediction is itself a contribution**: Most digital phenotyping studies report accuracy against chance. Against the appropriate clinician baseline, smartphone behavioral features alone perform near chance. This calibrates expectations for the field.

---

## Appendix: Analysis Parameters

- All models: ElasticNet, XGBoost, LightGBM, SVM
- Grid search: per-condition, on validation set (N=395)
- Bootstrap: 1000 resamples, percentile CIs, seed=42
- Test set: N=411, 96 persons
- Feature conditions: 1 (prior_cesd), 2 (prior_cesd + pmcesd), 21 (base), 37 (behav+lag no prior), 38 (base+lag), 39 (full)
- Script: `scripts/run_ablation_update.py` (working repo)
- Results CSV: `models/bootstrap_ci/ablation_results.csv`
