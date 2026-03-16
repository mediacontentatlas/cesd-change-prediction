# Feature Importance — Multi-Method Comparison

## Overview

Four complementary feature importance methods are compared to understand what drives predictions across the three outcome classes (improving, stable, worsening). The methods differ in what they capture:

| Method | Model | What it measures | Signed? | Per-class? |
|---|---|---|---|---|
| **Gain importance** | XGBoost | Reduction in loss at splits using this feature | No | No |
| **Gain importance** | LightGBM | Number of splits using this feature | No | No |
| **SHAP** | XGBoost, LightGBM | Marginal contribution to each prediction | Yes | Yes |
| **ElasticNet coefficients** | Logistic Regression | Linear association with log-odds of each class | Yes | Yes |

Because SHAP values for tree models are extremely sparse (max_depth=3 means only ~5 features per class get nonzero SHAP), **ElasticNet coefficients and gain importance provide essential complementary views** that reveal behavioral feature contributions invisible to SHAP.

---

## 1. XGBoost Gain Importance (Top Features)

Gain importance reflects how much each feature reduces the training loss across all trees and all classes. It is unsigned and not class-specific.

| Rank | Feature | Gain |
|---|---|---|
| 1 | **prior_cesd** | 0.272 |
| 2 | **person_mean_cesd** | 0.175 |
| 3 | lag_mean_daily_social_ratio | 0.060 |
| 4 | lag_mean_daily_social_screens_delta | 0.047 |
| 5 | clip_dispersion_delta | 0.042 |
| 6 | mean_daily_overnight_ratio_delta | 0.041 |
| 7 | mean_daily_switches_delta | 0.041 |
| 8 | mean_daily_screens | 0.039 |
| 9 | mean_daily_screens_delta | 0.038 |
| 10 | lag_mean_daily_social_ratio_delta | 0.036 |

**Key finding**: 18 of 39 features have nonzero gain in XGBoost. After the two CES-D features (45% combined), the next tier is behavioral — social media ratio from the prior period, changes in switching behavior, clip dispersion change, and overnight usage change. These features refine predictions within CES-D-defined splits.

---

## 2. LightGBM Gain Importance (Top Features)

LightGBM uses split count rather than loss reduction.

| Rank | Feature | Splits |
|---|---|---|
| 1 | **person_mean_cesd** | 300 |
| 2 | **prior_cesd** | 269 |
| 3 | mean_daily_switches_delta | 33 |
| 4 | mean_daily_screens_delta | 22 |
| 5 | lag_mean_daily_social_ratio | 22 |
| 6 | lag_switches_per_screen | 17 |
| 7 | mean_daily_switches | 12 |
| 8 | switches_per_screen_delta | 12 |
| 9 | mean_daily_overnight_ratio_delta | 9 |
| 10 | mean_daily_screens | 4 |

**Key finding**: LightGBM uses only 10 of 39 features total — even more sparse than XGBoost. The behavioral signal is concentrated in switching behavior (switches, switches_delta, switches_per_screen), daily screen activity, social ratio, and overnight usage change.

---

## 3. SHAP Values — Cross-Class Aggregation

SHAP values are computed per class. Because shallow trees (max_depth=3) create extreme sparsity — only 5 features have nonzero SHAP for the worsening class — we aggregate mean |SHAP| across all 3 classes for a complete picture.

### XGBoost — Top Behavioral Features by Cross-Class Mean |SHAP|

| Feature | Improving | Stable | Worsening | Total |
|---|---|---|---|---|
| **prior_cesd** | 0.657 | 0.217 | 0.300 | **1.174** |
| **person_mean_cesd** | 0.096 | 0.388 | 0.393 | **0.877** |
| mean_daily_switches_delta | 0.000 | 0.016 | 0.000 | 0.016 |
| switches_per_screen_delta | 0.000 | 0.012 | 0.000 | 0.012 |
| mean_daily_overnight_ratio_delta | 0.009 | 0.000 | 0.000 | 0.009 |
| lag_mean_daily_social_ratio | 0.008 | 0.000 | 0.000 | 0.008 |
| mean_daily_social_screens_delta | 0.008 | 0.003 | 0.000 | 0.011 |
| mean_daily_screens_delta | 0.000 | 0.006 | 0.007 | 0.013 |
| lag_clip_dispersion_delta | 0.006 | 0.000 | 0.000 | 0.006 |
| mean_daily_screens | 0.005 | 0.000 | 0.000 | 0.005 |

**Key finding**: Behavioral features contribute to SHAP through the *stable* and *improving* classes, not the worsening class. The worsening class is almost entirely driven by the two CES-D features because worsening is so rare (9%) that the model needs only 3-5 features to classify it.

### LightGBM — Top Behavioral Features by Cross-Class Mean |SHAP|

| Feature | Improving | Stable | Worsening | Total |
|---|---|---|---|---|
| **prior_cesd** | 0.340 | 0.146 | 0.166 | **0.652** |
| **person_mean_cesd** | 0.060 | 0.285 | 0.237 | **0.581** |
| mean_daily_switches_delta | 0.000 | 0.015 | 0.000 | 0.015 |
| lag_switches_per_screen | 0.000 | 0.011 | 0.000 | 0.011 |
| mean_daily_screens_delta | 0.000 | 0.010 | 0.001 | 0.011 |
| lag_mean_daily_social_ratio | 0.008 | 0.000 | 0.000 | 0.008 |
| switches_per_screen_delta | 0.000 | 0.004 | 0.000 | 0.004 |
| mean_daily_screens | 0.000 | 0.004 | 0.000 | 0.004 |
| mean_daily_switches | 0.003 | 0.000 | 0.000 | 0.003 |
| mean_daily_overnight_ratio_delta | 0.000 | 0.003 | 0.000 | 0.003 |

---

## 4. ElasticNet Coefficients — Signed Per-Class Feature Weights

ElasticNet provides the most interpretable view because coefficients are signed (direction of effect) and per-class. Two regularization levels are reported.

### Canonical ElasticNet (C=0.01, l1_ratio=0.99 — high regularization)

This is the hyperparameter-tuned version that achieved best validation accuracy. Extreme sparsity — only 3 features survive:

| Feature | Improving | Stable | Worsening |
|---|---|---|---|
| prior_cesd | **+0.186** | 0.000 | **−0.123** |
| person_mean_cesd | −0.048 | 0.000 | **+0.203** |

**Interpretation**: Higher prior CES-D is associated with improving (regression to mean). Higher person-level mean CES-D is associated with worsening. No behavioral features survive the strong L1 penalty.

### Less-Regularized ElasticNet (C=0.1, l1_ratio=0.9 — for interpretability)

This version trades some predictive optimality for a richer picture of which features the model *would* use with less regularization. **15 behavioral features have nonzero coefficients.**

#### Worsening Class — Behavioral Risk Factors

| Feature | Coefficient | Interpretation |
|---|---|---|
| mean_daily_social_screens_delta | **+0.202** | Increase in social media screen time → higher worsening risk |
| mean_daily_unique_apps_delta | +0.066 | Increased app diversity → worsening |
| switches_per_screen | +0.059 | Higher switching rate → worsening |
| age | +0.057 | Older participants → slightly higher worsening risk |
| mean_daily_screens | −0.055 | Fewer total screens → worsening |
| clip_dispersion_delta | +0.047 | Increased temporal dispersion → worsening |
| lag_mean_daily_social_screens | +0.036 | Prior period social screens → worsening |
| lag_clip_dispersion | +0.013 | Prior dispersion level → worsening |

#### Improving Class — Behavioral Protective Factors

| Feature | Coefficient | Interpretation |
|---|---|---|
| lag_clip_dispersion_delta | **+0.072** | Increased prior-period dispersion change → improving |
| lag_mean_daily_social_screens_delta | +0.028 | Prior increase in social screens → now improving |
| lag_mean_daily_unique_apps_delta | +0.017 | Prior increase in app diversity → now improving |
| lag_mean_daily_screens_delta | +0.006 | Prior screen increase → now improving |
| mean_daily_social_screens_delta | −0.029 | Current social screen increase → less likely improving |
| lag_mean_daily_unique_apps | −0.034 | Prior high app diversity → less likely improving |
| clip_dispersion_delta | −0.021 | Current dispersion increase → less likely improving |

#### Stable Class — Features That Predict No Change

| Feature | Coefficient | Interpretation |
|---|---|---|
| mean_daily_switches_delta | **−0.101** | Large switching changes → less stable |
| mean_daily_screens | +0.057 | Higher overall screen use → more stable |
| lag_mean_daily_screens_delta | −0.054 | Prior screen changes → less stable |
| lag_mean_daily_social_screens | −0.050 | Prior social screens → less stable |
| lag_clip_dispersion | −0.040 | Prior dispersion → less stable |
| switches_per_screen | −0.035 | Higher switching rate → less stable |

---

## 5. Cross-Method Convergence

The following behavioral features appear as important across multiple methods:

| Feature | XGB Gain | LGBM Gain | SHAP (any class) | ElasticNet (interp) | Consensus |
|---|---|---|---|---|---|
| mean_daily_switches_delta | rank 7 | rank 3 | stable class | stable (−0.10) | **4/4 methods** |
| mean_daily_screens_delta | rank 9 | rank 4 | stable, worsening | — | 3/4 |
| mean_daily_social_screens_delta | rank 11 | — | improving | worsening (+0.20) | 3/4 |
| lag_mean_daily_social_ratio | rank 3 | rank 5 | improving | — | 3/4 |
| clip_dispersion_delta | rank 5 | — | improving | worsening (+0.05) | 3/4 |
| switches_per_screen_delta | rank 17 | rank 8 | stable | — | 3/4 |
| mean_daily_overnight_ratio_delta | rank 6 | rank 9 | improving | — | 3/4 |
| mean_daily_screens | rank 8 | rank 10 | improving | stable (+0.06), wors (−0.05) | 4/4 |

**Bottom line**: While the two CES-D features dominate all methods, **behavioral features — especially changes in app switching, social media time, screen counts, and usage dispersion — consistently appear across multiple methods and classes**. These features don't substantially improve AUC (ablation shows +0.004, n.s.) but they refine within-class predictions and would become more important in deployment scenarios without current CES-D.

---

## 6. Cross-Label Comparison: Feature Importance Under Alternative Labels

Feature importance was extracted for all three label types using the same methods (XGBoost gain, LightGBM gain, SHAP, ElasticNet coefficients). The alternative labels reveal dramatically different feature utilization patterns.

### Feature utilization by label type

| Metric | sev_crossing | balanced_tercile | personal_sd |
|---|---|---|---|
| XGBoost nonzero gain features | 18/39 | **39/39** | **37/39** |
| LightGBM nonzero gain features | 10/39 | **39/39** | **37/39** |
| XGBoost nonzero SHAP features (worsening) | 5/39 | **38/39** | **33/39** |
| ElasticNet nonzero coefficients (canonical) | 3/117 | **60/117** | **45/117** |
| ElasticNet nonzero coefficients (interp) | 15/117 | **31/117** | **31/117** |
| CES-D features' share of gain (XGB) | 45% | 15% | 13% |

**Key finding**: The sev_crossing label is so dominated by CES-D features (clinical threshold boundaries) that the tree models barely use behavioral features. Under balanced tercile and personal SD labels — where predicting change magnitude rather than threshold crossings — **the models engage nearly all 39 features**, with behavioral features receiving much more balanced importance.

### XGBoost Gain — Top 10 Features by Label

#### sev_crossing (CES-D dominated)

| Rank | Feature | Gain |
|---|---|---|
| 1 | prior_cesd | 0.272 |
| 2 | person_mean_cesd | 0.175 |
| 3 | lag_mean_daily_social_ratio | 0.060 |
| 4–10 | (behavioral features) | 0.036–0.047 |

#### balanced_tercile (much more distributed)

| Rank | Feature | Gain |
|---|---|---|
| 1 | prior_cesd | 0.074 |
| 2 | person_mean_cesd | 0.072 |
| 3 | age | 0.033 |
| 4 | gender_mode_1 | 0.032 |
| 5 | lag_clip_dispersion | 0.028 |
| 6 | mean_daily_social_ratio_delta | 0.028 |
| 7 | mean_daily_social_screens_delta | 0.028 |
| 8 | mean_daily_social_screens | 0.026 |
| 9 | lag_clip_dispersion_delta | 0.025 |
| 10 | mean_daily_overnight_ratio | 0.025 |

#### personal_sd (behavioral features prominent, esp. lagged)

| Rank | Feature | Gain |
|---|---|---|
| 1 | prior_cesd | 0.068 |
| 2 | person_mean_cesd | 0.063 |
| 3 | lag_mean_daily_switches | 0.037 |
| 4 | lag_mean_daily_switches_delta | 0.036 |
| 5 | lag_active_day_ratio_delta | 0.034 |
| 6 | lag_switches_per_screen_delta | 0.033 |
| 7 | lag_mean_daily_social_screens | 0.033 |
| 8 | lag_mean_daily_social_ratio_delta | 0.031 |
| 9 | lag_mean_daily_social_screens_delta | 0.030 |
| 10 | mean_daily_social_screens_delta | 0.030 |

**Interpretation**: Under personal SD labels, **lagged behavioral features dominate the top 10**, particularly switching behavior and social media usage from the prior period. This makes sense: personal SD labels measure whether someone changed more than their own typical variability, so the model needs to capture individual-level behavioral patterns over time.

### SHAP Values — Worsening Class, Cross-Label

The most dramatic difference is in SHAP for the worsening class:

**sev_crossing** — only 5 features with nonzero worsening SHAP (mean |SHAP| for top behavioral feature: 0.007)

**balanced_tercile** — 38 features with nonzero worsening SHAP:

| Feature | Worsening SHAP |
|---|---|
| mean_daily_social_ratio_delta | 0.058 |
| mean_daily_social_screens_delta | 0.061 |
| mean_daily_overnight_ratio | 0.053 |
| lag_mean_daily_switches_delta | 0.033 |
| mean_daily_overnight_ratio_delta | 0.032 |
| mean_daily_screens_delta | 0.031 |
| lag_clip_dispersion_delta | 0.029 |
| lag_switches_per_screen_delta | 0.029 |

**personal_sd** — 33 features with nonzero worsening SHAP:

| Feature | Worsening SHAP |
|---|---|
| lag_clip_dispersion_delta | 0.026 |
| mean_daily_overnight_ratio_delta | 0.018 |
| lag_switches_per_screen | 0.015 |
| switches_per_screen | 0.015 |
| mean_daily_social_screens_delta | 0.015 |
| lag_mean_daily_overnight_ratio_delta | 0.014 |
| lag_mean_daily_screens_delta | 0.013 |
| lag_mean_daily_switches_delta | 0.012 |

### ElasticNet Coefficients — Worsening Class, Cross-Label

| Feature | sev_crossing | balanced_tercile | personal_sd |
|---|---|---|---|
| person_mean_cesd | +0.249 | **+1.418** | **+0.932** |
| prior_cesd | −0.172 | **−0.755** | **−0.836** |
| mean_daily_social_screens_delta | +0.202 | +0.051 | — |
| switches_per_screen | +0.059 | +0.022 | +0.045 |
| mean_daily_overnight_ratio | — | +0.064 | — |
| lag_switches_per_screen_delta | — | +0.043 | — |
| mean_daily_overnight_ratio_delta | — | — | +0.025 |
| mean_daily_social_ratio_delta | — | — | +0.014 |

**Key finding across labels**: The CES-D features have **much larger** coefficients under alternative labels (person_mean_cesd: +0.249 for sev_crossing vs +1.418 for balanced tercile). This is because the alternative labels are harder tasks where the model has to "work harder" — the CES-D signal is less directly aligned with the label definition, so larger coefficients are needed to extract the weaker signal.

### Summary: What Alternative Labels Reveal

1. **Behavioral features matter more under harder labels**. When the label isn't directly determined by CES-D threshold crossings, the models must use behavioral features to discriminate.
2. **Social media and switching behavior are the most consistent behavioral signals** across all three labels and all methods.
3. **Lagged features become particularly important under personal SD labels**, suggesting that individual-level behavioral trajectories (not just current-period values) help predict person-specific unusual changes.
4. **The sev_crossing SHAP sparsity is a property of the label, not the model**. The same XGBoost architecture produces rich SHAP distributions under alternative labels.

---

## Source Files

### sev_crossing (primary label)

| File | Contents |
|---|---|
| `models/feature_importance/feature_importance.csv` | XGBoost gain importance |
| `models/feature_importance/lgbm_feature_importance.csv` | LightGBM gain importance |
| `models/feature_importance/feature_coefficients.csv` | ElasticNet canonical (C=0.01, l1=0.99) |
| `models/feature_importance/feature_coefficients_interp.csv` | ElasticNet interpretable (C=0.1, l1=0.9) |
| `models/feature_importance/shap_summary_xgboost.csv` | XGBoost SHAP per feature × class |
| `models/feature_importance/shap_summary_lightgbm.csv` | LightGBM SHAP per feature × class |

### balanced_tercile

| File | Contents |
|---|---|
| `models/feature_importance/balanced_tercile/feature_importance.csv` | XGBoost gain importance |
| `models/feature_importance/balanced_tercile/lgbm_feature_importance.csv` | LightGBM gain importance |
| `models/feature_importance/balanced_tercile/feature_coefficients.csv` | ElasticNet canonical |
| `models/feature_importance/balanced_tercile/feature_coefficients_interp.csv` | ElasticNet interpretable (C=0.1, l1=0.9) |
| `models/feature_importance/balanced_tercile/shap_summary_xgboost.csv` | XGBoost SHAP per feature × class |
| `models/feature_importance/balanced_tercile/shap_summary_lightgbm.csv` | LightGBM SHAP per feature × class |

### personal_sd

| File | Contents |
|---|---|
| `models/feature_importance/personal_sd/feature_importance.csv` | XGBoost gain importance |
| `models/feature_importance/personal_sd/lgbm_feature_importance.csv` | LightGBM gain importance |
| `models/feature_importance/personal_sd/feature_coefficients.csv` | ElasticNet canonical |
| `models/feature_importance/personal_sd/feature_coefficients_interp.csv` | ElasticNet interpretable (C=0.1, l1=0.9) |
| `models/feature_importance/personal_sd/shap_summary_xgboost.csv` | XGBoost SHAP per feature × class |
| `models/feature_importance/personal_sd/shap_summary_lightgbm.csv` | LightGBM SHAP per feature × class |

### Figures

| File | Contents |
|---|---|
| `reports/figures/12_feature_importance.png` | Gain + coefficient figure (sev_crossing) |
| `reports/figures/14_shap_summary.png` | SHAP stacked bar chart (3 classes, sev_crossing) |
