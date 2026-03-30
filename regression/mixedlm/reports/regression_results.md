# MixedLM Results Summary

## 1. Task Definition

The outcome is the **continuous CES-D delta** (change in CES-D score from one biweekly assessment to the next). Positive values indicate worsening; negative values indicate improvement.

| Property | Value |
|---|---|
| Target variable | CES-D delta (continuous, range approximately −40 to +40) |
| Unit | CES-D points |
| Task type | Within-person longitudinal regression |
| Participants | 96 (all present in every split) |
| Train | 1,196 observations |
| Val | 395 observations |
| Test | 411 observations |

This is a **within-person longitudinal regression task** — not between-person depression severity prediction. CES-D delta has near-zero between-person ICC (ICC ~ 0; person means explain only 0.4% of delta variance), meaning no person consistently changes more than another. The prediction target is inherently within-person.

**Relationship to classification task:** The classification task predicts the *direction* of change (improving / stable / worsening). This regression task predicts the *magnitude* of change. The regression predictions can be post-hoc converted to direction predictions for direct comparison with classifiers.

---

## 2. Model: Mixed-Effects Linear Model (MixedLM)

### 2.1 Model specification

`statsmodels.MixedLM` with REML estimation.

```
y_delta ~ X_fixed + (1 | person_id)       # random intercept only (primary)
y_delta ~ X_fixed + (1 + Z | person_id)    # random intercept + slopes (sweep)
```

| Setting | Value |
|---|---|
| Estimation | REML (Restricted Maximum Likelihood) |
| Fixed effects | Feature-dependent (1–30 features, see ablation conditions) |
| Random effects | Per-person random intercept (primary); additional random slopes explored in sweep |
| Optimization | Sequential fallback: `lbfgs` → `bfgs` → `powell` |
| Convergence | If random slopes fail convergence, simplifies to random intercept only |
| Prediction (known persons) | Fixed effects + person-specific random effects (intercept + slopes) |
| Prediction (new persons) | Fixed effects only (population-level prediction) |

### 2.2 Why MixedLM

The mixed-effects model accounts for repeated measures within participants — a structural requirement of this longitudinal dataset. The random intercept captures person-specific baseline tendencies in CES-D change, while fixed effects estimate population-average associations between behavioral features and symptom change magnitude. This allows each person's predictions to be anchored to their individual baseline, rather than forcing all persons through the same intercept.

---

## 3. Features

### 3.1 Feature ablation conditions

Four nested conditions matching the classification task for direct comparability:

| Condition | Features | N features | Description |
|---|---|---|---|
| `prior_cesd` | `prior_cesd` | 1 | Baseline: prior symptom score only |
| `base` | All 21 base features | 21 | Full base feature set (behavioral + demographics + prior CES-D) |
| `base_dev` | Base + within-person deviations | 29 | Base + 8 within-person deviation features |
| `base_dev_pmcesd` | Base + dev + `person_mean_cesd` | 30 | Adds person-level chronic severity anchor |

### 3.2 Base features (21)

| Feature | Description |
|---|---|
| `prior_cesd` | CES-D score at the start of this period |
| `active_day_ratio_delta` | Change in proportion of active days |
| `mean_daily_overnight_ratio` | Proportion of overnight screen use |
| `mean_daily_overnight_ratio_delta` | Change in overnight screen proportion |
| `mean_daily_social_ratio` | Proportion of social app screen time |
| `mean_daily_social_ratio_delta` | Change in social screen proportion |
| `age` | Participant age |
| `mean_daily_screens` | Mean daily screen sessions |
| `mean_daily_screens_delta` | Change in daily screen sessions |
| `mean_daily_unique_apps` | Mean unique apps used daily |
| `mean_daily_unique_apps_delta` | Change in unique apps used |
| `mean_daily_switches` | Mean daily app switches |
| `mean_daily_switches_delta` | Change in app switches |
| `switches_per_screen` | App switches per screen session (fragmentation index) |
| `switches_per_screen_delta` | Change in fragmentation |
| `mean_daily_social_screens` | Mean daily social app sessions |
| `mean_daily_social_screens_delta` | Change in social app sessions |
| `clip_dispersion` | Variety of content viewed |
| `clip_dispersion_delta` | Change in content variety |
| `gender_mode_1`, `gender_mode_2` | Gender indicators |

> `_delta` features are changes relative to the person's own prior period — they encode within-person deviation from baseline.

### 3.3 Within-person deviation features (8)

Eight deviation features (`dev_0` through `dev_7`) computed as each person's current-period behavioral value minus their training-period mean. These capture how unusual a person's current behavior is relative to their own baseline, complementing the `_delta` features (which capture period-over-period change).

### 3.4 Person-level trait: `person_mean_cesd`

```
person_mean_cesd_i = mean(prior_cesd over all training periods for person i)
```

**No-leakage implementation:** Computed from each person's training-period observations only. Assigned to all their val and test rows via pid lookup. Val and test CES-D values never enter the computation.

**Rationale:** `prior_cesd` encodes where a person is *right now* (state). `person_mean_cesd` encodes their chronic level (trait). This distinction is critical for regression: a person with prior_cesd=20 and person_mean_cesd=8 (acutely elevated) has different expected change dynamics than a person with prior_cesd=20 and person_mean_cesd=22 (chronically moderate). See Section 6 for the large impact of this feature.

---

## 4. Metrics

All metrics are reported on train, val, and test splits.

| Metric | Description | Why it matters |
|---|---|---|
| MAE | Mean absolute error (CES-D points) | Interpretable clinical scale — average prediction error in CES-D points |
| RMSE | Root mean squared error | Penalizes large errors, clinically important for detecting extreme changes |
| R² | Overall variance explained | Standard goodness-of-fit measure |
| Within-person R² (median) | **Primary R² metric.** Median of per-person R² values | CES-D delta has near-zero between-person ICC (ICC ~ 0), so within-person R² guards against inflated estimates from person-level random effects. Uses median (not mean) because persons with very few test observations produce extreme negative R² that distorts the mean |
| Between-person R² | Variance explained between person-level means | Reported for completeness; expect it to be low or negative (CES-D delta has minimal between-person structure) |

**Important:** A high overall R² that comes primarily from person-level random effects is misleading for this target. Within-person R² (median) is the primary R² metric.

---

## 5. Baselines (B0–B4)

Five baselines are computed on the training set and applied identically to val and test:

| Baseline | Description | Rationale |
|---|---|---|
| B0: No Change | Predict 0 (no change) for everyone | Lower bound — assumes symptoms never change |
| B1: Population Mean | Predict the training-set mean delta | Assumes everyone changes by the average amount |
| B2: Last Value Carried Forward | Each person's last training delta | Tests whether momentum/inertia from last observation helps |
| B3: Person-Specific Mean | Each person's mean training delta | Best available person-specific naive forecast |
| B4: Regression to Mean | Shrunk person mean (shrinkage = 0.5) | Balances person-specific signal with population average |

---

## 6. Primary Results: Feature Ablation (Random Intercept Only)

### 6.1 Test set — all four ablation conditions

| Condition | N feat | MAE | RMSE | R² | W-R² (med) | Med. Person MAE |
|---|---|---|---|---|---|---|
| `prior_cesd` | 1 | 4.17 | 6.17 | 0.303 | 0.180 | 3.09 |
| `base` | 21 | 4.24 | 6.23 | 0.290 | 0.211 | 3.07 |
| `base_dev` | 29 | 4.30 | 6.32 | 0.270 | 0.211 | 3.51 |
| **`base_dev_pmcesd`** | **30** | **3.80** | **5.85** | **0.376** | **0.389** | **2.61** |


### 6.2 Validation set — all four ablation conditions

| Condition | N feat | MAE | RMSE | R² | W-R² (med) |
|---|---|---|---|---|---|
| `prior_cesd` | 1 | 3.90 | 5.63 | 0.275 | 0.176 |
| `base` | 21 | 3.95 | 5.68 | 0.262 | 0.189 |
| `base_dev` | 29 | 4.00 | 5.74 | 0.247 | 0.195 |
| **`base_dev_pmcesd`** | **30** | **3.45** | **5.13** | **0.399** | **0.411** |

### 6.3 Training set — all four ablation conditions

| Condition | N feat | MAE | RMSE | R² | W-R² (med) |
|---|---|---|---|---|---|
| `prior_cesd` | 1 | 3.75 | 5.71 | 0.410 | 0.408 |
| `base` | 21 | 3.76 | 5.62 | 0.427 | 0.383 |
| `base_dev` | 29 | 3.76 | 5.62 | 0.429 | 0.385 |
| **`base_dev_pmcesd`** | **30** | **3.75** | **5.63** | **0.426** | **0.385** |

### 6.4 Val → Test generalization

| Condition | Val MAE | Test MAE | Δ MAE | Val R² | Test R² | Δ R² |
|---|---|---|---|---|---|---|
| `prior_cesd` | 3.90 | 4.17 | +0.27 | 0.275 | 0.303 | +0.028 |
| `base` | 3.95 | 4.24 | +0.29 | 0.262 | 0.290 | +0.028 |
| `base_dev` | 4.00 | 4.30 | +0.30 | 0.247 | 0.270 | +0.023 |
| `base_dev_pmcesd` | 3.45 | 3.80 | +0.35 | 0.399 | 0.376 | −0.023 |

MAE increases modestly from val to test (~0.3 CES-D points), consistent with mild temporal drift. R² is stable or slightly improved on test. The `base_dev_pmcesd` model shows the best test performance overall, with R² improving from 0.270 to 0.376 (a 39% relative improvement) by adding `person_mean_cesd`.

### 6.5 Key findings — feature ablation

1. **`prior_cesd` alone is a strong baseline** (MAE = 4.17, R² = 0.303). The prior symptom score carries substantial predictive signal for the magnitude of symptom change.

2. **Adding 20 behavioral features does not improve over `prior_cesd` alone** (MAE: 4.17 → 4.24; R² drops from 0.303 to 0.290). This parallels the classification result: behavioral features without a person-level anchor add no incremental value.

3. **Within-person deviation features add no further improvement** (`base_dev` MAE = 4.30 vs. `base` MAE = 4.24). The deviation features may introduce noise that degrades predictions in the linear model framework.

4. **`person_mean_cesd` is the single most impactful feature addition** — adding it transforms performance:
   - MAE: 4.30 → **3.80** (−0.50 CES-D points, 12% reduction)
   - RMSE: 6.32 → **5.85** (−0.47, 7% reduction)
   - R²: 0.270 → **0.376** (+0.106, 39% relative increase)
   - Within-person R²: 0.211 → **0.389** (+0.178, 85% relative increase)
   - Median person MAE: 3.51 → **2.61** (−0.90, 26% reduction)

   This mirrors the classification result where `person_mean_cesd` was also the dominant additive feature (+0.031 AUC for XGBoost). The mechanism is the same: the trait anchor allows the model to distinguish between a person acutely elevated (expected to revert) and chronically elevated (different change trajectory).

---

## 7. Baseline Comparison

### 7.1 Best model (`base_dev_pmcesd`) vs. all baselines — test set

| Method | MAE | RMSE | R² | W-R² (med) |
|---|---|---|---|---|
| **MixedLM (base_dev_pmcesd)** | **3.80** | **5.85** | **0.376** | **0.389** |
| B0: No Change | 4.60 | 7.42 | −0.006 | −0.044 |
| B1: Population Mean | 4.62 | 7.41 | −0.003 | −0.049 |
| B2: Last Value Carried Forward | 7.03 | 11.13 | −1.265 | −0.649 |
| B3: Person-Specific Mean | 4.77 | 7.52 | −0.032 | −0.080 |
| B4: Regression to Mean | 4.68 | 7.45 | −0.015 | −0.074 |

### 7.2 Improvement over baselines

| Comparison | Δ MAE | Δ MAE (%) | Δ R² |
|---|---|---|---|
| vs. B0 (No Change) | −0.80 | −17.4% | +0.382 |
| vs. B1 (Population Mean) | −0.82 | −17.7% | +0.379 |
| vs. B3 (Person-Specific Mean) | −0.97 | −20.3% | +0.408 |
| vs. B4 (Regression to Mean) | −0.88 | −18.9% | +0.391 |

All baselines have negative R² on the test set (they explain less variance than a horizontal line at the mean). The MixedLM substantially outperforms all baselines, with the largest improvement over the person-specific mean baseline (−0.97 MAE, +0.408 R²).

**B2 (Last Value Carried Forward) is the worst baseline** (MAE = 7.03, R² = −1.265), confirming that CES-D delta has near-zero autocorrelation — the last observed change is a poor predictor of the next change.

---

## 8. Random Effects Sweep: Optimizing Person-Level Structure

Beyond the random-intercept-only models added and ran 9 random effects models as detailed below.

### 8.1 All 9 models — test set

| # | Model | Random Effects | MAE | RMSE | R² | W-R² (med) |
|---|---|---|---|---|---|---|
| 1 | Pooled (no PID) | None | 4.71 | 6.92 | 0.125 | — |
| 2 | Intercept only | Intercept | 4.24 | 6.23 | 0.290 | 0.211 |
| 3 | + prior_cesd slope | Intercept + prior_cesd | 4.23 | 6.25 | 0.285 | 0.203 |
| 4 | + prior + switches | Intercept + prior_cesd + switches | 4.24 | 6.19 | 0.300 | 0.133 |
| 5 | + prior + social ratio | Intercept + prior_cesd + social_ratio | 4.24 | 6.37 | 0.259 | 0.129 |
| 6 | + prior + soc extended | Intercept + prior_cesd + soc_screens + ratio | 4.24 | 6.27 | 0.282 | 0.198 |
| 7 | switches + screens | Intercept + switches + screens | 4.30 | 6.19 | 0.299 | 0.193 |
| **8** | **prior + sw + scr** | **Intercept + prior_cesd + switches + screens** | **4.24** | **6.16** | **0.307** | 0.143 |
| 9 | + dev features | Intercept + prior_cesd (29 features) | 4.31 | 6.36 | 0.261 | 0.194 |

### 8.2 Key findings — random effects sweep

1. **Person-level grouping matters enormously**: The pooled model (no person effects, #1) achieves R² = 0.125 vs. the intercept-only model (#2) at R² = 0.290 — a +0.165 improvement from simply adding a random intercept per person.

2. **Best random-effects model by RMSE/R²**: Model #8 (prior_cesd + switches + screens as random slopes) achieves the lowest RMSE (6.16) and highest R² (0.307) among the sweep models.

3. **Random slopes provide marginal improvement over intercept-only**: The best sweep model (#8, R² = 0.307) improves over intercept-only (#2, R² = 0.290) by only +0.017 R². Most of the person-level value is captured by the random intercept alone.

4. **Adding within-person deviation features (#9) does not help**: Model #9 (29 features with dev + random prior_cesd slope) performs worse than the base 21-feature intercept-only model (R² = 0.261 vs. 0.290).

5. **All sweep models are surpassed by `base_dev_pmcesd`**: The best sweep model (#8, R² = 0.307) is substantially below `base_dev_pmcesd` (R² = 0.376). `person_mean_cesd` as a fixed effect provides more predictive value than any random slope configuration.

---

## 9. Fixed Effects Coefficients — Best Model (`base_dev_pmcesd`)

The two dominant fixed effects, with all other features held constant:

| Feature | Coefficient | Std. Error | z | p-value | Interpretation |
|---|---|---|---|---|---|
| `prior_cesd` | **−0.805** | 0.028 | −28.66 | <0.001 | Strong regression to the mean: higher prior CES-D predicts larger decrease (improvement) |
| `person_mean_cesd` | **+0.794** | 0.033 | 24.28 | <0.001 | Chronic severity anchor: higher trait CES-D predicts worsening tendency |
| `mean_daily_screens_delta` | −0.656 | 0.252 | −2.60 | 0.009 | Increasing screen sessions predicts improvement |
| `dev_2` | +1.961 | 0.797 | 2.46 | 0.014 | Above-average deviation in behavioral feature predicts worsening |
| `dev_0` | +1.443 | 0.692 | 2.08 | 0.037 | Above-average deviation predicts worsening |
| `dev_1` | −2.004 | 0.956 | −2.10 | 0.036 | Below-average deviation predicts worsening |

### 9.1 Interpreting the dominant features

The two CES-D features (`prior_cesd` = −0.805, `person_mean_cesd` = +0.794) together implement a regression-to-the-mean mechanism anchored by the person's chronic level:

- **If prior_cesd = person_mean_cesd** (person at their typical level): the two effects roughly cancel, predicting near-zero change — symptoms stay stable.
- **If prior_cesd > person_mean_cesd** (person acutely elevated above their norm): the negative `prior_cesd` coefficient dominates, predicting improvement (return toward baseline).
- **If prior_cesd < person_mean_cesd** (person currently below their chronic level): the positive `person_mean_cesd` coefficient creates an upward pull, predicting worsening (return toward chronic level).

This is a **person-specific regression-to-the-mean** model: each person reverts toward their own chronic CES-D level, not the population mean. The model's primary mechanism is recognizing *how far the person is from their own baseline*, which determines both the direction and magnitude of predicted change.

### 9.2 Behavioral features

Only `mean_daily_screens_delta` reaches significance (p = 0.009) among the 21 base behavioral features. Three within-person deviation features also reach significance (p < 0.04). This pattern — strong clinical features, weak behavioral features — mirrors the classification result where behavioral features added minimal incremental AUC (+0.004) above `prior_cesd` alone.

**Random intercept variance = 0.000**: The random intercept is estimated at zero. This means `person_mean_cesd` as a fixed effect has fully absorbed the person-level variation that the random intercept would otherwise capture. This is expected: when the model knows each person's chronic CES-D level, there is no residual between-person variation to absorb.

---

## 10. Post-Hoc Direction Analysis

Classification labels from `classification/labels/sev_crossing/` are used to evaluate whether the regression model's continuous predictions can recover the *direction* of symptom change. Regression predictions are converted to direction predictions using the same clinical severity thresholds used to create the classification labels.

### 10.1 Direction classification metrics — all models, test set (sev_crossing)

| Model | BalAcc | AUC (OvR) | Sens-W | PPV-W | F1 macro |
|---|---|---|---|---|---|
| 1_pooled | 0.458 | 0.582 | 0.486 | 0.070 | 0.172 |
| 2_intercept (= base) | 0.515 | 0.704 | 0.703 | 0.114 | 0.174 |
| 3_prior_slope | 0.506 | 0.700 | 0.676 | 0.110 | 0.172 |
| 4_prior_switches | 0.513 | 0.704 | 0.676 | 0.108 | 0.176 |
| 5_prior_social | 0.506 | 0.703 | 0.676 | 0.111 | 0.171 |
| 6_prior_soc_ext | 0.506 | 0.701 | 0.676 | 0.111 | 0.171 |
| 7_sw_screens | 0.488 | 0.694 | 0.622 | 0.100 | 0.167 |
| 8_prior_sw_scr | 0.506 | 0.702 | 0.676 | 0.109 | 0.172 |
| 9_dev_features (= base_dev) | 0.498 | 0.694 | 0.676 | 0.110 | 0.169 |
| prior_cesd | 0.516 | 0.704 | 0.730 | 0.108 | 0.180 |
| **base_dev_pmcesd** | **0.542** | **0.756** | **0.784** | **0.145** | **0.178** |

### 10.2 Best regression model vs. classification models (sev_crossing, test set)

| Model Type | Model | AUC (OvR) | BalAcc | Sens-W | PPV-W |
|---|---|---|---|---|---|
| **Classification** | XGBoost (39 feat) | **0.906** | **0.834** | **0.838** | **0.356** |
| Classification | LightGBM (39 feat) | 0.901 | 0.842 | 0.865 | 0.344 |
| Classification | ElasticNet (39 feat) | 0.829 | 0.691 | 0.730 | 0.248 |
| Classification | SVM (39 feat) | 0.841 | 0.696 | 0.649 | 0.304 |
| **Regression** | MixedLM (base_dev_pmcesd) | 0.756 | 0.542 | 0.784 | 0.145 |
| Regression | MixedLM (prior_cesd) | 0.704 | 0.516 | 0.730 | 0.108 |
| Regression | MixedLM (base, intercept) | 0.704 | 0.515 | 0.703 | 0.114 |

**Key finding**: The dedicated classifiers substantially outperform regression-based direction prediction. The best regression model (`base_dev_pmcesd`) achieves AUC = 0.756 and BalAcc = 0.542, compared to XGBoost classification at AUC = 0.906 and BalAcc = 0.834. The regression model has reasonable worsening sensitivity (Sens-W = 0.784) but very low precision (PPV-W = 0.145 — roughly 1 true positive per 6 alarms), confirming that optimizing MAE is the wrong objective for a directional task (as discussed in the classification results, Section 3.1).

### 10.3 Stratified regression error by direction class — test set

Stratifying regression error by the true direction of change (from `sev_crossing` labels) reveals systematic bias:

#### Best model: `base_dev_pmcesd`

| Direction | N | MAE | RMSE | Bias |
|---|---|---|---|---|
| Improving | 44 | 8.65 | 10.41 | **+8.48** |
| Stable | 330 | 2.59 | 3.80 | +0.16 |
| Worsening | 37 | 8.82 | 11.03 | **−8.36** |

#### Intercept-only model: `base` (21 features)

| Direction | N | MAE | RMSE | Bias |
|---|---|---|---|---|
| Improving | 44 | 8.22 | 10.07 | **+8.21** |
| Stable | 330 | 3.21 | 4.60 | +0.67 |
| Worsening | 37 | 9.32 | 11.59 | **−8.93** |

#### Prior CES-D only

| Direction | N | MAE | RMSE | Bias |
|---|---|---|---|---|
| Improving | 44 | 8.54 | 10.29 | **+8.45** |
| Stable | 330 | 3.00 | 4.26 | +0.76 |
| Worsening | 37 | 9.34 | 11.64 | **−8.79** |

#### Pooled model (no person effects)

| Direction | N | MAE | RMSE | Bias |
|---|---|---|---|---|
| Improving | 44 | 10.82 | 12.39 | **+10.67** |
| Stable | 330 | 3.08 | 4.11 | +0.61 |
| Worsening | 37 | 11.94 | 14.10 | **−11.94** |

### 10.4 Key findings — stratified error

1. **All models show large, symmetric directional bias**: Improving cases have large positive bias (+8 to +11 CES-D points — model under-predicts improvement), worsening cases have large negative bias (−8 to −12 — model under-predicts worsening). This is the hallmark of a regression model predicting near-zero change for everyone, which is the optimal MAE strategy when the majority class (stable, 80%) dominates.

2. **Stable cases are predicted well**: MAE = 2.59–3.21 for stable cases across models, with near-zero bias. This confirms the model is highly accurate for the majority class.

3. **`base_dev_pmcesd` slightly improves directional prediction**: Compared to the base intercept-only model, `base_dev_pmcesd` reduces stable-class MAE (3.21 → 2.59) and slightly reduces worsening bias (−8.93 → −8.36), consistent with better calibration at the tails.

4. **The pooled model is worst for directional prediction**: Without person-level effects, directional bias reaches ±11–12 CES-D points — the model cannot distinguish individual trajectories at all.

5. **This pattern motivates the classifier-first approach**: A regression model trained on MAE will always produce near-zero predictions for the majority of observations, making directional classification fundamentally difficult. The classifiers, trained directly on class boundaries, avoid this collapse.


---

## 11. Comparison: Regression vs. Classification

### 11.1 The complementary roles of regression and classification

| Dimension | MixedLM Regression | XGBoost Classification |
|---|---|---|
| **Target** | Continuous CES-D delta (magnitude) | Categorical direction (improving/stable/worsening) |
| **Best test metric** | MAE = 3.80, R² = 0.376 | AUC = 0.906, BalAcc = 0.834 |
| **Directional accuracy** | BalAcc = 0.542 (post-hoc) | BalAcc = 0.834 (direct) |
| **Worsening sensitivity** | Sens-W = 0.784 (post-hoc) | Sens-W = 0.838 |
| **Worsening precision** | PPV-W = 0.145 | PPV-W = 0.356 |
| **Strength** | Calibrated magnitude predictions for stable cases | Strong directional discrimination, especially at tails |
| **Weakness** | Systematic under-prediction at tails (±8 CES-D pt bias) | No continuous magnitude estimate |
| **`person_mean_cesd` effect** | +0.106 R² (0.270 → 0.376) | +0.031 AUC (0.875 → 0.906) |

### 11.2 When to use which

- **For clinical alerting** (detecting worsening episodes): Use the classifier. It achieves 2.5× higher directional precision (PPV-W = 0.356 vs. 0.145) and substantially better balanced accuracy (0.834 vs. 0.542).

- **For understanding symptom dynamics** (how much change to expect): Use the regression model. It provides calibrated continuous predictions that are useful for effect size estimation, individual trajectory forecasting, and understanding person-level change dynamics.

- **For mechanistic insight**: Both models converge on the same key finding — `person_mean_cesd` is the dominant additive feature, and behavioral features alone add minimal signal without a severity anchor. The regression model's coefficients (Section 9) provide direct interpretability of the person-specific regression-to-the-mean mechanism.

### 11.3 Convergent findings across regression and classification

1. **`person_mean_cesd` is the key feature addition in both frameworks**: +0.106 R² for regression, +0.031 AUC for classification. Both show the largest gains from adding this single trait-level feature.

2. **Behavioral features alone add minimal signal**: In regression, adding 20 behavioral features to `prior_cesd` alone actually *decreases* performance (R²: 0.303 → 0.290). In classification, behavioral features add +0.004 AUC (n.s.) over `prior_cesd` alone.

3. **Person-level structure is essential**: The pooled regression model (R² = 0.125) performs far worse than the mixed-effects model (R² = 0.290). In classification, cold-start performance drops from AUC = 0.906 to 0.821.

4. **Within-person deviation features provide limited additional value**: In regression, `base_dev` performs similarly to or slightly worse than `base`. In classification, behavioral lag features show no significant improvement over base features alone.

---

## 12. Per-Person Analysis

### 12.1 Median person-level MAE across models (test set)

| Condition | Median Person MAE | Q1 | Q3 | IQR |
|---|---|---|---|---|
| `prior_cesd` | 3.09 | 1.67 | 5.42 | 3.75 |
| `base` | 3.07 | 1.80 | 5.54 | 3.75 |
| `base_dev` | 3.51 | 1.79 | 5.63 | 3.84 |
| **`base_dev_pmcesd`** | **2.61** | **1.46** | **5.29** | **3.83** |

The best model achieves a median per-person MAE of 2.61 CES-D points — meaning half of participants have average prediction errors below 2.61 points. The interquartile range (1.46–5.29) shows substantial between-person variability in prediction accuracy, as expected given individual differences in symptom trajectory predictability.

---

## 13. Model Diagnostics and Convergence

### 13.1 Convergence across models

| Condition | Optimizer Used | Attempts | Converged |
|---|---|---|---|
| `prior_cesd` | lbfgs | 1 | Yes |
| `base` | lbfgs | 1 | Yes |
| `base_dev` | lbfgs | 1 | Yes |
| `base_dev_pmcesd` | powell (fallback) | 3 | Yes |

The `base_dev_pmcesd` model required three optimization attempts — `lbfgs` and `bfgs` failed to converge, but `powell` succeeded. This is expected for the 30-feature model with `person_mean_cesd`, which creates a near-collinearity with the random intercept (both capture between-person variation). The final model converged successfully and produced stable predictions.

### 13.2 Random intercept variance

| Condition | Group Variance | Interpretation |
|---|---|---|
| `prior_cesd` | > 0 | Random intercept absorbs some between-person variation |
| `base` | > 0 | Same |
| `base_dev` | > 0 | Same |
| `base_dev_pmcesd` | **0.000** | `person_mean_cesd` fully absorbs between-person variation |

The zero random intercept variance in `base_dev_pmcesd` is not a convergence failure — it is the correct statistical result. When `person_mean_cesd` enters as a fixed effect, it captures all the between-person variation in CES-D delta that the random intercept would otherwise absorb. This means the mixed model with `person_mean_cesd` effectively reduces to a fixed-effects-only model with person-specific intercepts via the `person_mean_cesd` coefficient.

---

## 14. Data and Software

| Item | Detail |
|---|---|
| Sample | N = 96 participants, smartphone behavioral data |
| Observation unit | Biweekly observation period |
| Train / Val / Test | 1,196 / 395 / 411 observations (all 96 persons in each split) |
| CES-D instrument | CES-D-20 (range 0–60) |
| Software | `statsmodels.MixedLM` (Python) |
| Estimation | REML |

### 14.1 Reproducibility

```bash
# Train all four ablation conditions
python regression/mixedlm/scripts/train_mixedlm.py

# Run the full 9-model random effects sweep
python regression/mixedlm/scripts/train_mixedlm.py --full-sweep

# Run post-hoc direction analysis
python regression/mixedlm/scripts/posthoc_mixedlm.py
```

### 14.2 File index

#### Models and predictions

| File | Contents |
|---|---|
| `regression/mixedlm/models/<condition>/model.pkl` | Pickled fitted model |
| `regression/mixedlm/models/<condition>/y_pred_{train,val,test}.npy` | Predictions per split |
| `regression/mixedlm/models/<condition>/random_effects.csv` | Per-person random intercepts/slopes |
| `regression/mixedlm/models/<condition>/convergence_info.json` | Optimizer attempts and convergence status |
| `regression/mixedlm/models/<condition>/model_summary.txt` | Full statsmodels summary |
| `regression/mixedlm/models/<condition>/training_results.json` | Model config, feature names, fit metrics |
| `regression/mixedlm/models/<condition>/{train,val,test}_aggregate_comparison.csv` | Model vs. 5 baselines (MAE, RMSE, R², W-R², B-R²) |

#### Post-hoc direction analysis

| File | Contents |
|---|---|
| `regression/mixedlm/reports/posthoc/all_posthoc_results.csv` | Combined classification metrics (all models × splits) |
| `regression/mixedlm/reports/posthoc/posthoc_summary.md` | Markdown summary table |
| `regression/mixedlm/reports/posthoc/<condition>/posthoc_classification_sev_crossing.csv` | BalAcc, AUC, Sens-W, PPV-W per split |
| `regression/mixedlm/reports/posthoc/<condition>/{train,val,test}_sev_crossing_stratified_error.csv` | MAE/RMSE/Bias by direction class |
| `regression/mixedlm/reports/posthoc/<condition>/{train,val,test}_sev_crossing_confusion_matrix.png` | 3×3 direction confusion matrix |
| `regression/mixedlm/reports/posthoc/<condition>/test_pred_vs_actual.png` | Scatter with identity line |
| `regression/mixedlm/reports/posthoc/<condition>/test_residual_vs_predicted.png` | Residual diagnostics |
| `regression/mixedlm/reports/posthoc/<condition>/test_person_trajectories.png` | Per-person actual vs. predicted over time |

#### Scripts

| Script | Purpose |
|---|---|
| `regression/mixedlm/scripts/train_mixedlm.py` | Train MixedLM across feature ablation conditions and random effects sweep |
| `regression/mixedlm/scripts/posthoc_mixedlm.py` | Post-hoc direction analysis using classification labels |
| `regression/mixedlm/scripts/model.py` | `MixedLMModel` class — fit, predict, convergence diagnostics |
| `regression/mixedlm/scripts/metrics.py` | Evaluation utilities — aggregate metrics, baselines, direction classification |

---

## 15. Recommended Presentation Strategy

### For a paper/manuscript

1. **Main text Table**: Feature ablation results (Section 6.1) — the four conditions with MAE, RMSE, R², W-R²
2. **Main text Table**: Best model vs. baselines (Section 7.1)
3. **Main text paragraph**: `person_mean_cesd` as dominant feature (+0.106 R²), paralleling classification finding
4. **Main text paragraph**: Regression vs. classification comparison (Section 11.1) — motivating the complementary approach
5. **Supplementary Table S-R1**: Random effects sweep — all 9 models (Section 8.1)
6. **Supplementary Table S-R2**: Fixed effects coefficients (Section 9)
7. **Supplementary Table S-R3**: Stratified regression error by direction class (Section 10.3)
8. **Supplementary Table S-R4**: Regression-as-classifier metrics vs. dedicated classifiers (Section 10.2)
9. **Supplementary Table S-R5**: Val/test generalization comparison (Section 6.4)
10. **Supplementary Table S-R6**: Train/val/test metrics across all ablation conditions (Sections 6.1–6.3)

### For a talk/presentation

- Lead with the feature ablation table showing `person_mean_cesd` as the key feature
- Show the stratified error to explain why regression under-predicts at tails
- End with the head-to-head comparison with classifiers to motivate the complementary approach

### Key narrative points for the paper

1. **MixedLM achieves clinically meaningful prediction accuracy** (MAE = 3.80 CES-D points, R² = 0.376) when equipped with the person-level trait anchor
2. **`person_mean_cesd` is the dominant additive feature** — consistent across both regression (+0.106 R²) and classification (+0.031 AUC) frameworks
3. **Behavioral features alone add no incremental value** — paralleling the classification finding. The signal is in symptom history, not behavioral phenotype
4. **Person-specific regression-to-the-mean is the primary mechanism** — the model predicts each person will revert toward their chronic CES-D level, with the magnitude of predicted change proportional to their current displacement from baseline
5. **For clinical alerting, classification substantially outperforms regression-as-classifier** — optimizing MAE produces near-zero predictions that collapse directional discrimination. Dedicated classifiers trained on clinical boundaries are the appropriate tool for worsening detection
