# ElasticNet Regression Results Summary

## 1. Task Definition

The outcome for our regression is the **continuous CES-D delta** -- the change in CES-D score from one biweekly assessment to the next. Positive values indicate worsening; negative values indicate improvement. The range is -40 to +40 CES-D points.

We have N = 96 participants contributing 2,002 total observations. The data are split into 1,196 training, 395 validation, and 411 test observations using a within-person temporal split: for each participant, the earliest ~60% of their survey periods form the training set, the next ~20% form validation, and the final ~20% form the test set. All 96 participants appear in every split. This preserves the temporal structure of the data: the model is always predicting forward in time, never seeing future observations during training or hyperparameter selection.

CES-D delta has near-zero between-person ICC (ICC ~ 0; person means explain only 0.4% of delta variance), meaning no person consistently changes more than another. The prediction target is inherently within-person.

**Relationship to classification task:** The classification task predicts the *direction* of change (improving / stable / worsening). This regression task predicts the continuous change in CES-D score, that is, how much a person's depression score changes from one biweekly assessment to the next, not just which direction. The regression predictions can be post-hoc converted to direction predictions for direct comparison with classifiers (see Section 10).

---

## 2. Model: ElasticNet Regression

### 2.1 Model specification

`sklearn.linear_model.ElasticNet` with L1+L2 regularization.

```
y_delta ~ X * beta, with penalty = alpha * [l1_ratio * ||beta||_1 + 0.5 * (1 - l1_ratio) * ||beta||_2^2]
```

**Table 2.1: ElasticNet hyperparameter configuration**

| Setting | Value |
|---|---|
| Regularization | L1 + L2 (ElasticNet) |
| Hyperparameter grid | 13 alphas (0.0001-100) x 6 l1_ratios (0.1-0.99) = 78 combinations |
| Selection criterion | Minimum validation MAE |
| Max iterations | 10,000 |

The selection criterion is minimum validation MAE because MAE is directly interpretable on the CES-D scale (i.e., "the model is off by X points on average") and is robust to the heavy-tailed distribution of CES-D delta scores, where a few extreme changes could disproportionately influence RMSE-based selection.

### 2.2 Why ElasticNet

ElasticNet serves as a **global pooled model (M1)** -- a single model trained across all participants, assuming universal behavior-symptom relationships. All coefficients are shared across individuals. This represents the standard population-level approach when individual data is limited or when assuming behavior-symptom relationships are universal. It establishes a strong baseline for assessing the value of personalization.

ElasticNet regularization (mixing L1 and L2 penalties) is well-suited to our feature space for two reasons:
- **L1 penalty (Lasso)** performs automatic feature selection, zeroing out irrelevant predictors -- important given our high-dimensional feature space (up to 130 features across conditions)
- **L2 penalty (Ridge)** handles multicollinearity among behavioral features (e.g., daily screens and daily switches are correlated)

The mixing parameter `l1_ratio` tunes the balance between sparsity and stability, and is tuned alongside the regularization strength `alpha` via grid search.

This contrasts with the **personalized mixed-effects model (M2, MixedLM)**, which allows behavior-symptom relationships to vary across individuals via random intercepts and random slopes. The MixedLM provides automatic regularization through partial pooling: person-specific estimates are shrunk toward population means, preventing overfitting for individuals with limited data while allowing genuine individual differences to emerge. Together, M1 and M2 form a minimal contrast for testing whether personalization improves prediction of change in CES-D score.

### 2.3 Training protocol

The pipeline follows a strict 3-phase protocol to prevent data leakage:

| Phase | Data | Purpose |
|---|---|---|
| 1. Grid search | Fit on Train, evaluate on Val | Select best (alpha, l1_ratio) by min Val MAE |
| 2. Dev model | Fit on Train with locked params | Generate Train + Val predictions; Val metrics are the valid development-phase numbers |
| 3. Final model | Refit on Train+Val with locked params | Generate Test predictions; Test metrics are the final unbiased generalization estimate |

Hyperparameters are **locked** after Phase 1. The Train+Val combination in Phase 3 uses the same locked hyperparameters -- no decisions are made based on Test.

---

## 3. Features

### 3.1 Feature ablation conditions

Eleven feature-set conditions: 4 matching the classification task and 7 from the original screenome feature-engineering variants.

> **MERVE AND ANDREA, HELP:** The classification task uses 4 cumulative feature-set conditions (prior_cesd only, base behavioral, base + lag, base + lag + person_mean_cesd). The ElasticNet regression runs those same 4 plus 7 additional conditions that combine the original screenome-derived feature groups (within-person deviations, phenotype clusters, participant ID encoding). I currently label these as "required" vs. "extra" in tables and code, but this framing feels wrong: "required" implies necessity rather than "matching the classification task for comparability," and "extra" is disingenuous feature groups we (specifically, Merve) deliberately engineered. What should we call these???

| Condition | Features | N feat | N retained | Group |
|---|---|---|---|---|
| `prior_cesd` | prior_cesd only | 1 | 1 | required |
| `base` | All 21 base features | 21 | 1 | required |
| `base_lag` | Base + 17 behavioral lag features | 38 | 1 | required |
| `base_lag_pmcesd` | Base + lag + person_mean_cesd | 39 | 2 | required |
| `dev` | Base + 8 within-person deviation | 29 | 1 | extra |
| `pheno` | Base + 5 phenotype | 26 | 1 | extra |
| `pid` | Base + ~96 PID one-hot | 117 | 101 | extra |
| `dev_pheno` | Base + dev + pheno | 34 | 1 | extra |
| `dev_pid` | Base + dev + PID OHE | 125 | 107 | extra |
| `pheno_pid` | Base + pheno + PID OHE | 122 | 120 | extra |
| `dev_pheno_pid` | All feature types | 130 | 111 | extra |

Six conditions (prior_cesd, base, base_lag, dev, pheno, dev_pheno) collapse to identical models -- ElasticNet zeros out all features except `prior_cesd`. Without person-level information (person_mean_cesd or PID encoding), no behavioral feature survives regularization.

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

> `_delta` features are changes relative to the person's own prior period -- they encode within-person deviation from baseline.

### 3.3 Lag features (17)

For each of the 17 time-varying behavioral features, we compute the lag-1 value (previous period's value for the same person). Static demographics (age, gender_mode_1, gender_mode_2) are excluded since they are constant across periods. Clinical lags (lag_prior_cesd, lag_cesd_delta) are excluded per ablation design (see DATA_README.md). Missing lags (first observation per person) are filled with 0.

### 3.4 Person-level trait: `person_mean_cesd`

```
person_mean_cesd_i = mean(prior_cesd over all training periods for person i)
```

To ensure no data leakage, `person_mean_cesd` is computed from each person's training-period observations only. The resulting value is then assigned to all of that person's validation and test rows via participant ID lookup, so that no validation or test CES-D values ever enter the computation.

`prior_cesd` encodes where a person is *right now* (state), while `person_mean_cesd` encodes their chronic level (trait). This distinction is critical for regression: a person with prior_cesd = 20 and person_mean_cesd = 8 (elevated above their normal cesd) has different expected change dynamics than a person with prior_cesd = 20 and person_mean_cesd = 22 (near their moderate baseline).

---

## 4. Metrics

All metrics are reported on train, val, and test splits.

| Metric | Description | Why it matters |
|---|---|---|
| MAE | Mean absolute error (CES-D points) | Interpretable clinical scale -- average prediction error in CES-D points |
| RMSE | Root mean squared error | Penalizes large errors, clinically important for detecting extreme changes |
| R² | Overall variance explained | Standard goodness-of-fit measure |
| Within-person R² (median) | **Primary R² metric.** Median of per-person R² values | CES-D delta has near-zero between-person ICC (ICC ~ 0), so within-person R² guards against inflated estimates. Uses median (not mean) because persons with very few test observations can produce extreme negative R² that distorts the mean. Persons with < 2 observations are excluded. |
| Between-person R² | Variance explained between person-level means | Reported for completeness; expected to be low or negative |

---

## 5. Baselines (B0-B4)

Five baselines are computed on the training set and applied identically to val and test:

| Baseline | Description | Rationale |
|---|---|---|
| B0: No Change | Predict 0 (no change) for everyone | Lower bound -- assumes symptoms never change |
| B1: Population Mean | Predict the training-set mean delta | Assumes everyone changes by the average amount |
| B2: Last Value Carried Forward | Each person's last training delta | Tests whether momentum/inertia from last observation helps |
| B3: Person-Specific Mean | Each person's mean training delta | Best available person-specific naive forecast |
| B4: Regression to Mean | Shrunk person mean (shrinkage = 0.5) | Balances person-specific signal with population average |

---

## 6. Primary Results: Feature Ablation

### 6.1 Test set -- required conditions

| Condition | N feat | Retained | MAE | RMSE | R² | W-R² (med) |
|---|---|---|---|---|---|---|
| `prior_cesd` | 1 | 1 | 4.58 | 7.39 | -0.006 | -0.042 |
| `base` | 21 | 1 | 4.58 | 7.39 | -0.006 | -0.042 |
| `base_lag` | 38 | 1 | 4.58 | 7.39 | -0.006 | -0.042 |
| **`base_lag_pmcesd`** | **39** | **2** | **4.13** | **6.28** | **0.279** | **0.186** |

The first three conditions collapse to an identical prior_cesd-only model. ElasticNet zeros out all 20 behavioral features and all 17 lag features.

### 6.2 Validation set -- required conditions

| Condition | N feat | MAE | RMSE | R² | W-R² (med) |
|---|---|---|---|---|---|
| collapsed (prior_cesd / base / base_lag) | 1 | 4.29 | 6.57 | -0.001 | -0.018 |
| **`base_lag_pmcesd`** | **39** | **3.92** | **5.73** | **0.249** | **0.239** |

### 6.3 Training set -- required conditions

| Condition | N feat | MAE | RMSE | R² | W-R² (med) |
|---|---|---|---|---|---|
| collapsed (prior_cesd / base / base_lag) | 1 | 4.65 | 7.34 | 0.000 | 0.019 |
| **`base_lag_pmcesd`** | **39** | **3.90** | **6.00** | **0.348** | **0.336** |

### 6.4 Val -> Test generalization

| Condition | Val MAE | Test MAE | Delta MAE | Val W-R² | Test W-R² | Delta W-R² |
|---|---|---|---|---|---|---|
| collapsed | 4.29 | 4.58 | +0.29 | -0.018 | -0.042 | -0.024 |
| `base_lag_pmcesd` | 3.92 | 4.13 | +0.21 | 0.239 | 0.186 | -0.053 |

MAE increases modestly from val to test (~0.2-0.3 CES-D points), consistent with mild temporal drift. The `base_lag_pmcesd` model generalizes slightly better in MAE terms (Delta = +0.21 vs +0.29) but shows some W-R² degradation (-0.053). This degradation may reflect that the relationship between a person's chronic CES-D level and their predicted change is slightly less stable in later survey periods -- for instance, if some participants' depression trajectories shift over time (improving or worsening trends), the training-set-derived `person_mean_cesd` becomes a less accurate anchor for their current dynamics.

---

## 7. All 11 Conditions -- Cross-Condition Ranking

Sorted by Val MAE (selection criterion). Conditions marked with `*` are required (parity with classification).

| Condition | Group | N feat | Retained | Train MAE | Val MAE | Test MAE | Test W-R² (med) | Test AUC | Test BalAcc |
|---|---|---|---|---|---|---|---|---|---|
| pheno_pid | extra | 122 | 120 | 3.78 | 3.92 | **4.06** | **0.219** | 0.723 | 0.579 |
| base_lag_pmcesd* | req | 39 | 2 | 3.90 | **3.92** | 4.13 | 0.186 | 0.722 | 0.540 |
| pid | extra | 117 | 101 | 3.78 | 3.93 | 4.09 | 0.204 | 0.715 | 0.602 |
| dev_pid | extra | 125 | 107 | 3.78 | 3.95 | 4.10 | 0.185 | 0.715 | 0.603 |
| dev_pheno_pid | extra | 130 | 111 | 3.77 | 3.97 | 4.10 | 0.191 | 0.717 | 0.593 |
| prior_cesd* | req | 1 | 1 | 4.65 | 4.29 | 4.58 | -0.042 | 0.678 | 0.403 |
| base* | req | 21 | 1 | 4.65 | 4.29 | 4.58 | -0.042 | 0.678 | 0.403 |
| base_lag* | req | 38 | 1 | 4.65 | 4.29 | 4.58 | -0.042 | 0.678 | 0.403 |
| dev | extra | 29 | 1 | 4.65 | 4.29 | 4.58 | -0.042 | 0.678 | 0.403 |
| pheno | extra | 26 | 1 | 4.65 | 4.29 | 4.58 | -0.042 | 0.678 | 0.403 |
| dev_pheno | extra | 34 | 1 | 4.65 | 4.29 | 4.58 | -0.042 | 0.678 | 0.403 |

Direction metrics are from posthoc sev_crossing analysis. AUC = OvR macro, BalAcc = macro recall over 3 classes (improving / stable / worsening).

**Key findings across all 11 feature-set conditions:**

1. **Six of 11 feature-set conditions collapse to an identical prior_cesd-only model.** ElasticNet zeros out all behavioral features, all lag features, all deviation features, and all phenotype features. Without person-level information, no behavioral feature survives regularization. This is visible in the table above: prior_cesd, base, base_lag, dev, pheno, and dev_pheno all produce identical metrics.

2. **`person_mean_cesd` is the single most impactful feature addition.** Comparing the collapsed prior_cesd-only model to `base_lag_pmcesd` in Section 6.1: MAE drops from 4.58 to 4.13 (-0.45 CES-D points, -10%), R² jumps from -0.006 to 0.279 (+0.285), and W-R² goes from -0.042 to 0.186 (+0.228). No other feature addition produces a comparable improvement.

3. **The top 5 feature-set conditions all include PID one-hot encoding or person_mean_cesd** -- person-level information is critical for any improvement over the autoregressive baseline.

4. **`base_lag_pmcesd` wins on Val MAE despite using far fewer parameters** (2 retained features vs. 100+ for PID-based conditions). `pheno_pid` achieves the best Test MAE (4.06) and Test W-R² (0.219) by learning person-specific intercepts, but at the cost of 120 parameters for 96 persons.

5. **Val-to-test generalization is stable** across conditions. For the best model (`base_lag_pmcesd`), Delta MAE = +0.21 from val to test (see Section 6.4), indicating no severe overfitting despite the Train+Val refit in Phase 3.

---

## 8. Baseline Comparison

### 8.1 Best model (`base_lag_pmcesd`) vs. all baselines -- test set

| Method | MAE | RMSE | R² | W-R² (med) | Median Person MAE | Q1 MAE | Q3 MAE |
|---|---|---|---|---|---|---|---|
| **ElasticNet (base_lag_pmcesd)** | **4.13** | **6.28** | **0.279** | **0.186** | **3.18** | **1.62** | **5.01** |
| B0: No Change | 4.60 | 7.42 | -0.006 | -0.044 | 2.78 | 1.60 | 6.05 |
| B1: Population Mean | 4.62 | 7.41 | -0.003 | -0.049 | 2.83 | 1.69 | 6.08 |
| B2: Last Value Carried Forward | 7.03 | 11.13 | -1.265 | -0.649 | 5.00 | 2.24 | 9.53 |
| B3: Person-Specific Mean | 4.77 | 7.52 | -0.032 | -0.080 | 3.00 | 1.76 | 6.38 |
| B4: Regression to Mean | 4.68 | 7.45 | -0.015 | -0.074 | 3.00 | 1.75 | 6.40 |

### 8.2 Improvement over baselines

| Comparison | Delta MAE | Delta MAE (%) | Delta W-R² |
|---|---|---|---|
| vs. B0 (No Change) | -0.47 | -10.2% | +0.230 |
| vs. B1 (Population Mean) | -0.49 | -10.6% | +0.235 |
| vs. B3 (Person-Specific Mean) | -0.64 | -13.4% | +0.266 |
| vs. B4 (Regression to Mean) | -0.55 | -11.8% | +0.260 |

All baselines have negative R² and negative W-R² on the test set (they explain less variance than a horizontal line at the mean). The ElasticNet substantially outperforms all baselines, with the largest improvement over the person-specific mean baseline (-0.64 MAE, +0.266 W-R²).

**B2 (Last Value Carried Forward) is the worst baseline** (MAE = 7.03, W-R² = -0.649), confirming that CES-D delta has near-zero autocorrelation -- the last observed change is a poor predictor of the next change.

---

## 9. Model Coefficients and Interpretation

### 9.1 Best condition: `base_lag_pmcesd` (2 of 39 features retained)

Hyperparameters: alpha = 10.0, l1_ratio = 0.1 (strongly Ridge-dominant).

| Feature | Dev Coef | Final Coef | Interpretation |
|---|---|---|---|
| `prior_cesd` | -0.502 | -0.458 | Higher current CES-D predicts a *decrease* (regression toward mean) |
| `person_mean_cesd` | +0.447 | +0.418 | Higher personal average predicts an *increase* (pull back up) |

The two coefficients nearly cancel. The predicted delta approximates:

```
delta_hat = 0.45 * (person_mean_cesd - prior_cesd)
```

This is a **regression-to-the-mean machine**: it predicts that CES-D will move back toward each person's long-run average. No behavioral feature (screen time, app switching, social media use, lag features, etc.) contributes.

### 9.2 PID-based models

The PID-encoded models (pheno_pid, pid, dev_pid, dev_pheno_pid) learn **person-specific intercepts** via one-hot encoded participant IDs instead of modeling the regression-to-mean mechanism explicitly.

**pheno_pid (120 of 122 features retained):**

| Category | Example features | Coef range | Role |
|---|---|---|---|
| PID intercepts | pid_5343 (+18.2), pid_1257 (-10.3) | -10.3 to +18.2 | Person-specific baseline delta |
| Gender | gender_mode_1 (-4.5), gender_mode_2 (-3.1) | -4.5 to -3.1 | Group-level offset |
| Phenotype | pheno_1 (-3.0), pheno_2 (-1.7), pheno_0 (-0.8) | -3.0 to -0.5 | Latent subtype adjustment |
| Behavioral | active_day_ratio_delta (+3.0), overnight_ratio (+1.0) | -1.7 to +3.0 | Small behavioral signals |
| Prior CES-D | prior_cesd (-0.64) | -0.64 | Regression to mean (diminished vs base_lag_pmcesd) |

Behavioral features *do* enter the PID-based models, but with smaller coefficients than the PID intercepts. The largest behavioral effect (active_day_ratio_delta = +3.0) is dwarfed by the person intercepts (+/-10-18).

### 9.3 Summary

1. **Screenome features add negligible value in a pooled linear model**: CES-D history dominates entirely
2. **Person-level information is the key**: whether encoded explicitly (person_mean_cesd) or via PID dummies, knowing *who* the person is matters far more than *what they did on their phone*
3. **PID-based models risk overfitting**: 120 parameters for 96 persons means more free parameters than training subjects; Test MAE still improves suggesting person effects are real, but performance would likely degrade on a new sample of persons
4. **The regression-to-mean mechanism is dominant**: CES-D scores naturally fluctuate toward each person's average, and ElasticNet identifies this as the strongest predictive signal

---

## 10. Post-Hoc Direction Analysis

The regression model predicts a continuous CES-D delta, but clinically we often care about the *direction* of change: is a person improving, stable, or worsening? This post-hoc analysis converts the continuous regression predictions into three-class direction labels by applying the same labeling function used to generate the ground-truth classification labels (e.g., for `sev_crossing`, a predicted delta that would cross upward past a clinical severity threshold is labeled "worsening"). This lets us directly compare the regression model's ability to recover direction against the dedicated classifiers trained explicitly on those labels -- answering whether a single regression model can serve double duty, or whether separate classification models are needed for clinical alerting.

### 10.1 Direction classification metrics -- `base_lag_pmcesd`

#### sev_crossing (primary)

Clinical severity boundary crossing (CES-D thresholds at 16 and 24).

| Split | BalAcc | AUC (OvR) | Sens-W | PPV-W |
|---|---|---|---|---|
| Val | 0.571 | 0.732 | 0.200 | 0.636 |
| Test | 0.540 | 0.722 | 0.243 | 0.750 |

#### personal_sd (sensitivity analysis)

Person-specific SD-based thresholds (k = 1.0).

| Split | BalAcc | AUC (OvR) | Sens-W | PPV-W |
|---|---|---|---|---|
| Val | 0.374 | 0.693 | 0.000 | 0.000 |
| Test | 0.385 | 0.751 | 0.000 | 0.000 |

#### balanced_tercile (supplementary)

Rank-based equal thirds (bottom = improving, middle = stable, top = worsening).

| Split | BalAcc | AUC (OvR) | Sens-W | PPV-W |
|---|---|---|---|---|
| Val | 0.542 | 0.698 | 0.519 | 0.519 |
| Test | 0.518 | 0.684 | 0.518 | 0.518 |

### 10.2 Stratified regression error by direction class (test)

#### sev_crossing

| Direction | N | MAE | RMSE | Bias |
|---|---|---|---|---|
| Improving | 44 | 10.04 | 11.68 | +10.03 |
| Stable | 330 | 2.72 | 3.80 | +0.55 |
| Worsening | 37 | 9.66 | 12.14 | -9.48 |

#### personal_sd

| Direction | N | MAE | RMSE | Bias |
|---|---|---|---|---|
| Improving | 50 | 9.22 | 10.68 | +9.01 |
| Stable | 320 | 2.56 | 3.63 | +0.74 |
| Worsening | 41 | 10.17 | 12.41 | -10.17 |

#### balanced_tercile

| Direction | N | MAE | RMSE | Bias |
|---|---|---|---|---|
| Improving | 137 | 6.04 | 7.69 | +5.69 |
| Stable | 137 | 1.52 | 2.10 | +0.90 |
| Worsening | 137 | 4.82 | 7.41 | -4.61 |

The model predicts the **stable** class well (MAE 1.5-2.7, near-zero bias) but severely mispredicts **improving** and **worsening** cases (MAE 5-10, strong directional bias). This is consistent with a regression-to-the-mean model: it predicts small deltas near zero, which are correct for stable cases but miss the large swings. The balanced_tercile stratification shows somewhat reduced bias compared to sev_crossing because its equal-sized classes capture less extreme changes.

### 10.3 Best regression model vs. classification models (sev_crossing, test set)

| Model Type | Model | AUC (OvR) | BalAcc | Sens-W | PPV-W |
|---|---|---|---|---|---|
| **Classification** | XGBoost (39 feat) | **0.906** | **0.834** | **0.838** | **0.356** |
| Classification | LightGBM (39 feat) | 0.901 | 0.842 | 0.865 | 0.344 |
| **Regression** | ElasticNet (base_lag_pmcesd) | 0.722 | 0.540 | 0.243 | 0.750 |
| Regression | MixedLM (base_dev_pmcesd) | 0.756 | 0.542 | 0.784 | 0.145 |

---

## 11. Per-Person Analysis

### 11.1 Performer tier stats (test, `base_lag_pmcesd`)

Participants classified into tiers by per-person MAE (25th / 75th percentile thresholds).

| Tier | N | MAE Mean | MAE Std | MAE Median | RMSE Mean |
|---|---|---|---|---|---|
| High (good) | 24 | 1.06 | 0.38 | 1.08 | 1.25 |
| Medium | 48 | 3.14 | 0.99 | 3.04 | 3.59 |
| Low (poor) | 24 | 8.81 | 3.12 | 7.83 | 10.40 |

### 11.2 Performer tier stats (val, `base_lag_pmcesd`)

| Tier | N | MAE Mean | MAE Std | MAE Median | RMSE Mean |
|---|---|---|---|---|---|
| High (good) | 24 | 1.20 | 0.48 | 1.08 | 1.36 |
| Medium | 48 | 3.19 | 0.78 | 3.16 | 3.91 |
| Low (poor) | 24 | 8.11 | 2.41 | 7.70 | 9.36 |

The model works well for some participants (high-tier MAE ~1 CES-D point) and very poorly for others (low-tier MAE ~9 points), an 8x spread. Examining the performer analysis, the tiers map onto baseline depression severity: low-tier (poorly predicted) participants have a mean prior CES-D of ~21 -- above the clinical cutoff of 16, indicating moderate-to-severe depression with large, volatile score swings that the regression-to-mean model cannot track. High-tier (well-predicted) participants have a mean prior CES-D of ~4, with scores that stay low and change little, making them easy targets for a model that predicts small deltas near zero. To summarize, the model succeeds at predicting stability in mildly symptomatic participants but fails at predicting the large changes that matter most clinically.

---

## 12. Comparison: ElasticNet vs. MixedLM

Both regression models are now evaluated with the same within-person R² metric (median of per-person R²).

| Metric | ElasticNet (base_lag_pmcesd) | MixedLM (base_dev_pmcesd) |
|---|---|---|
| Test MAE | 4.13 | **3.80** |
| Test RMSE | 6.28 | **5.85** |
| Test R² | 0.279 | **0.376** |
| Test W-R² (median) | 0.186 | **0.389** |
| N features retained | 2 | 30 (few significant) |
| Dominant features | prior_cesd, person_mean_cesd | prior_cesd, person_mean_cesd |
| Direction AUC (sev_crossing) | 0.722 | **0.756** |
| Direction BalAcc | 0.540 | 0.542 |

Both models converge on the same regression-to-mean mechanism (prior_cesd and person_mean_cesd as dominant features). MixedLM outperforms ElasticNet by ~0.33 MAE points and substantially on W-R² (0.389 vs 0.186). The MixedLM advantage likely comes from the random intercept, which captures person-level variation more flexibly than ElasticNet's explicit person_mean_cesd feature alone.

---

## 13. Convergent Findings Across Regression and Classification

1. **`person_mean_cesd` is the key feature addition in both frameworks**: For ElasticNet regression, adding person_mean_cesd transforms performance (MAE 4.58 -> 4.13, W-R² -0.042 -> 0.186). For classification, it provides +0.031 AUC for XGBoost. Both show the largest gains from this single trait-level feature.

2. **Behavioral features alone add minimal signal**: In ElasticNet regression, adding 20 behavioral features or 17 lag features to prior_cesd changes nothing -- all are zeroed out. In MixedLM, adding 20 behavioral features actually *decreases* performance (R²: 0.303 -> 0.290). In classification, behavioral features add +0.004 AUC (n.s.) over prior_cesd alone.

3. **Person-level structure is essential**: ElasticNet conditions without person-level information (prior_cesd, base, base_lag, dev, pheno, dev_pheno) all collapse to the same model. In MixedLM, the pooled model (R² = 0.125) performs far worse than the mixed-effects model (R² = 0.290). In classification, cold-start performance drops from AUC = 0.906 to 0.821.

4. **Classification substantially outperforms regression-as-classifier for direction prediction**: The best regression model (MixedLM, AUC = 0.756) falls well below XGBoost classification (AUC = 0.906). Optimizing MAE produces near-zero predictions that collapse directional discrimination. Dedicated classifiers trained on clinical boundaries are the appropriate tool for worsening detection.

---

## 14. Reproducibility

```bash
# Run all 11 conditions end-to-end (train + posthoc + performer tiers):
python regression/elasticnet/scripts/run_all_conditions.py

# Run a single condition:
python regression/elasticnet/scripts/run_all_conditions.py --only base_lag_pmcesd

# Generate report figures/tables after all conditions are done:
python regression/elasticnet/scripts/build_report.py
```

For data details, software versions, output file index, and pipeline documentation, see [`regression/elasticnet/README.md`](../README.md).

---

## 15. Key Narrative Points

1. **ElasticNet achieves modest but consistent improvement over baselines** (MAE = 4.13, -10% vs B0) when equipped with the person-level trait anchor
2. **`person_mean_cesd` is the dominant additive feature** -- consistent across ElasticNet (+0.228 W-R²), MixedLM (+0.178 W-R²), and classification (+0.031 AUC)
3. **Behavioral features alone add no incremental value** -- paralleling the classification finding. The signal is in symptom history, not behavioral phenotype
4. **Person-specific regression-to-the-mean is the primary mechanism** -- the model predicts each person will revert toward their chronic CES-D level, with the magnitude of predicted change proportional to their current displacement from baseline
5. **MixedLM outperforms ElasticNet** (MAE 3.80 vs 4.13, W-R² 0.389 vs 0.186) -- the random intercept captures person-level structure more flexibly
6. **For clinical alerting, classification substantially outperforms regression-as-classifier** -- dedicated classifiers trained on clinical boundaries are the appropriate tool for worsening detection
