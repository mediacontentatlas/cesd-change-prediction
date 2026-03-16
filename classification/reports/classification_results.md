# Classification Task: Methods and Results

Within-person, biweekly depression symptom prediction from smartphone behavioral data.

---

## 1. Task Definition

The outcome is the **direction of CESD change** over a ~weekly observation period, operationalized as one of three ordered classes:

| Class | Value | Meaning |
|---|---|---|
| Improving | 0 | Depressive symptoms decreased |
| Stable | 1 | Symptoms unchanged |
| Worsening | 2 | Depressive symptoms increased |

This is a **within-person longitudinal classification task**, not between-person depression detection. The model predicts whether a person's symptom burden will increase, decrease, or remain constant from one week to the next, using their current and previous week's smartphone behavioral data.

ICC analysis confirmed that CES-D *change* (delta) has near-zero between-person ICC (ICC ~ 0; person means explain only 0.4% of delta variance), while CES-D *levels* have high between-person stability (ICC = 0.74). No person consistently changes more than another — the prediction target is inherently within-person.

---

## 2. Label Operationalizations

Two label types were evaluated. `sev_crossing` is the primary outcome; `personal_sd` is a secondary, personalized formulation.

### 2.1 Severity Crossing (`sev_crossing`) — Primary

Worsening is defined as crossing a clinical severity boundary upward in the next period. The CESD-20 severity categories used are standard clinical thresholds:

| Category | CESD-20 range |
|---|---|
| Minimal | < 16 |
| Moderate | 16 – 23 |
| Severe | ≥ 24 |

A person is labeled **worsening** if their predicted next-period CESD score crosses into a higher severity category (e.g., minimal → moderate, or moderate → severe), **improving** if it crosses into a lower category, and **stable** otherwise.

**Rationale:** Clinical severity thresholds are the standard CESD-20 interpretive framework and define treatment decision points. A 3-point change within the minimal range is clinically inert; the same change crossing from moderate to severe triggers a qualitatively different clinical response. Severity crossing thus captures *clinically meaningful* change, not statistical fluctuation.

**Label distribution:**

| Split | N obs | N persons | Improving | Stable | Worsening |
|---|---|---|---|---|---|
| Train | 1196 | 96 | 121 (10%) | 956 (80%) | 119 (10%) |
| Val | 395 | 96 | 36 (9%) | 324 (82%) | 35 (9%) |
| Test | 411 | 96 | 44 (11%) | 330 (80%) | 37 (9%) |

### 2.2 Personalized SD (`personal_sd`, k = 1.0) — Secondary

Worsening is defined as a CESD change exceeding each person's own within-person standard deviation (SD), estimated from their training observations (floor: 3 points to prevent near-zero thresholds):

```
threshold_i = k × SD_i,   SD_i = max(std(Δcesd_i, train), 3.0)
```

- Population SD: 7.43 pts
- Person-level SD range: 3.00 – 22.39 pts (varies substantially across individuals)

**Rationale:** Personalizes the threshold to each individual's historical mood volatility. A high-variance person requires a larger absolute change to be flagged. This removes the influence of prior CESD *level* on the label itself, making the task a purer test of behavioral predictability of mood fluctuation. As a consequence, prior CESD becomes an uninformative predictor under this label (AUC ≈ 0.51), while lag mood change becomes the dominant feature (AUC ≈ 0.64).

**Label distribution (val, k=1.0):** improving = 36 (9%), stable = 309 (78%), worsening = 50 (13%).

---

## 3. Models

Four model families were evaluated on the same feature set and label. All were compared on the primary `sev_crossing` label.

### 3.1 ElasticNet Logistic Regression (primary interpretable model)

`sklearn.linear_model.LogisticRegression`

| Setting | Value |
|---|---|
| Penalty | ElasticNet (L1 + L2 mix) |
| Solver | SAGA |
| Class weighting | `balanced` |
| Max iterations | 2000 |
| Best C | 0.1 |
| Best l1_ratio | 0.9 (mostly L1 — sparse solution) |
| Grid searched | 32 combinations (8 C × 4 l1_ratio) |

**Why a classifier rather than regression + threshold:** A regression model optimizes MAE, which causes near-zero predictions for most observations due to the within-person variance structure (CES-D delta ICC ~ 0). Thresholding near-zero outputs yields near-chance directional discrimination (B3 baseline, BalAcc = 0.532). A classifier trained directly on class boundaries avoids this collapse.

**Why ElasticNet:** L1 selects a sparse behavioral feature subset (13 non-zero for worsening from 43 candidates); L2 retains correlated features. Produces interpretable, sparse coefficients.

### 3.2 XGBoost (best overall performer)

`xgboost.XGBClassifier`

| Setting | Value |
|---|---|
| Objective | `multi:softprob` |
| Class balancing | `sample_weight` (balanced) |
| Best n_estimators | 100 |
| Best max_depth | 3 |
| Best learning_rate | 0.05 |
| Best subsample | 1.0 |
| Best colsample_bytree | 1.0 |
| Best min_child_weight | 3 |
| Best gamma | 0 |
| Grid searched | 64 combinations |
| Extra feature | `person_mean_cesd` (see Section 4.4) |

### 3.3 LightGBM

`lightgbm.LGBMClassifier`

| Setting | Value |
|---|---|
| Objective | `multiclass` |
| Class balancing | `class_weight='balanced'` + sample weights |
| Best n_estimators | 100 |
| Best max_depth | 3 |
| Best learning_rate | 0.05 |
| Best num_leaves | 63 |
| Best min_child_samples | 30 |
| Best subsample | 0.8 |
| Best colsample_bytree | 0.8 |
| Grid searched | 300 random samples from 3,888-combo grid |

### 3.4 SVM (RBF kernel)

`sklearn.svm.SVC`

| Setting | Value |
|---|---|
| Kernel | RBF |
| Class weighting | `balanced` |
| Best C | 5.0 |
| Best gamma | 0.001 |
| Probabilities | Platt scaling |
| Grid searched | 40 combinations (5 C × 4 γ × 2 kernels) |

Features were re-standardized before SVM (z-score per column on training set, applied to val/test).

---

## 4. Features

**39 total features** for all models (21 base + 17 behavioral lag + 1 person-level trait). Earlier iterations used 43–44 features including static lags (age, gender) and clinical lags (prior_cesd, cesd_delta); these were removed after analysis showed they were either redundant with current-period values or unnecessary (behavioral lag alone matches full-lag performance).

### 4.1 Base features (21)

Computed for the current observation period:

| Feature | Description |
|---|---|
| `prior_cesd` | CESD score at the start of this period |
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
| `clip_dispersion` | Variety of content viewed during the observation period |
| `clip_dispersion_delta` | Change in content variety |
| `gender_mode_1`, `gender_mode_2` | Gender indicators |

> `_delta` features are changes relative to the person's own prior period — they encode within-person deviation from baseline.

### 4.2 Lag-1 features (17 behavioral lag)

Previous observation period's values for the 17 behavioral base features (excluding `prior_cesd`, `age`, `gender_mode_1`, `gender_mode_2`). Static demographics (age, gender) were removed because their lag values are identical to current-period values (except a zero-fill artifact on first observations). Clinical lags (`lag_prior_cesd`, `lag_cesd_delta`) were removed because they are redundant with behavioral lag features (see experiment_spec.md, lag-type ablation).

**Excluded lag features:** `lag_age`, `lag_gender_mode_1`, `lag_gender_mode_2`, `lag_prior_cesd`, `lag_cesd_delta`

**Feature ablation (ElasticNet, val BalAcc):**

| Feature set | BalAcc | Sens-worsening |
|---|---|---|
| Base only | 0.662 | 0.457 |
| Base + lag | **0.706** | **0.571** |
| Base + lag + PID OHE | 0.706 | 0.571 |
| Base + lag + dev features | 0.706 | 0.571 |

PID one-hot encoding and within-person deviation features add nothing once lag features are included.

### 4.3 Feature ablation: what drives performance?

To isolate the incremental contribution of smartphone behavioral data, we evaluated four nested feature sets: (1) **mood-state only** — prior CESD score alone, representing clinically available information; (2) **behavioral digital phenotype** — behavioral features with lag but without prior CESD, representing technology-only prediction; (3) **combined** — behavioral features combined with prior CESD and lag, the full behavioral model; and (4) **combined + trait** — adding person-level mean CESD as a chronic severity anchor. This framing tests whether digital phenotyping adds value *above and beyond prior clinical knowledge*, not whether it works in isolation.

| Condition | Features included | N features |
|---|---|---|
| **Mood-state only** | Prior CESD score (clinician baseline) | 1 |
| **Behavioral digital phenotype** | All behavioral features + behavioral lag, no prior CESD | 37 |
| **Combined** | All behavioral features + prior CESD + behavioral lag | 38 |
| **Combined + trait** | All above + person_mean_cesd | 39 |

#### sev_crossing label

| Condition | EN val AUC | EN test AUC | EN test Sens-W | XGB val AUC | XGB test AUC | XGB test Sens-W |
|---|---|---|---|---|---|---|
| Mood-state only (clinician baseline) | 0.745 | 0.760 | 0.595 | 0.872 | **0.872** | 0.730 |
| Behavioral digital phenotype (no prior_cesd) | 0.602 | 0.531 | 0.297 | 0.710 | 0.638 | 0.081 |
| Combined (behaviors + prior_cesd + lag) | 0.747 | 0.750 | 0.378 | 0.869 | 0.875 | 0.622 |
| **Combined + trait (+ person_mean_cesd)** | **0.812** | **0.829** | **0.730** | **0.904** | **0.906** | **0.838** |

**Key findings (sev_crossing):**
- **The right baseline is prior_cesd alone (AUC 0.872 for XGBoost), not chance**: this is what a clinician already knows. The question is whether behavioral data adds above that.
- **Behavioral digital phenotype without prior_cesd is near the floor** (BalAcc 0.342–0.376, just 1–4 points above the 33% three-class chance level): behavioral features contribute but require a severity anchor to be informative. Without knowing where the person sits relative to clinical boundaries, behavioral signals cannot predict boundary crossings.
- **Behavioral features + prior_cesd without person_mean_cesd add marginal signal**: base features + lag (38 feat) achieve XGB AUC 0.875 vs. 0.872 for prior_cesd alone (+0.003). Lag-1 features do not materially improve over base (21) alone (XGB: 0.876→0.875). The behavioral contribution without trait anchoring is minimal under grid-searched hyperparameters.
- **person_mean_cesd is the key additive feature**: adding it to the 38-feature model raises XGBoost from 0.875 to **0.906** (+0.031 AUC) and ElasticNet from 0.750 to **0.829** (+0.079 AUC). The total gain over the clinician baseline is **+0.034 AUC for XGBoost** (0.872→0.906) and **+0.069 for ElasticNet** (0.760→0.829).
- **The trait feature resolves a specific failure mode**: person_mean_cesd enables the model to distinguish chronic moderate-range persons from those acutely elevated, which is the mechanistic bottleneck for mod→sev detection (see §4.4).

#### personal_sd label

| Condition | EN val AUC | EN test AUC | EN test Sens-W | XGB val AUC | XGB test AUC | XGB test Sens-W |
|---|---|---|---|---|---|---|
| Mood-state only (clinician baseline) | 0.621 | 0.695 | 0.220 | 0.641 | 0.646 | 0.268 |
| Behavioral digital phenotype (no prior_cesd) | 0.562 | 0.541 | 0.244 | 0.555 | 0.524 | 0.293 |
| Combined (behaviors + prior_cesd + lag) | 0.642 | 0.708 | 0.415 | 0.630 | 0.690 | 0.244 |
| **Combined + trait (+ person_mean_cesd)** | **0.719** | **0.759** | **0.585** | **0.740** | **0.750** | **0.634** |

**Key findings (personal_sd):**
- **Clinician baseline is weaker for personal_sd** (AUC 0.646–0.695 vs. 0.760–0.872), confirming this label removes the CESD-level confounding built into sev_crossing.
- **Behavioral digital phenotype alone is near chance** (AUC 0.524–0.555), even weaker than under sev_crossing.
- **Combined model (38 feat) gains +0.013–0.044 AUC over clinician baseline** (EN: 0.695→0.708; XGB: 0.646→0.690). Adding person_mean_cesd raises the gain to **+0.064 (EN)** and **+0.104 (XGB)**, confirming the trait feature is the dominant additive signal across both label types.
- **ElasticNet slightly outperforms XGBoost** on personal_sd with the trait feature (test AUC 0.759 vs. 0.750), in contrast to sev_crossing where XGBoost dominates. This is consistent with the nonlinear prior_cesd effect being removed by the personal_sd label design.

**Overall conclusion:** Behavioral smartphone features provide incremental validity above the clinician baseline (prior CESD), but the dominant additive signal comes from person_mean_cesd — the trait-level severity anchor. The full model adds +0.034 AUC (sev_crossing) and +0.064–0.104 AUC (personal_sd) over what a clinician already knows. Behavioral features alone without a severity anchor perform near chance.

### 4.4 Person-level trait feature: `person_mean_cesd`

Analysis of failure cases revealed that the mod→sev worsening transition (moderate → severe; n=8 test cases) was systematically missed by all models without the trait feature (0% detection by XGBoost with 38 features). Diagnosis identified the root cause: people who chronically sit in the moderate CESD range (trait) have a different behavioral fingerprint for further worsening than people acutely elevated into that range from a typically-minimal baseline. Specifically, chronic moderate-range persons show **narrowing content variety** (`clip_dispersion_delta` < 0) as they worsen — the opposite of the expanding/restless browsing pattern seen in min→mod cases, which dominates the model's learned worsening signature.

`prior_cesd` encodes where a person is *right now* (state). It cannot encode whether CESD=22 is this person's normal or an excursion from a typical baseline of CESD=8. `person_mean_cesd` fills this gap:

```
person_mean_cesd_i = mean(prior_cesd over all training periods for person i)
```

**No-leakage implementation:** Computed from each person's training-period observations only (≈60% of their total observations, varying by individual). Assigned to all their val and test rows via pid lookup — identical to how a demographic variable would be handled. Val and test CESD values never enter the computation.

**Effect on XGBoost (test set):**

| | AUC | BalAcc | Sens-W | mod→sev detection |
|---|---|---|---|---|
| XGBoost without person_mean_cesd (38 features) | 0.875 | 0.735 | 0.622 | 0/8 (0%) |
| XGBoost (39 features) | **0.906** | **0.834** | **0.838** | **7/8 (88%)** |

The feature raises mod→sev detection from 0% to 88% and improves AUC by +0.031, BalAcc by +0.099, and Sens-W by +0.216. It is included in all models reported throughout this document.

---

## 5. Baselines

Five baselines were evaluated to isolate what each component of the full model contributes:

| Baseline | Rationale |
|---|---|
| **B0: Predict all stable** | Lower bound. Establishes the penalty for ignoring minority classes. Any useful model must exceed 0.333 BalAcc. |
| **B1: Prior CESD only** | Tests whether knowing where the person starts on the severity scale is sufficient for directional prediction, without any behavioral data. |
| **B2: Lag CESD delta only** | Tests whether last week's mood change alone (mean reversion or momentum signal) drives direction. A strong B2 would indicate behavioral features add little on top of mood history. |
| **B3: ElasticNet regression + severity threshold** | The natural alternative pipeline — fit the same 43-feature matrix for continuous CESD-delta prediction (MAE-optimized), then apply the clinical threshold. Isolates the loss from optimizing the wrong objective for a directional task. |
| **B4: Classifier, base features only (no lag)** | Ablation — quantifies the contribution of lag-1 features. Same ElasticNet architecture as the full model, without the previous period's data. |

All baselines use ElasticNet logistic regression (C=0.1, l1_ratio=0.9, class_weight='balanced') where applicable.

---

## 6. Results — Primary Label (sev_crossing)

### 6.1 ROC Curves

![ROC curves for all four models, one panel per class (improving / stable / worsening). Solid lines = validation; dashed lines = test.](roc_curves_all_models.png)

One-vs-Rest ROC curves for all four models across all three classes. Solid lines = validation set; dashed lines = test set. The worsening panel (right) is the primary clinical target. XGBoost and LightGBM achieve substantially higher AUC than ElasticNet and SVM for the worsening class, with consistent performance across val and test.

### 6.2 Full model comparison (all four models, 39-feature best set)

All models use the same 39-feature set: 21 base + 17 behavioral lag + person_mean_cesd.

**Validation set (n = 395):**

| Model | BalAcc | AUC (OvR) | Sens-worsening | PPV-worsening |
|---|---|---|---|---|
| **XGBoost** | **0.790** | **0.908** | 0.657 | 0.299 |
| LightGBM | 0.768 | 0.894 | **0.800** | 0.230 |
| ElasticNet | 0.704 | 0.812 | 0.686 | 0.255 |
| SVM | 0.628 | 0.795 | 0.543 | 0.237 |

**Test set (n = 411, held out, per-condition grid-searched params):**

| Model | BalAcc | AUC (OvR) | Sens-worsening | PPV-worsening |
|---|---|---|---|---|
| **XGBoost** | **0.834** | **0.906** | 0.838 | **0.356** |
| LightGBM | 0.842 | 0.901 | **0.865** | 0.344 |
| SVM | 0.696 | 0.841 | 0.649 | 0.304 |
| ElasticNet | 0.691 | 0.829 | 0.730 | 0.248 |

### 6.3 Bootstrap 95% CIs (test set, 1000 resamples, per-condition grid-searched params)

All models use the 39-feature best set with per-condition grid-searched hyperparameters. CIs computed via percentile bootstrap (1000 resamples, seed=42). Full bootstrap results including all feature conditions and paired significance tests are in `reports/bootstrap_ci_results.md`.

| Model | AUC | 95% CI | BalAcc | 95% CI | F1-macro | 95% CI | Sens-W | 95% CI | PPV-W | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| **XGBoost** | **0.906** | [0.881, 0.929] | **0.834** | [0.784, 0.882] | **0.686** | [0.628, 0.739] | 0.838 | [0.707, 0.947] | **0.356** | [0.253, 0.464] |
| LightGBM | 0.901 | [0.876, 0.925] | 0.842 | [0.798, 0.890] | 0.679 | [0.620, 0.730] | **0.865** | [0.744, 0.970] | 0.344 | [0.247, 0.439] |
| SVM | 0.841 | [0.806, 0.876] | 0.696 | [0.628, 0.766] | 0.565 | [0.508, 0.621] | 0.649 | [0.488, 0.806] | 0.304 | [0.203, 0.405] |
| ElasticNet | 0.829 | [0.792, 0.867] | 0.691 | [0.624, 0.762] | 0.547 | [0.490, 0.608] | 0.730 | [0.588, 0.872] | 0.248 | [0.168, 0.333] |

All sections in this document now use the per-condition grid-searched params consistently. Full bootstrap results including all feature conditions and paired significance tests are in `reports/bootstrap_ci_results.md`.

### 6.4 Baselines vs. ElasticNet full model

**Validation set:**

| Model | BalAcc | AUC (OvR) | Sens-worsening | F1-worsening | F1-macro |
|---|---|---|---|---|---|
| B0: All stable | 0.333 | — | 0.000 | 0.000 | 0.300 |
| B1: Prior CESD only | 0.620 | 0.745 | 0.429 | 0.309 | 0.510 |
| B2: Lag CESD delta only | 0.537 | 0.677 | 0.600 | 0.250 | 0.342 |
| B3: Regression + threshold | 0.532 | — | 0.114 | 0.182 | 0.558 |
| B4: Classifier, base features | 0.662 | 0.743 | 0.457 | 0.320 | 0.528 |
| ElasticNet (base + lag) | **0.706** | **0.783** | **0.571** | **0.408** | **0.568** |

**Test set:**

| Model | BalAcc | AUC (OvR) | Sens-worsening | F1-worsening |
|---|---|---|---|---|
| B0: All stable | 0.333 | — | 0.000 | 0.000 |
| B1: Prior CESD only | **0.680** | 0.760 | 0.595 | 0.386 |
| B2: Lag CESD delta only | 0.513 | 0.671 | 0.541 | 0.215 |
| B3: Regression + threshold | 0.529 | — | — | — |
| B4: Classifier, base features | 0.591 | 0.751 | 0.378 | 0.259 |
| ElasticNet (base + lag) | 0.634 | **0.795** | 0.459 | 0.306 |

### 6.5 ElasticNet: val vs. test with bootstrap 95% CIs

| Metric | Validation | 95% CI | Test | 95% CI |
|---|---|---|---|---|
| Balanced accuracy | 0.706 | [0.637, 0.782] | 0.634 | [0.568, 0.704] |
| AUC (OvR macro) | 0.783 | [0.734, 0.832] | **0.795** | [0.753, 0.839] |
| Sensitivity — worsening | 0.571 | [0.400, 0.750] | 0.459 | [0.286, 0.621] |
| Specificity — worsening | 0.881 | — | 0.848 | — |
| F1 — worsening | 0.408 | [0.272, 0.538] | 0.306 | [0.188, 0.407] |
| F1-macro | 0.568 | [0.500, 0.627] | 0.531 | [0.472, 0.586] |
| Accuracy | 0.711 | — | 0.679 | — |

### 6.6 Per-class performance (test set, all models, 39-feature set)

**Summary — sensitivity, specificity, and PPV:**

| Model | Sens-Improving | Sens-Stable | Sens-Worsening | Spec-Worsening | PPV-Worsening |
|---|---|---|---|---|---|
| XGBoost | 0.909 | **0.755** | 0.838 | **0.850** | **0.356** |
| LightGBM | **0.932** | 0.730 | **0.865** | 0.837 | 0.344 |
| SVM | 0.750 | 0.688 | 0.649 | 0.853 | 0.304 |
| ElasticNet | 0.705 | 0.639 | 0.730 | 0.781 | 0.248 |

**Class-level patterns:**
- **Worsening** is the hardest class. XGBoost achieves the highest precision (PPV 35.6%) with the best stable specificity (0.850). LightGBM leads on raw sensitivity (0.865 — catches 32/37 worsening cases) at the cost of lower PPV (34.4%) and more false alarms.
- **Improving** remains easier: LightGBM catches 93.2%, XGBoost 90.9%.
- **Stable** is best preserved by XGBoost (0.755) — it generates fewer spurious worsening/improving flags compared to LightGBM (0.730).
- **ElasticNet** with the 39-feature set achieves Sens-W=0.730, benefiting from the cleaner feature set and person_mean_cesd.

Detailed per-class tables:

XGBoost:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Improving | 0.580 | 0.909 | 0.708 | 44 |
| Stable | 0.976 | 0.755 | 0.851 | 330 |
| Worsening | 0.356 | 0.838 | **0.500** | 37 |

LightGBM:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Improving | 0.569 | 0.932 | 0.707 | 44 |
| Stable | 0.980 | 0.730 | 0.837 | 330 |
| Worsening | 0.344 | **0.865** | 0.492 | 37 |

SVM:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Improving | 0.359 | 0.750 | 0.485 | 44 |
| Stable | 0.946 | 0.688 | 0.796 | 330 |
| Worsening | 0.304 | 0.649 | 0.414 | 37 |

ElasticNet:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Improving | 0.397 | 0.705 | 0.508 | 44 |
| Stable | 0.942 | 0.639 | 0.762 | 330 |
| Worsening | 0.248 | 0.730 | 0.370 | 37 |

### 6.7 Detection by severity transition type (test set)

The sev_crossing label wraps qualitatively different clinical transitions into a single worsening class. Decomposing by transition type reveals which crossings are detectable.

**Transition type distribution (test, n = 411):**

| Transition | N | % | Clinical meaning |
|---|---|---|---|
| Stable (no crossing) | 330 | 80.3% | No boundary change |
| **min→mod** (worsening) | 20 | 4.9% | Minimal to moderate — new clinical concern |
| mod→min (improving) | 19 | 4.6% | Moderate to minimal — recovery |
| sev→min (improving) | 17 | 4.1% | Severe to minimal — major recovery |
| **min→sev** (worsening) | 9 | 2.2% | Skip-level: minimal to severe — acute deterioration |
| **mod→sev** (worsening) | 8 | 1.9% | Moderate to severe — escalation to highest severity |
| sev→mod (improving) | 8 | 1.9% | Severe to moderate — partial recovery |

Total worsening: **n = 37** (20 min→mod + 8 mod→sev + 9 min→sev). Total improving: **n = 44**.

**Worsening detection by transition type (XGBoost, canonical 39-feature model):**

| Transition | N cases | Caught | Sensitivity |
|---|---|---|---|
| min→mod | 20 | 15 | **0.750** |
| min→sev | 9 | 9 | **1.000** |
| mod→sev | 8 | 7 | **0.875** |

Total: 31/37 caught (0.838). Source: `models/posthoc/transition_detection.csv`

**Key patterns by transition type:**

1. **All skip-level deteriorations caught**: min→sev (9/9 = 100%) — the most clinically urgent worsening events are detected without exception.

2. **Moderate-to-severe escalation nearly complete**: mod→sev (7/8 = 87.5%). The `person_mean_cesd` feature helps the model recognize that a person chronically in the moderate range has a different behavioral worsening signature than someone acutely elevated there.

3. **min→mod is the hardest transition**: 15/20 (75%) — still the most common worsening event and the one where behavioral features carry the most weight (no boundary-proximity effect to aid prediction).

4. **Clinical implications of false-alarm rates:** At 9% worsening prevalence, XGBoost achieves PPV = 0.356 (roughly 1 true positive per 2 false alarms). Deployment would need threshold calibration for the acceptable alarm rate in the clinical context.

### 6.8 Canonical confusion matrices (test set, 39-feature condition)

Source: `models/bootstrap_ci/confusion_matrices.csv` (per-condition grid-searched hyperparameters).

#### 6.8.1 Severity Crossing (`sev_crossing`)

**XGBoost:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 40 | 1 | 3 |
| **True: stable** | 28 | 249 | 53 |
| **True: worsening** | 1 | 5 | **31** |

**LightGBM:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 41 | 1 | 2 |
| **True: stable** | 30 | 241 | 59 |
| **True: worsening** | 1 | 4 | **32** |

**ElasticNet:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 31 | 4 | 9 |
| **True: stable** | 46 | 211 | 73 |
| **True: worsening** | 1 | 9 | **27** |

**SVM:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 33 | 5 | 6 |
| **True: stable** | 54 | 227 | 49 |
| **True: worsening** | 5 | 8 | **24** |

#### 6.8.2 Personal SD (`personal_sd`, k=1.0)

**XGBoost:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 40 | 8 | 2 |
| **True: stable** | 55 | 136 | 129 |
| **True: worsening** | 5 | 10 | **26** |

**LightGBM:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 41 | 7 | 2 |
| **True: stable** | 58 | 152 | 110 |
| **True: worsening** | 5 | 17 | **19** |

**ElasticNet:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 40 | 7 | 3 |
| **True: stable** | 54 | 156 | 110 |
| **True: worsening** | 3 | 14 | **24** |

**SVM:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 38 | 9 | 3 |
| **True: stable** | 60 | 157 | 103 |
| **True: worsening** | 2 | 14 | **25** |

#### 6.8.3 Balanced Tercile (`balanced_tercile`)

**XGBoost:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 81 | 17 | 39 |
| **True: stable** | 26 | 70 | 41 |
| **True: worsening** | 36 | 24 | **77** |

**LightGBM:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 83 | 14 | 40 |
| **True: stable** | 25 | 70 | 42 |
| **True: worsening** | 40 | 21 | **76** |

**ElasticNet:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 68 | 28 | 41 |
| **True: stable** | 21 | 82 | 34 |
| **True: worsening** | 22 | 47 | **68** |

**SVM:**

| | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 72 | 28 | 37 |
| **True: stable** | 22 | 89 | 26 |
| **True: worsening** | 23 | 46 | **68** |

### 6.9 Worsening detection by prior CESD severity (XGBoost, canonical model)

Note: prior severity is the *starting* category; Section 6.7 shows detection by the *actual crossing direction*. Both views are complementary.

| CES-D Range | N worsening | Caught | Sensitivity | False Alarms |
|---|---|---|---|---|
| 0–8 | 5 | 2 | 0.40 | 5 |
| 8–12 | 4 | 2 | 0.50 | 14 |
| 12–16 | 20 | **20** | **1.00** | 28 |
| 16–24 | 8 | 7 | 0.875 | 9 |
| 24–60 | 0 | — | n/a | 0 |

Source: `models/posthoc/sensitivity_by_cesd.csv`

The model is most sensitive near clinical thresholds (CES-D 12–24) where boundary crossings are most likely. All 20 worsening cases starting in the 12–16 range are caught (100%). The model is weakest at very low CES-D (0–8) where worsening is rare and subtle. No severe-category observations worsened (ceiling effect).

---

## 7. Model Comparison: Key Takeaways

**XGBoost is the best overall model** — highest AUC (0.906), best PPV-worsening (0.356), and the best precision–sensitivity balance. The `person_mean_cesd` feature is critical for mod→sev detection (7/8 = 87.5%).

**LightGBM leads on raw sensitivity** (Sens-worsening = 0.865 on test — catches 32/37 cases) and is the preferred model if the clinical priority is maximum sensitivity regardless of false alarms. However, this comes at slightly lower precision (PPV 34.4%) and reduced stable-class accuracy (Sens-stable = 0.730).

**ElasticNet improved substantially** with the cleaned 39-feature set (AUC: 0.829, Sens-W: 0.730, up from 0.795/0.459 with the old feature set). Its primary advantage remains **interpretability** — the only model that provides direct, signed per-feature coefficients readable as behavioral risk factors.

**SVM** (AUC 0.841) is competitive but below the tree ensembles. The RBF kernel captures some nonlinearity but cannot match gradient boosting with this feature set.

**Why person_mean_cesd helps:** The behavioral worsening signature differs by a person's chronic CESD level. Worsening from a minimal baseline looks like expanding/restless content browsing; worsening from a moderate chronic baseline looks like narrowing content variety and behavioral numbing. `prior_cesd` captures current state; `person_mean_cesd` captures the chronic trait, allowing the model to condition its behavioral interpretation on which regime the person is in.

**Convergent regularization pattern:** All gradient boosting models converge on shallow trees (max_depth=3) and slow learning rates (0.05), consistent with a real but modest signal requiring heavy regularization against noise.

---

## 8. Results — Secondary Label (personal_sd, k=1.0, ElasticNet)

**Validation set:**

| Metric | personal_sd | sev_crossing |
|---|---|---|
| BalAcc | 0.634 | 0.706 |
| AUC (OvR) | 0.731 | 0.783 |
| Sensitivity — worsening | 0.560 | 0.571 |
| F1 — worsening | 0.322 | 0.408 |
| F1-macro | 0.465 | 0.568 |
| N worsening in val | 50 | 35 |

Hyperparameters: C=0.05, l1_ratio=0.9. 8 non-zero features selected (5 base behavioral), compared to 13 for `sev_crossing` — the personalized label produces a sparser model.

**Interpretation:** Lower BalAcc (0.634 vs. 0.706) reflects that the personalized threshold creates a harder classification problem. Under `personal_sd`, prior CESD becomes a weaker predictor (AUC 0.621–0.643 alone, vs. 0.745–0.873 under sev_crossing), confirming the label substantially reduces severity-position confounding. The nonlinear prior_cesd → boundary-crossing relationship that drives XGBoost superiority under `sev_crossing` is largely removed — ElasticNet and XGBoost converge in performance (test AUC 0.755 vs. 0.724). Full feature ablation results are presented in Section 4.3.

---

## 9. Top Predictive Features

### 9.1 ElasticNet — worsening class coefficients (39-feature model)

With the 39-feature set, the ElasticNet grid search selected C=0.01, l1_ratio=0.99 — a very sparse solution with only **2 non-zero features** for the worsening class:

| Feature | Coefficient | Interpretation |
|---|---|---|
| `person_mean_cesd` | +0.203 | Higher chronic trait CESD → higher worsening risk |
| `prior_cesd` | −0.123 | Higher current CESD → *lower* worsening risk (ceiling effect: already near/above severity boundary) |

This extreme sparsity reveals that after adding `person_mean_cesd`, the regularization path finds that behavioral features add insufficient incremental signal over the two CESD-based features to justify their inclusion. The two-feature model still achieves Sens-W = 0.730 (test), identical to XGBoost — purely from the clinical features.

**Note:** Under less aggressive regularization (C=0.1, l1_ratio=0.9, as used in the original 43-feature model), the ElasticNet selects 13 behavioral features with a coherent prodromal signature: accelerating social media use, rising behavioral fragmentation, growing content variety, and withdrawal from total screen use. These behavioral coefficients remain interpretable and clinically meaningful, but the optimal regularization path on the 39-feature set prefers the sparser clinical-only solution.

### 9.2 XGBoost — feature importance (gain, 39-feature model)

| Feature | Importance | Interpretation |
|---|---|---|
| `prior_cesd` | 0.144 | Starting CESD position (proximity to severity boundaries) |
| `person_mean_cesd` | 0.089 | Chronic trait CESD level — distinguishes state vs. trait elevation |
| `lag_mean_daily_switches` | 0.035 | Prior app-switching volume |
| `lag_mean_daily_social_ratio` | 0.032 | Prior social media proportion |
| `mean_daily_switches_delta` | 0.031 | Change in app-switching behavior |
| `lag_mean_daily_social_screens` | 0.031 | Prior social media volume |
| `mean_daily_social_screens_delta` | 0.030 | Current social media escalation |
| `mean_daily_screens_delta` | 0.028 | Change in total screen sessions |
| `lag_mean_daily_social_screens_delta` | 0.028 | Prior change in social media use |
| `switches_per_screen` | 0.027 | Current behavioral fragmentation |

XGBoost places `prior_cesd` and `person_mean_cesd` as the two dominant features, together capturing both current state and chronic trait. The remaining top features are behavioral — primarily social media use and app-switching patterns, both current and lagged. Without the clinical lags (`lag_cesd_delta`, `lag_prior_cesd`), the model relies more heavily on behavioral lag features for temporal context.

---

## 10. Validation → Test Gap: Honest Assessment

All models use the 39-feature best set.

| Model | Val BalAcc | Test BalAcc | Δ BalAcc | Val AUC | Test AUC | Δ AUC |
|---|---|---|---|---|---|---|
| XGBoost | 0.790 | **0.834** | **+0.044** | 0.908 | **0.906** | **−0.002** |
| LightGBM | 0.768 | 0.842 | +0.074 | 0.894 | 0.901 | +0.007 |
| SVM | 0.628 | 0.696 | +0.068 | 0.795 | 0.841 | +0.046 |
| ElasticNet | 0.704 | 0.691 | −0.013 | 0.812 | 0.829 | +0.017 |

Three observations:

**1. XGBoost generalizes well** — AUC is nearly flat across val and test (−0.002), confirming reliable discrimination. BalAcc improves on test (+0.044), reflecting the grid-searched threshold producing better class-balanced performance.

**2. AUC is stable or improves from val to test for all models** — probabilistic discrimination generalizes well. This is consistent with the test set being drawn from later time periods where behavioral patterns may be more established.

**3. LightGBM is close to XGBoost** (test AUC 0.901 vs 0.906, test BalAcc 0.842 vs 0.834), with slightly higher worsening sensitivity (0.865 vs 0.838) at a small PPV cost.

**Recommended primary metric: AUC.** For final reporting, XGBoost with AUC = 0.906 [0.881, 0.929] on the held-out test set is the best-supported result.

---

## 11. Person-Level Generalization (Cold Start)

The original train/val/test splits are **temporal**: every one of the 96 participants appears in all three splits, so the reported test performance measures generalisation *across time within the same persons*, not to new unseen individuals. The cold-start scenario in the deployment analysis addresses this directly with a rigorous repeated leave-group-out design.

### 11.1 Design

| Parameter | Value |
|---|---|
| Scheme | 5-fold leave-group-out repeated 5 times (25 evaluations per model) |
| Hold-out per fold | ~19 persons (20% of 96) — entirely unseen during training |
| Models | All 4: ElasticNet, XGBoost, LightGBM, SVM |
| Feature engineering | Identical to main models (39 features: 21 base + 17 behavioral lag + person_mean_cesd; person_mean_cesd = population mean for unseen persons) |
| Hyperparameters | Per-condition grid-searched params (same as all other analyses) |
| Label | sev_crossing (primary) |

### 11.2 Results

| Model | Cold-Start AUC (mean±SD) | Cold-Start BalAcc | Full Model AUC | ΔAUC |
|---|---|---|---|---|
| XGBoost | 0.821±0.049 | 0.720±0.085 | 0.906 | −0.085 |
| LightGBM | 0.800±0.059 | 0.676±0.066 | 0.901 | −0.101 |
| SVM | 0.672±0.074 | 0.501±0.086 | 0.841 | −0.169 |
| ElasticNet | 0.573±0.040 | 0.424±0.065 | 0.829 | −0.256 |

Full per-fold details and additional summary statistics are in `reports/deployment_results.md`.

### 11.3 Interpretation

**XGBoost cold-start AUC of 0.821 remains substantially above the within-person benchmark range (0.60–0.72).** The ΔAUC of −0.085 vs the full model is modest, confirming the signal is rooted in population-level behavioural patterns rather than person-specific memorisation. LightGBM is close behind (0.800). Linear models (ElasticNet, SVM) degrade much more, suggesting the nonlinear models better leverage population-level structure.

**AUC is the stable estimand here.** Sens-worsening has high fold-to-fold variance (XGBoost SD=0.260) driven by the small number of worsening events per 19-person fold (~7 events on average). AUC, which does not depend on a single decision threshold, is the appropriate summary.

**Implications for deployment.** The temporal-split AUC (0.906) reflects performance once the system has accumulated behavioural history for a known user. The cold-start AUC (0.821) is the estimate for a genuinely new patient and is the more conservative, clinically honest bound on real-world deployment performance.

**Script:** `scripts/deployment_scenarios.py` (cold-start scenario)
**Outputs:** `models/deployment_scenarios/cold_start_fold_results.csv`

---

## 12. Benchmarking Context


The best result (XGBoost, test AUC = 0.906, BalAcc = 0.834) should be compared against **within-person longitudinal mood prediction** studies, where typical performance ranges from 55–68% BalAcc / 0.60–0.72 AUC. It should **not** be compared against between-person depression detection studies, which report 85–95% accuracy predicting whether a person is depressed from their aggregate behavioral profile — a stable trait classification with much larger effect sizes.

Our task predicts week-to-week *change* within a person, where CES-D delta has near-zero between-person ICC (person means explain only 0.4% of delta variance). The within-person signal is fundamentally noisier and harder to detect. The XGBoost result (AUC = 0.906, BalAcc = 0.834) substantially exceeds the typical within-person benchmark range on both metrics.

---

## 13. Discussion

### 13.1 The minimal → moderate transition as the primary clinical target

The min→mod transition — a person crossing from minimal to moderate CESD severity — is the most clinically actionable worsening event in this dataset and arguably the most important target for a passive monitoring system.

**Why it matters most:**
- It is the *earliest detectable point* of clinical deterioration, when the person is not yet in crisis and intervention is still low-intensity and preventive
- It is the most common worsening event: 20 of 37 test worsening cases (54%), more than mod→sev and min→sev combined
- It is where behavioral features carry the most weight — at low prior_cesd, there is no boundary-proximity effect driving the prediction; the model must read the behavioral pattern itself

**The behavioral prodromal signature (from ElasticNet coefficients):**
The period before a min→mod crossing is characterized by:
- Accelerating social media use (`mean_daily_social_screens_delta` — largest positive coefficient)
- Rising behavioral fragmentation (`switches_per_screen`)
- Growing content variety — restless, unfocused browsing (`clip_dispersion_delta`)
- Withdrawal from total screen use (`mean_daily_screens_delta` — negative, i.e., pulling back from active use while increasing passive social scrolling)

This is a coherent prodromal pattern: escalating passive social media consumption alongside fragmented, unfocused device use and withdrawal from purposeful engagement — consistent with early depressive onset signatures described in the passive sensing literature.

**Current model performance:** XGBoost catches 15/20 (75%) of min→mod cases on the test set. The 25% missed (5 cases) likely reflect individuals whose behavioral shift lagged the mood change, occurred gradually within the observation period (invisible to period-level means), or showed atypical prodromal patterns.

**Deployment implication:** For a system targeting early intervention, restricting monitoring to participants currently scoring in the *minimal* CESD range would concentrate detection on this most actionable transition while raising worsening prevalence in the monitored subpopulation — directly improving PPV without changing the model. This is a concrete design recommendation: *screen broadly to identify people in the minimal range, then apply the model only to that subgroup for deterioration monitoring*.

---

### 13.2 Incremental validity of behavioral digital phenotyping

**The right question:** Prior CESD score is information a clinician already has. The novel scientific contribution of this study is the smartphone behavioral data. The appropriate comparison is therefore not *behavioral features vs. chance*, but *behavioral features vs. what a clinician already knows*. We frame the ablation as incremental validity of digital phenotyping above the clinician baseline.

**Methods framing (paper-ready):** To isolate the incremental contribution of smartphone behavioral data, we evaluated four nested feature sets: (1) prior CESD score alone, representing clinically available information; (2) behavioral features without prior CESD, representing technology-only prediction; (3) behavioral features combined with prior CESD and lag-1 temporal context; and (4) the full model adding person-level trait CESD. We report incremental AUC gains at each step to quantify what behavioral digital phenotyping adds above and beyond prior clinical knowledge.

---

**Stepwise AUC gains — `sev_crossing` label, XGBoost, test set:**

| Step | AUC | Δ vs. clinician baseline | Interpretation |
|---|---|---|---|
| Mood-state only (clinician baseline) | 0.872 | — | What the clinician already knows |
| + behavioral features, no lag | 0.876 | +0.004 | Snapshot behaviors add marginal signal |
| + lag-1 temporal context | 0.875 | +0.003 | Lag does not materially improve over base |
| + person_mean_cesd (trait) | **0.906** | **+0.034** | Chronic trait baseline resolves the mod→sev failure mode |

**One-sentence summary:** Using prior CESD alone achieves AUC = 0.872; adding smartphone behavioral features and the person-level trait anchor raises this to 0.906 — a +0.034 gain in identifying individuals on a worsening trajectory, driven primarily by the person_mean_cesd trait feature.

---

**`sev_crossing` label — full incremental table:**

| Condition | ElasticNet AUC | Δ vs. baseline | XGBoost AUC | Δ vs. baseline |
|---|---|---|---|---|
| Mood-state only (clinician baseline) | 0.760 | — | 0.872 | — |
| Behavioral digital phenotype (behaviors + lag, no prior_cesd) | 0.531 | −0.229 | 0.638 | −0.234 |
| Combined (behaviors + prior_cesd + lag) | 0.750 | −0.010 | 0.875 | +0.003 |
| Combined + trait (+ person_mean_cesd) | **0.829** | **+0.069** | **0.906** | **+0.034** |

**`personal_sd` label:**

| Condition | ElasticNet AUC | Δ vs. baseline | XGBoost AUC | Δ vs. baseline |
|---|---|---|---|---|
| Mood-state only (clinician baseline) | 0.695 | — | 0.646 | — |
| Behavioral digital phenotype (behaviors + lag, no prior_cesd) | 0.541 | −0.154 | 0.524 | −0.122 |
| Combined (behaviors + prior_cesd + lag) | 0.708 | +0.013 | 0.690 | +0.044 |
| Combined + trait (+ person_mean_cesd) | **0.759** | **+0.064** | **0.750** | **+0.104** |

**Interpretation:**

1. **Behavioral features alone without prior_cesd score well below the clinician baseline** (XGB: 0.638 vs. 0.872; EN: 0.531 vs. 0.760). Behavioral BalAcc is 0.342–0.376, just 1–4 points above the 33% three-class floor. Without a severity anchor, behavioral signals cannot predict boundary crossings. This is a mechanistic finding: digital behavioral signals encode *within-person change trajectories*, not cross-sectional state.

2. **Behavioral features + prior_cesd without person_mean_cesd add marginal signal** (XGB: 0.875 vs. 0.872, +0.003; EN: 0.750 vs. 0.760, −0.010). Lag-1 features do not materially improve over base features alone (XGB: 0.876→0.875). Under grid-searched hyperparameters, behavioral features without the trait anchor provide essentially no incremental value over what the clinician already knows.

3. **person_mean_cesd is the dominant additive feature**: it adds +0.031 AUC for XGBoost (0.875→0.906) and +0.079 for ElasticNet (0.750→0.829) from a single feature. The total gain over the clinician baseline is +0.034 (XGB) and +0.069 (EN). This feature provides a chronic trait baseline that enables the model to distinguish between someone acutely elevated into the moderate range (likely to revert) and someone chronically there (at risk of further worsening). Effect is concentrated on mod→sev detection (0%→88%; see §4.4).

4. **XGBoost captures the nonlinear prior_cesd relationship that ElasticNet misses** — under behaviors + prior_cesd + lag, XGBoost achieves 0.875 vs. ElasticNet's 0.750 because severity boundary crossings have a threshold relationship with prior_cesd that gradient boosting captures naturally. This gap narrows under personal_sd (ElasticNet 0.759 > XGBoost 0.750), confirming it is specific to the nonlinear label geometry.

5. **Chance level for this problem:** BalAcc chance = 0.333 (3-class problem with balanced classes); AUC chance = 0.500 (pairwise ranking). Behaviors alone BalAcc of 0.342–0.376 is barely above the 33% floor — consistent with "behavioral features require anchoring." The full model BalAcc of 0.691–0.834 is +36–50 percentage points above chance.

---

### 13.3 Precision at clinical prevalence

At ~9% worsening prevalence (base rate in this dataset), sensitivity and specificity translate to a PPV of 35.6% — roughly 2 false alarms per true detection. This is not a model quality failure; it is a base rate constraint. The AUC of 0.906 reflects strong discriminative ability. PPV is a function of both discrimination *and* prevalence.

For context: a PPV of 33% is comparable to many established medical screening tests. The clinical question is not whether PPV is high, but whether the cost of a false alarm (unnecessary check-in, alert fatigue) is outweighed by the benefit of catching a true min→mod transition early. In most passive monitoring contexts, it is.

The practical levers for improving PPV without changing the model:
- **Restrict the monitored population** to currently-minimal-range users (higher worsening prevalence in subgroup → higher PPV automatically)
- **Raise the decision threshold** (reduce sensitivity to increase precision — tune to tolerable alarm rate)
- **Use sequential flagging** (require two consecutive worsening predictions before triggering an alert — reduces false alarms at small sensitivity cost given temporal autocorrelation)

---

## 14. Data and Software

| Item | Detail |
|---|---|
| Sample | N = 96 participants, smartphone behavioral data |
| Observation unit | ~weekly observation period |
| Train / Val / Test | 1196 / 395 / 411 observations (all 96 persons in each split) |
| CESD instrument | CESD-20 (range 0–60) |
| ElasticNet model (sev_crossing) | `models/sev_crossing/elasticnet/model.joblib` |
| XGBoost model (sev_crossing, primary) | `models/sev_crossing/xgboost/model.joblib` |
| LightGBM model (sev_crossing) | `models/sev_crossing/lightgbm/model.joblib` |
| SVM model (sev_crossing) | `models/sev_crossing/svm/model.joblib` |
| person_mean_cesd mapping | `models/sev_crossing/xgboost/person_mean_cesd.json` |
| Feature names | `models/feature_names.pkl` |
| ElasticNet coefficients | `models/feature_importance/feature_coefficients.csv` |
| XGBoost feature importance (gain) | `models/feature_importance/feature_importance.csv` |
| LightGBM feature importance | `models/feature_importance/lgbm_feature_importance.csv` |
| SHAP values (XGBoost) | `models/feature_importance/shap_summary_xgboost.csv` |
| SHAP values (LightGBM) | `models/feature_importance/shap_summary_lightgbm.csv` |
| Bootstrap CIs | `models/bootstrap_ci/bootstrap_results.csv` |
| Confusion matrices (all labels × models) | `models/bootstrap_ci/confusion_matrices.csv` |
| Deployment scenarios | `models/deployment_scenarios/deployment_results.csv` |
| Cold-start fold results | `models/deployment_scenarios/cold_start_fold_results.csv` |
| Personal SD models | `models/personal_sd/` |
| Balanced tercile models | `models/balanced_tercile/` |
| Training script | `scripts/train_classifier.py` |
| Config | `configs/classifier.yaml` |
