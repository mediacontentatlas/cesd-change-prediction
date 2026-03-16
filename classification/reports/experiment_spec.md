# Experiment Specification — Ablations & Model Analysis

## Context
Within-person depression symptom prediction from Screenome behavioral data.
- 96 participants, biweekly CES-D assessments
- Train/Val/Test: 1196 / 395 / 411 observations
- Primary label: sev_crossing (clinical severity boundary crossing at CES-D 16/24)
- Secondary label: personal_sd (k=1.0, personalized threshold)
- Primary model: XGBoost; Secondary model: ElasticNet; also LightGBM and SVM
- Hyperparameters are **grid-searched per model × feature condition** on the validation set (consistent across all analyses)
- Report test set AUC (OvR macro), BalAcc, F1-macro, Sens-W, PPV-W for every condition
- Bootstrap CIs (1000 resamples, seed=42) for all metrics; paired tests for feature ablation significance
- Use existing train/val/test splits as-is

---

## Feature definitions

### Base features (21)
These are computed for the CURRENT observation period t:
- prior_cesd (CES-D score at the start of period t)
- active_day_ratio_delta
- mean_daily_overnight_ratio, mean_daily_overnight_ratio_delta
- mean_daily_social_ratio, mean_daily_social_ratio_delta
- mean_daily_screens, mean_daily_screens_delta
- mean_daily_unique_apps, mean_daily_unique_apps_delta
- mean_daily_switches, mean_daily_switches_delta
- switches_per_screen, switches_per_screen_delta
- mean_daily_social_screens, mean_daily_social_screens_delta
- clip_dispersion, clip_dispersion_delta
- age, gender_mode_1, gender_mode_2

Note: active_day_ratio level was dropped by VIF filtering — only active_day_ratio_delta remains.
_delta features = change within current period relative to person's own prior-period baseline.

### Clinical features
- prior_cesd: CES-D score at the START of the current observation period t
- person_mean_cesd: each person's mean prior_cesd across ALL their training observations only. Constant per person. Assigned to val/test via pid lookup — never uses val/test CES-D values. **Important: this is a CES-D-derived feature.**

### Lag features — from observation period t-1
Lag features are computed only for time-varying behavioral features. Static demographics (age, gender_mode_1, gender_mode_2) are excluded from lag construction since they don't change between periods.

**Behavioral lag (17 features):** lag_ prefix on the 17 non-static, non-clinical base features (all behavioral level + delta features).

**Clinical lag (2 features):**
- lag_prior_cesd: CES-D score at the start of period t-1
- lag_cesd_delta: CES-D CHANGE during period t-1

**Total lag: 19 features** (17 behavioral + 2 clinical)

Note: For each person's first observation (where t-1 doesn't exist), lag features are filled with 0 (not dropped).

### Dev features (8) — not in current final model, tested separately
Each dev feature = current period value minus person's own training-period mean for that feature.
Computed for a subset of behavioral features (the 8 non-delta screenome features):
- mean_daily_overnight_ratio_dev, mean_daily_social_ratio_dev, mean_daily_screens_dev, mean_daily_unique_apps_dev, mean_daily_switches_dev, switches_per_screen_dev, mean_daily_social_screens_dev, clip_dispersion_dev

Like person_mean_cesd, these are computed from training observations only and assigned to val/test via pid lookup.

---

## Best model: Behavioral lag only + person_mean_cesd (39 features)

The recommended model drops the 2 clinical lag features (lag_prior_cesd, lag_cesd_delta) since behavioral lag alone matches full model performance. It also excludes redundant static-feature lags.

**Feature set (39 features):**
- 21 base features (prior_cesd + 17 behavioral + 3 demographics)
- 17 behavioral lag features (excluding lag of age, gender, and clinical features)
- 1 person_mean_cesd

### Performance — sev_crossing label (per-condition grid-searched params)

| Model | AUC [95% CI] | BalAcc | F1-macro | Sens-W | PPV-W |
|---|---|---|---|---|---|
| XGBoost | **0.906** [0.881, 0.929] | 0.834 | 0.686 | 0.838 | 0.356 |
| LightGBM | **0.901** [0.876, 0.925] | 0.842 | 0.679 | 0.865 | 0.344 |
| SVM | 0.841 [0.806, 0.876] | 0.696 | 0.565 | 0.649 | 0.304 |
| ElasticNet | 0.829 [0.792, 0.867] | 0.691 | 0.547 | 0.730 | 0.248 |

Note: AUC 0.906 uses per-condition grid-searched params (lr=0.01, max_depth=3, min_child_weight=1). Earlier fixed-param results (AUC 0.915, lr=0.05) are superseded for consistency across all analyses.

### Confusion matrices

**XGBoost (39 features):**
```
                 pred_impr  pred_stab  pred_wors
true_improving          39          2          3
true_stable             28        251         51
true_worsening           2          8         27
```

**ElasticNet (39 features):**
```
                 pred_impr  pred_stab  pred_wors
true_improving          31          6          7
true_stable             45        206         79
true_worsening           2         10         25
```

---

## Feature importance — Best model (39 features)

### XGBoost feature importance (gain)

| Rank | Feature | Gain | Cumulative |
|---|---|---|---|
| 1 | prior_cesd | 0.145 | 14.5% |
| 2 | person_mean_cesd | 0.088 | 23.4% |
| 3 | lag_mean_daily_switches | 0.036 | 26.9% |
| 4 | lag_mean_daily_social_screens | 0.033 | 30.2% |
| 5 | lag_mean_daily_social_ratio | 0.033 | 33.4% |
| 6 | mean_daily_switches_delta | 0.032 | 36.7% |
| 7 | mean_daily_social_screens_delta | 0.030 | 39.7% |
| 8 | mean_daily_screens_delta | 0.029 | 42.6% |
| 9 | lag_mean_daily_social_screens_delta | 0.028 | 45.4% |
| 10 | switches_per_screen | 0.028 | 48.1% |

Two CES-D features account for 23% of total gain. The remaining 37 behavioral features contribute 1–4% each. XGBoost uses 37/39 features (gender_mode_1, gender_mode_2 have zero gain).

### ElasticNet coefficients — worsening class

| Rank | Feature | Coef | Interpretation |
|---|---|---|---|
| 1 | person_mean_cesd | **+0.249** | Higher trait CES-D → worsening |
| 2 | mean_daily_social_screens_delta | **+0.202** | Increasing social screen time → worsening |
| 3 | prior_cesd | **-0.172** | Lower current CES-D → room to worsen |
| 4 | mean_daily_unique_apps_delta | +0.066 | More app variety → worsening |
| 5 | switches_per_screen | +0.059 | More fragmented attention → worsening |
| 6 | age | +0.057 | Older → worsening |
| 7 | mean_daily_screens | -0.055 | Less total usage → worsening |
| 8 | clip_dispersion_delta | +0.047 | Increasing content diversity → worsening |
| 9 | lag_mean_daily_social_screens | +0.036 | Prior social screen volume → worsening |
| 10 | lag_mean_daily_social_screens_delta | -0.036 | Prior social screen increase → less worsening |
| 11 | lag_clip_dispersion | +0.013 | Prior content diversity → worsening |

ElasticNet selects 11/39 features for worsening — highly sparse. Non-zero counts: improving=9, stable=7, worsening=11.

### ElasticNet coefficients — improving class

| Rank | Feature | Coef |
|---|---|---|
| 1 | prior_cesd | +0.219 |
| 2 | person_mean_cesd | -0.085 |
| 3 | lag_clip_dispersion_delta | +0.072 |
| 4 | lag_mean_daily_unique_apps | -0.034 |
| 5 | mean_daily_social_screens_delta | -0.029 |
| 6 | lag_mean_daily_social_screens_delta | +0.028 |
| 7 | clip_dispersion_delta | -0.021 |
| 8 | lag_mean_daily_unique_apps_delta | +0.017 |
| 9 | lag_mean_daily_screens_delta | +0.006 |

### ElasticNet coefficients — stable class

| Rank | Feature | Coef |
|---|---|---|
| 1 | mean_daily_switches_delta | -0.101 |
| 2 | mean_daily_screens | +0.057 |
| 3 | lag_mean_daily_screens_delta | -0.054 |
| 4 | lag_mean_daily_social_screens | -0.050 |
| 5 | lag_clip_dispersion | -0.040 |
| 6 | switches_per_screen | -0.035 |
| 7 | lag_mean_daily_screens | +0.013 |

### SHAP values — worsening class (XGBoost, true worsening cases)

**All true worsening (n=37):**

| Rank | Feature | Mean |SHAP| | Mean SHAP |
|---|---|---|---|
| 1 | person_mean_cesd | 0.621 | +0.354 |
| 2 | prior_cesd | 0.487 | +0.470 |
| 3 | mean_daily_social_screens_delta | 0.095 | -0.014 |
| 4 | lag_mean_daily_overnight_ratio_delta | 0.047 | +0.016 |
| 5 | lag_mean_daily_social_screens_delta | 0.047 | +0.002 |
| 6 | mean_daily_overnight_ratio | 0.040 | -0.009 |
| 7 | lag_mean_daily_unique_apps_delta | 0.038 | -0.007 |
| 8 | lag_mean_daily_switches_delta | 0.037 | +0.006 |
| 9 | mean_daily_switches_delta | 0.036 | -0.022 |
| 10 | active_day_ratio_delta | 0.035 | +0.001 |

**By transition type:**

| Rank | minimal→moderate (n=20) | minimal→severe (n=9) | moderate→severe (n=8) |
|---|---|---|---|
| 1 | prior_cesd (0.586) | person_mean_cesd (0.703) | person_mean_cesd (0.786) |
| 2 | person_mean_cesd (0.518) | prior_cesd (0.599) | prior_cesd (0.110) |
| 3 | social_screens_delta (0.086) | social_screens_delta (0.113) | social_screens_delta (0.097) |
| 4 | lag_switches_delta (0.050) | lag_social_screens_delta (0.057) | overnight_ratio (0.094) |
| 5 | lag_overnight_ratio_delta (0.042) | switches_delta (0.056) | lag_overnight_ratio_delta (0.076) |

Key pattern: For minimal→moderate, `prior_cesd` (boundary proximity) is the top driver. For moderate→severe, `person_mean_cesd` (person-level trait) dominates while `prior_cesd` drops to rank 2 with much lower SHAP. The behavioral feature `social_screens_delta` is consistently rank 3 across all transitions.

### Clip dispersion features — position in rankings

| Feature | XGB Gain Rank | EN Worsening Rank | SHAP Rank (all worsening) |
|---|---|---|---|
| lag_clip_dispersion_delta | 12 (0.027) | — (zeroed) | 24 (0.004) |
| clip_dispersion | 20 (0.021) | — (zeroed) | 32 (0.000) |
| clip_dispersion_delta | 23 (0.019) | 8 (+0.047) | 22 (0.009) |
| lag_clip_dispersion | 35 (0.011) | 11 (+0.013) | 29 (0.000) |

Clip dispersion features are selected by ElasticNet (rank 8 for worsening) but have near-zero SHAP contribution in XGBoost for worsening specifically. XGBoost splits on them (moderate gain) but those splits primarily affect improving/stable class boundaries.

---

## Worsening prediction analysis (XGBoost, 39-feature model)

### True worsening by transition type

| Transition | N | Caught | Missed | Sensitivity |
|---|---|---|---|---|
| minimal → moderate | 20 | 15 | 5 | 0.750 |
| minimal → severe | 9 | 9 | 0 | **1.000** |
| moderate → severe | 8 | 7 | 1 | 0.875 |
| **Total** | **37** | **31** | **6** | **0.838** |

Source: `models/posthoc/transition_detection.csv`

### Caught vs missed — worsening probabilities

> **Note**: The probability breakdowns below are from the initial training run (27 caught / 10 missed). The canonical model catches 31/37 (see transition detection above). Probability distributions are qualitatively similar — missed cases have substantially lower p(worsening) and tend to have low prior CES-D.

| Group | N | Mean p(wors) | Median | Min | Max |
|---|---|---|---|---|---|
| Caught | 27 | 0.754 | 0.805 | 0.444 | 0.932 |
| Missed | 10 | 0.325 | 0.325 | 0.185 | 0.450 |

### SHAP gap — caught vs missed

| Feature | Caught SHAP | Missed SHAP | Gap |
|---|---|---|---|
| **person_mean_cesd** | **+0.582** | **-0.260** | **+0.842** |
| prior_cesd | +0.533 | +0.301 | +0.232 |
| social_screens_delta | +0.008 | -0.073 | +0.081 |
| switches_delta | -0.005 | -0.070 | +0.065 |

person_mean_cesd is the dominant differentiator: it pushes caught cases toward worsening (+0.58) but pushes missed cases away (-0.26). Missed cases are people whose training-period CES-D average was low.

### Sensitivity by prior CES-D

| Prior CES-D | N worsening | Caught | Sensitivity | False Alarms |
|---|---|---|---|---|
| 0–8 | 5 | 2 | **0.400** | 5 |
| 8–12 | 4 | 2 | **0.500** | 14 |
| 12–16 | 20 | 20 | **1.000** | 28 |
| 16–24 | 8 | 7 | **0.875** | 9 |

Source: `models/posthoc/sensitivity_by_cesd.csv`

Model performs poorly for individuals starting in the low minimal range (CES-D < 12). Near-threshold individuals (12–16) are detected perfectly.

### Threshold sensitivity

| Threshold | Sensitivity | PPV | TP | FP | FN |
|---|---|---|---|---|---|
| 0.20 | 0.946 | 0.235 | 35 | 114 | 2 |
| 0.25 | 0.946 | 0.259 | 35 | 100 | 2 |
| 0.30 | 0.865 | 0.286 | 32 | 80 | 5 |
| 0.35 | 0.865 | 0.323 | 32 | 67 | 5 |
| 0.40 | 0.811 | 0.333 | 30 | 60 | 7 |
| 0.45 | 0.703 | 0.317 | 26 | 56 | 11 |
| 0.50 | 0.703 | 0.347 | 26 | 49 | 11 |

### False alarm analysis (44 minimal→minimal false positives)

False alarms are **boundary-proximity cases**, not spread across the range:

| Prior CES-D | N false alarms | % |
|---|---|---|
| 0–5 | 0 | 0% |
| 5–8 | 4 | 9% |
| 8–11 | 11 | 25% |
| 11–14 | 16 | 36% |
| 14–15 | 13 | 30% |

Mean prior CES-D of false alarms = **11.5** vs **4.3** for correctly-predicted stable. These are people near the minimal/moderate boundary (CES-D 8–15) who the model flagged as at-risk of crossing 16 — they didn't, but they were plausible candidates.

### Sequential flagging

Requiring two consecutive worsening predictions does **not** improve PPV:
- Single-flag: TP=27, FP=53, PPV=0.338
- Consecutive-flag: TP=3, FP=31, PPV=0.088

True worsening events are isolated spikes (boundary crossing in one period, then stabilizing), so requiring two flags filters out true positives more than false positives.

### Calibration

The model is **substantially overconfident** for worsening due to class-weight balancing:

| Predicted p(worsening) | Actual rate | Gap |
|---|---|---|
| 0.15 | 0.04 | -0.11 |
| 0.34 | 0.09 | -0.25 |
| 0.65 | 0.31 | -0.33 |
| 0.84 | 0.33 | -0.51 |

Brier score: 0.105. ECE: 0.142. Probabilities need recalibration (Platt scaling or isotonic regression) before being used for threshold-setting.

---

## EXPERIMENT SET 1 — Lag decomposition

### Purpose
Decompose how much of the lag contribution comes from clinical lag features vs behavioral lag features.

### Conditions

**1A — Clinical lag only:** 21 base + lag_prior_cesd + lag_cesd_delta + person_mean_cesd = 24 features
**1B — Behavioral lag only:** 21 base + 17 behavioral lag + person_mean_cesd = 39 features (recommended model)
**1C — Full lag (reference):** 21 base + 19 all lag + person_mean_cesd = 41 features

### Results — sev_crossing label

| Condition | N feat | XGB AUC | XGB BalAcc | XGB Sens-W | EN AUC | EN BalAcc | EN Sens-W |
|---|---|---|---|---|---|---|---|
| 1A — Clinical lag only | 24 | 0.906 | 0.789 | 0.730 | 0.821 | 0.674 | 0.676 |
| 1B — Behavioral lag only | 39 | 0.915 | 0.792 | 0.730 | 0.820 | 0.668 | 0.676 |
| 1C — Full lag (ref) | 41 | 0.914 | 0.784 | 0.703 | 0.821 | 0.663 | 0.676 |

### Results — personal_sd label

| Condition | N feat | XGB AUC | XGB BalAcc | XGB Sens-W | EN AUC | EN BalAcc | EN Sens-W |
|---|---|---|---|---|---|---|---|
| 1A — Clinical lag only | 24 | 0.757 | 0.566 | 0.488 | 0.765 | 0.655 | 0.659 |
| 1B — Behavioral lag only | 39 | 0.755 | 0.622 | 0.488 | 0.759 | 0.625 | 0.585 |
| 1C — Full lag (ref) | 41 | 0.754 | 0.590 | 0.488 | 0.755 | 0.618 | 0.537 |

### Interpretation
- Clinical lag alone (1A) captures nearly all of the lag benefit for AUC (0.906 vs 0.914 full).
- Behavioral lag alone (1B) matches the full model AUC (0.915 vs 0.914).
- Clinical and behavioral lag are **substitutable, not additive** — either alone recovers the full lag benefit.
- 1B is preferred: same performance as 1C, 2 fewer features, no clinical lag dependency.

---

## EXPERIMENT SET 2 — CES-D staleness

### Purpose
Simulate real-world deployment where a clinician's CES-D assessment is not always recent.

### Setup
"Staleness" means substituting the current prior_cesd with an older CES-D value. Behavioral features are ALWAYS from the current period — only the clinical CES-D anchor changes.

### Conditions

**2A — Current (k=0, reference)**
**2B — Stale by 1 period (k=2, 4 weeks ago)**
**2C — Stale by 2 periods (k=4, 8 weeks ago)**
**2D — No CES-D at all**

**Correction from original spec:** person_mean_cesd is CES-D-derived, so 2D is split into sub-conditions:
- **2D-a:** Remove prior_cesd; KEEP person_mean_cesd (38 features)
- **2D-b:** Remove prior_cesd AND person_mean_cesd (37 features)
- **2D-c:** Behavioral features only, no lag, no CES-D (20 features)

Note: lag_prior_cesd and lag_cesd_delta are already excluded in the 39-feature base model.

### Missing value handling
For stale conditions, observations where t-k doesn't exist use the current prior_cesd as fallback.

### Results — sev_crossing label (39-feature model, per-condition grid-searched params)

Note: These results use the deployment scenario framework — the same 39-feat model with degraded test inputs (not separately grid-searched models). Updated numbers from `scripts/deployment_scenarios.py`.

| Condition | N test | XGB AUC | XGB BalAcc | XGB Sens-W | XGB PPV-W |
|---|---|---|---|---|---|
| 2A — Current (ref) | 411 | 0.906 | 0.834 | 0.838 | 0.356 |
| 2B — Stale 4 weeks | 411 | 0.735 | 0.565 | 0.514 | 0.232 |
| 2C — Stale 8 weeks | 411 | 0.702 | 0.507 | 0.432 | 0.188 |
| 2D — No fresh CES-D (prior_cesd = pop_mean) | 411 | 0.666 | 0.506 | 0.892 | 0.170 |

### Interpretation
- CES-D staleness degrades performance substantially: 4-week stale ΔAUC = -0.171, 8-week stale ΔAUC = -0.204.
- No fresh CES-D inflates Sens-W (0.892) because the model over-predicts worsening without a real clinical anchor, but PPV-W drops to 0.170.
- person_mean_cesd partially compensates for missing prior_cesd but cannot fully substitute for current clinical state.
- Without any CES-D-derived features, behavioral features alone perform near chance (see bootstrap ablation: base features AUC 0.876 includes prior_cesd).

---

## EXPERIMENT SET 3 — Onboarding scenario

### Purpose
Test whether a single intake CES-D score can serve as a stable clinical anchor for all subsequent predictions.

### Setup
- Replace prior_cesd with each person's first-observation CES-D score (constant)
- Behavioral lag features from current period (17 behavioral lags, no clinical lags)
- person_mean_cesd = intake CES-D (in 3A/3B) or training-derived mean (in 3C)

### Results — sev_crossing label (per-condition grid-searched params)

Updated from deployment scenario analysis — same 39-feat model with intake CES-D substituted for prior_cesd and pmcesd at test time.

| Condition | N test | XGB AUC | XGB BalAcc | XGB Sens-W | XGB PPV-W |
|---|---|---|---|---|---|
| Onboarding (prior_cesd & pmcesd = intake CES-D) | 411 | 0.670 | 0.458 | 0.297 | 0.159 |

### Interpretation
- A single intake CES-D gives XGBoost AUC 0.670, well above population baseline (0.500) and near demographics-only (0.720).
- Still a large gap vs full model (0.906), confirming that accumulating person-specific history is critical.
- The onboarding scenario performs worse than stale CES-D (4wk: 0.735, 8wk: 0.702), suggesting even outdated CES-D is more informative than using only the very first measurement.

---

## EXPERIMENT SET 4 — Deployment baselines

### Purpose
Establish true floor baselines for realistic deployment where no individual-level data is available yet.

### Conditions

**4A — Population-mean CES-D only (1 feature)**
Every person receives the training-set population mean CES-D (12.51) as their prior_cesd. No behavioral features, no person_mean_cesd. Tests: what can you predict knowing only the population average?

**4B — Demographics only (3 features: age, gender_mode_1, gender_mode_2)**
Only intake-form demographics. No behavioral data, no CES-D score. Tests: what can you predict from an intake form alone?

### Results — sev_crossing label (XGBoost, test set, grid-searched params)

| Condition | N feat | N test | XGB AUC | XGB BalAcc | XGB Sens-W | XGB PPV-W |
|---|---|---|---|---|---|---|
| 4A — Pop baseline (predict all stable) | 0 | 411 | 0.500 | 0.333 | 0.000 | 0.000 |
| 4B — Demographics only (grid-searched) | 3 | 411 | 0.720 | 0.463 | 0.459 | 0.173 |

### Interpretation
- **Population baseline** computes actual metrics: F1-macro = 0.297 (non-zero for the stable class), AUC = 0.500 (chance). This is the absolute floor.
- **Demographics alone (AUC 0.720)** perform surprisingly well with grid-searched XGBoost, detecting 46% of worsening cases. Age and gender proxy for where people sit on the CES-D severity scale.

### Full deployment ladder (sev_crossing, XGBoost, per-condition grid-searched params)

All scenarios except intake form and population baseline use the **same 39-feature model** with degraded test inputs. Cold start uses leave-group-out CV by PID (5 folds, train/val/test split preserved).

| Stage | What you know | N feat | AUC | BalAcc | F1-macro | Sens-W | PPV-W |
|---|---|---|---|---|---|---|---|
| Population baseline | Predict all stable | 0 | 0.500 | 0.333 | 0.297 | 0.000 | 0.000 |
| Intake form only | Age + gender | 3 | 0.720 | 0.463 | 0.401 | 0.459 | 0.173 |
| Onboarding | Intake CES-D as anchor | 39 | 0.670 | 0.458 | 0.420 | 0.297 | 0.159 |
| Stale 4 weeks | prior_cesd from t-1 | 39 | 0.735 | 0.565 | 0.491 | 0.514 | 0.232 |
| Stale 8 weeks | prior_cesd from t-2 | 39 | 0.702 | 0.507 | 0.451 | 0.432 | 0.188 |
| No fresh CES-D | prior_cesd = pop_mean | 39 | 0.666 | 0.506 | 0.348 | 0.892 | 0.170 |
| Cold start (held-out) | Unseen person, pmcesd=pop_mean | 39 | 0.821 | 0.720 | 0.534 | 0.569 | 0.224 |
| **Full model** | **All features, known person** | **39** | **0.906** | **0.834** | **0.686** | **0.838** | **0.356** |

**Key insights:**
- Intake form → onboarding: ΔAUC = -0.050 (onboarding anchor helps less than demographics alone for XGBoost)
- Cold start → full model: ΔAUC = -0.085 (model generalizes reasonably to unseen people)
- Stale 4wk degradation vs full: ΔAUC = -0.171 (stale CES-D degrades substantially)
- No fresh CES-D inflates Sens-W to 0.892 but PPV-W drops to 0.170 — model over-predicts worsening without current CES-D

---

## Note on label validity in deployment scenarios

Labels (sev_crossing, personal_sd) are **ground truth outcomes** — what we predict, not input features.
- **sev_crossing labels remain valid** in all experiments, including no-CES-D and onboarding scenarios.
- **personal_sd labels have a limitation** in onboarding: computing person-specific thresholds requires CES-D history that wouldn't be available.

---

## Baseline ablation results (for reference)

| Condition | N feat | XGB AUC | EN AUC | Label |
|---|---|---|---|---|
| Full model (old, with static lags) | 44 | 0.914 | 0.795 | sev_crossing |
| Full model (EN + pmcesd) | 44 | — | 0.821 | sev_crossing |
| Behaviors + prior_cesd, no lag | 21 | 0.872 | 0.751 | sev_crossing |
| Prior CES-D only | 1 | 0.877 | 0.760 | sev_crossing |
| Behaviors only (no lag, no CES-D) | 20 | 0.616 | 0.544 | sev_crossing |
| Base + dev + prior_cesd | 29 | 0.867 | 0.756 | sev_crossing |

Note: The original "full model" (44 features) included redundant lag features for static demographics (lag_age, lag_gender_mode_1, lag_gender_mode_2). These are identical to the current-period values and have been removed in the cleaned 39-feature model. Additionally, the original ElasticNet "full model" AUC of 0.795 was trained without person_mean_cesd; with person_mean_cesd the ElasticNet achieves 0.821.

---

## Summary table of all conditions

### sev_crossing label

| Exp | Condition | N feat | N test | XGB AUC | XGB BalAcc | XGB Sens-W | EN AUC | EN BalAcc | EN Sens-W |
|---|---|---|---|---|---|---|---|---|---|
| **best** | **Behavioral lag + pmcesd (grid-searched)** | **39** | **411** | **0.906** | **0.834** | **0.838** | **0.829** | **0.691** | **0.730** |
| ref | Full model (old, with static lags) | 44 | 411 | 0.914 | 0.784 | 0.703 | 0.795 | 0.634 | 0.459 |
| ref | Full model (EN + pmcesd) | 44 | 411 | — | — | — | 0.821 | 0.663 | 0.676 |
| ref | Behaviors + prior_cesd, no lag | 21 | 411 | 0.872 | — | — | 0.751 | — | — |
| ref | Prior CES-D only | 1 | 411 | 0.877 | — | — | 0.760 | — | — |
| ref | Behaviors only (no CES-D) | 20 | 411 | 0.616 | — | — | 0.544 | — | — |
| ref | Base + dev + prior_cesd | 29 | 411 | 0.867 | — | — | 0.756 | — | — |
| 1A | Clinical lag only | 24 | 411 | 0.906 | 0.789 | 0.730 | 0.821 | 0.674 | 0.676 |
| 2B | Stale 4 weeks | 44 | 405 | 0.776 | 0.496 | 0.378 | — | — | — |
| 2C | Stale 8 weeks | 44 | 402 | 0.756 | 0.478 | 0.378 | — | — | — |
| 2D-a | No CES-D + keep pmcesd | 41 | 411 | 0.781 | 0.529 | 0.541 | 0.741 | 0.401 | 0.270 |
| 2D-b | No CES-D strict | 40 | 411 | 0.584 | 0.373 | 0.189 | 0.523 | 0.320 | 0.270 |
| 2D-c | Behavioral only, no lag | 20 | 411 | 0.610 | 0.383 | 0.297 | — | — | — |
| 3A | Onboarding + behav lag | 39 | 411 | 0.753 | 0.496 | 0.459 | — | — | — |
| 3B | Onboarding, no lag | 22 | 411 | 0.753 | 0.493 | 0.541 | — | — | — |
| 3C | Onboarding replaces prior_cesd | 39 | 411 | 0.780 | 0.513 | 0.514 | — | — | — |
| 4A | Pop-mean CES-D only | 1 | 411 | 0.500 | 0.333 | 0.000 | 0.500 | 0.333 | 0.000 |
| 4B | Demographics only (age+gender) | 3 | 411 | 0.687 | 0.416 | 0.243 | 0.603 | 0.394 | 0.000 |

### personal_sd label (Experiment 1 only)

| Exp | Condition | N feat | N test | XGB AUC | XGB BalAcc | XGB Sens-W | EN AUC | EN BalAcc | EN Sens-W |
|---|---|---|---|---|---|---|---|---|---|
| ref | Full model | 41 | 411 | 0.754 | 0.590 | 0.488 | 0.755 | 0.618 | 0.537 |
| 1A | Clinical lag only | 24 | 411 | 0.757 | 0.566 | 0.488 | 0.765 | 0.655 | 0.659 |
| 1B | Behavioral lag only | 39 | 411 | 0.755 | 0.622 | 0.488 | 0.759 | 0.625 | 0.585 |

---

## Key takeaways

1. **CES-D anchoring is essential**: Prior CES-D alone (AUC 0.872 XGBoost) outperforms all behavioral features combined. Behavioral features cannot independently predict mood direction changes.

2. **Lag features are valuable but substitutable**: Clinical lag and behavioral lag each independently recover the full lag benefit. They are redundant with each other, not additive. Behavioral lag only (39 features) is the recommended model.

3. **person_mean_cesd is the single most impactful feature addition**: Adding pmcesd to the 38-feature model produces the only statistically significant improvement across all labels (bootstrap paired test p < 0.01 for all 4 models on sev_crossing). ΔAUC = +0.031 [+0.008, +0.057] for XGBoost; ΔPPV-W = +0.124 [+0.059, +0.195].

4. **Stale CES-D degrades performance sharply**: 4-week stale ΔAUC = -0.171, 8-week ΔAUC = -0.204 vs full model. No fresh CES-D inflates Sens-W (0.892) but PPV-W drops to 0.170.

5. **Cold start is viable**: Leave-group-out CV (held-out PIDs) gives XGBoost AUC 0.821 — a ΔAUC of -0.085 vs full model. The model generalizes reasonably to unseen people.

6. **Onboarding provides limited value**: A single intake CES-D gives AUC 0.670, better than baseline but worse than stale CES-D. Accumulated person history is critical.

7. **Model weaknesses**: Sensitivity is poor for low-CES-D individuals. The model is substantially overconfident (ECE = 0.14) and needs probability recalibration. Sequential flagging does not improve PPV.

8. **personal_sd is harder to predict**: Best AUC 0.759 (ElasticNet) vs 0.906 for sev_crossing, reflecting the difficulty of predicting idiosyncratic within-person thresholds. Bootstrap analysis confirms pmcesd significantly improves personal_sd as well (3/4 models significant).

9. **PPV-W context**: Full model XGBoost PPV-W = 0.356 (1 true positive per ~2 false alarms). PPV-W improves with pmcesd across all labels but remains the most challenging metric due to low base rate of worsening.
