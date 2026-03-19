# Deployment Scenario Results — sev_crossing label


## Design

All model-based scenarios (except Intake form only) use the same 39-feature XGBoost/LightGBM/ElasticNet/SVM models with per-condition grid-searched hyperparameters from bootstrap analysis. Scenarios differ only in what information is available at test time. 95% CIs computed via 1000-iteration percentile bootstrap resampling of the test set (cold start uses fold-level SD instead).

## Scenario Descriptions

1. **Population baseline** — No model. Predict "stable" for every observation. This is the floor: any useful model must beat it. Included to anchor the scale.

2. **Revert-to-person-mean** — Rule-based, no ML model. For each person, compare their current CES-D severity band to the severity band of their training-period mean CES-D. If the mean is in a higher band → predict worsening; lower → improving; same → stable. Tests whether simple regression-to-the-mean explains transitions. A strong baseline (AUC=0.750) because severity changes are partly mean-reverting.

3. **Last-change-only** — Rule-based, no ML model. Predict that the previous period's severity transition repeats (momentum hypothesis). If last period crossed up → predict worsening again. Tests whether transitions have momentum. Very weak (AUC=0.556, Sens-W=0.054), confirming that CES-D changes have near-zero autocorrelation at the severity-crossing level.

4. **Intake form only** — ML model trained on 3 demographic features (age, gender_mode_1, gender_mode_2) with its own grid-searched hyperparameters. Simulates a scenario where only intake demographics are available — no CES-D history, no behavioral data. Tests the floor of ML-based prediction.

5. **Onboarding** — Full 39-feature model, but at test time prior_cesd and person_mean_cesd are both set to the person's first-ever CES-D score. Simulates a "frozen CES-D" deployment: behavioral features stream continuously from Screenome as usual, but the CES-D anchor never updates past the initial intake survey. Equivalent to asking: what if we only ever administered the CES-D once? AUC = 0.670 — a meaningful drop from the full model (0.906), indicating the CES-D anchor needs to update for the behavioral features to be useful.

6. **Stale 4 weeks** — Full 39-feature model, but prior_cesd comes from 1 biweekly period ago (i.e., the previous period's prior_cesd, not the current one). Simulates a missed assessment: behavioral data is current but the most recent CES-D is 4 weeks old. Tests graceful degradation when self-report lapses.

7. **Stale 8 weeks** — Same as Stale 4 weeks, but prior_cesd comes from 2 periods ago (8 weeks old). Tests how far the model degrades with increasingly outdated clinical self-report.

8. **No fresh CES-D** — Full 39-feature model, but prior_cesd is replaced with the population mean (~12, minimal severity band). Simulates a pure passive-monitoring scenario: behavioral features stream from the screenome, but the person has not completed any recent CES-D. person_mean_cesd is still available from historical data. **Caution: Sens-W is artificially inflated (0.892) and should not be interpreted as genuine sensitivity.** When prior_cesd is forced to the population mean (~12), it falls below most people's person_mean_cesd, making the gap (pmcesd − prior_cesd) large and positive for nearly everyone. The model learned during training that this pattern signals "currently below usual level → predict worsening," so it predicts worsening for almost all observations. This catches nearly every true worsening case by chance (high recall) but generates massive false positives — PPV-W of 0.170 means only 17% of predicted worsening cases are real. AUC (0.666) is a more honest summary of performance here.

9. **Cold start** — Full 39-feature model evaluated via repeated leave-group-out CV (5 repeats × 5 folds = 25 evaluations). Each fold holds out ~20% of persons entirely — these persons are never seen during training. person_mean_cesd is set to the population mean for held-out persons. Tests generalization to completely new individuals with no historical CES-D profile.

10. **Full model** — All 39 features with current information. The system has current behavioral data, current CES-D (prior_cesd from the most recent period), and a reliable person_mean_cesd computed from the person's training history. This is the ceiling for known, active participants.

## Statistical Baselines (B0–B4)

Reference performance for zero-information and simple-rule predictors on the same test set (n=411). Full results in `classification/reports/baseline_results.md`.

| Baseline | Rule | AUC | BalAcc | Sens-W | PPV-W |
|---|---|---|---|---|---|
| B0 No Change | Predict all stable | 0.500 | 0.333 | 0.000 | 0.000 |
| B1 Population Mean | Predict majority class (stable) | 0.500 | 0.333 | 0.000 | 0.000 |
| B2 LVCF | Repeat previous period's class | 0.557 | 0.336 | 0.054 | 0.053 |
| B3 Person Modal | Person's most common training class | 0.499 | 0.327 | 0.000 | 0.000 |
| B4 Regression to Mean | Direction toward person's training-mean severity band | 0.750 | 0.674 | 0.541 | 0.408 |

B4 is equivalent to the rule-based **Revert-to-person-mean** scenario in the deployment ladder. It is the strongest zero-model baseline, but it requires the person's training-period mean CES-D — a quantity computed from many historical observations. This makes B4 a well-resourced rule that is **not directly comparable to degraded-information scenarios** such as Onboarding, where that history is unavailable. B0–B3 are the appropriate lower bounds for degraded scenarios; B4 is only a fair comparison for the full model and other scenarios where person history is available.

## Deployment Ladder (XGBoost)


| Stage | What you know | N feat | AUC [95% CI] | BalAcc [95% CI] | Sens-W [95% CI] | PPV-W [95% CI] |
|---|---|---|---|---|---|---|
| Population baseline | Predict all stable | 0 | 0.500 [0.500, 0.500] | 0.333 [0.333, 0.333] | 0.000 [0.000, 0.000] | 0.000 |
| Revert-to-person-mean | Rule: predict severity moves toward person mean | 0 | 0.750 [0.697, 0.804] | 0.674 [0.601, 0.743] | 0.541 [0.382, 0.706] | 0.408 [0.267, 0.551] |
| Last-change-only | Rule: repeat last period's severity transition | 0 | 0.556 [0.527, 0.588] | 0.335 [0.301, 0.375] | 0.054 [0.000, 0.139] | 0.053 [0.000, 0.133] |
| Intake form only | Age + gender (3 feat) | 3 | 0.720 [0.667, 0.766] | 0.463 [0.385, 0.532] | 0.459 [0.289, 0.617] | 0.173 [0.103, 0.252] |
| Onboarding | 39-feat, prior_cesd & pmcesd = intake CES-D | 39 | 0.670 [0.613, 0.725] | 0.458 [0.385, 0.530] | 0.297 [0.151, 0.447] | 0.159 [0.074, 0.246] |
| Stale 4 weeks | 39-feat, prior_cesd from t-1 | 39 | 0.735 [0.682, 0.786] | 0.565 [0.489, 0.644] | 0.514 [0.333, 0.667] | 0.232 [0.143, 0.321] |
| Stale 8 weeks | 39-feat, prior_cesd from t-2 | 39 | 0.702 [0.647, 0.754] | 0.507 [0.434, 0.581] | 0.432 [0.276, 0.607] | 0.188 [0.106, 0.276] |
| No fresh CES-D | 39-feat, prior_cesd = pop_mean | 39 | 0.666 [0.627, 0.707] | 0.506 [0.467, 0.541] | 0.892 [0.781, 0.976] | 0.170 [0.118, 0.224] |
| Cold start | Leave-group-out (5x5 folds), train/val/test split, pmcesd=pop_mean | 39 | 0.821 [0.725, 0.917] | 0.720 [0.554, 0.886] | 0.569 [0.059, 1.078] | 0.224 [-0.037, 0.484] |
| Full model | All 39 features, known person | 39 | 0.906 [0.881, 0.929] | 0.834 [0.784, 0.882] | 0.838 [0.707, 0.947] | 0.356 [0.253, 0.464] |

## Full Results (All Models)


| Scenario | Model | N feat | AUC [95% CI] | BalAcc [95% CI] | Sens-W [95% CI] | PPV-W [95% CI] |
|---|---|---|---|---|---|---|
| Population baseline | ElasticNet | 0 | 0.500 [0.500, 0.500] | 0.333 [0.333, 0.333] | 0.000 [0.000, 0.000] | 0.000 |
| Population baseline | XGBoost | 0 | 0.500 [0.500, 0.500] | 0.333 [0.333, 0.333] | 0.000 [0.000, 0.000] | 0.000 |
| Population baseline | LightGBM | 0 | 0.500 [0.500, 0.500] | 0.333 [0.333, 0.333] | 0.000 [0.000, 0.000] | 0.000 |
| Population baseline | SVM | 0 | 0.500 [0.500, 0.500] | 0.333 [0.333, 0.333] | 0.000 [0.000, 0.000] | 0.000 |
| Revert-to-person-mean | ElasticNet | 0 | 0.750 [0.697, 0.804] | 0.674 [0.601, 0.743] | 0.541 [0.382, 0.706] | 0.408 [0.267, 0.551] |
| Revert-to-person-mean | XGBoost | 0 | 0.750 [0.697, 0.804] | 0.674 [0.601, 0.743] | 0.541 [0.382, 0.706] | 0.408 [0.267, 0.551] |
| Revert-to-person-mean | LightGBM | 0 | 0.750 [0.697, 0.804] | 0.674 [0.601, 0.743] | 0.541 [0.382, 0.706] | 0.408 [0.267, 0.551] |
| Revert-to-person-mean | SVM | 0 | 0.750 [0.697, 0.804] | 0.674 [0.601, 0.743] | 0.541 [0.382, 0.706] | 0.408 [0.267, 0.551] |
| Last-change-only | ElasticNet | 0 | 0.556 [0.527, 0.588] | 0.335 [0.301, 0.375] | 0.054 [0.000, 0.139] | 0.053 [0.000, 0.133] |
| Last-change-only | XGBoost | 0 | 0.556 [0.527, 0.588] | 0.335 [0.301, 0.375] | 0.054 [0.000, 0.139] | 0.053 [0.000, 0.133] |
| Last-change-only | LightGBM | 0 | 0.556 [0.527, 0.588] | 0.335 [0.301, 0.375] | 0.054 [0.000, 0.139] | 0.053 [0.000, 0.133] |
| Last-change-only | SVM | 0 | 0.556 [0.527, 0.588] | 0.335 [0.301, 0.375] | 0.054 [0.000, 0.139] | 0.053 [0.000, 0.133] |
| Intake form only | ElasticNet | 3 | 0.603 [0.541, 0.665] | 0.396 [0.341, 0.450] | 0.000 [0.000, 0.000] | 0.000 |
| Intake form only | XGBoost | 3 | 0.720 [0.667, 0.766] | 0.463 [0.385, 0.532] | 0.459 [0.289, 0.617] | 0.173 [0.103, 0.252] |
| Intake form only | LightGBM | 3 | 0.719 [0.666, 0.766] | 0.461 [0.382, 0.535] | 0.432 [0.273, 0.600] | 0.182 [0.105, 0.267] |
| Intake form only | SVM | 3 | 0.640 [0.581, 0.701] | 0.375 [0.307, 0.442] | 0.270 [0.132, 0.429] | 0.083 [0.037, 0.130] |
| Onboarding | ElasticNet | 39 | 0.720 [0.676, 0.764] | 0.455 [0.398, 0.516] | 0.054 [0.000, 0.143] | 0.200 [0.000, 0.500] |
| Onboarding | XGBoost | 39 | 0.670 [0.613, 0.725] | 0.458 [0.385, 0.530] | 0.297 [0.151, 0.447] | 0.159 [0.074, 0.246] |
| Onboarding | LightGBM | 39 | 0.629 [0.566, 0.686] | 0.448 [0.380, 0.518] | 0.297 [0.151, 0.447] | 0.155 [0.070, 0.240] |
| Onboarding | SVM | 39 | 0.733 [0.685, 0.780] | 0.505 [0.437, 0.572] | 0.243 [0.111, 0.385] | 0.196 [0.089, 0.310] |
| Stale 4 weeks | ElasticNet | 39 | 0.692 [0.640, 0.744] | 0.518 [0.445, 0.594] | 0.486 [0.324, 0.650] | 0.186 [0.110, 0.268] |
| Stale 4 weeks | XGBoost | 39 | 0.735 [0.682, 0.786] | 0.565 [0.489, 0.644] | 0.514 [0.333, 0.667] | 0.232 [0.143, 0.321] |
| Stale 4 weeks | LightGBM | 39 | 0.721 [0.668, 0.775] | 0.543 [0.468, 0.620] | 0.514 [0.333, 0.667] | 0.211 [0.128, 0.290] |
| Stale 4 weeks | SVM | 39 | 0.731 [0.680, 0.781] | 0.509 [0.436, 0.582] | 0.432 [0.276, 0.600] | 0.198 [0.111, 0.278] |
| Stale 8 weeks | ElasticNet | 39 | 0.643 [0.583, 0.700] | 0.459 [0.386, 0.534] | 0.405 [0.242, 0.583] | 0.155 [0.085, 0.228] |
| Stale 8 weeks | XGBoost | 39 | 0.702 [0.647, 0.754] | 0.507 [0.434, 0.581] | 0.432 [0.276, 0.607] | 0.188 [0.106, 0.276] |
| Stale 8 weeks | LightGBM | 39 | 0.683 [0.620, 0.739] | 0.508 [0.433, 0.584] | 0.459 [0.300, 0.641] | 0.187 [0.113, 0.269] |
| Stale 8 weeks | SVM | 39 | 0.694 [0.640, 0.744] | 0.472 [0.398, 0.545] | 0.351 [0.210, 0.526] | 0.171 [0.090, 0.257] |
| No fresh CES-D | ElasticNet | 39 | 0.587 [0.558, 0.618] | 0.489 [0.443, 0.540] | 0.730 [0.595, 0.867] | 0.186 [0.121, 0.253] |
| No fresh CES-D | XGBoost | 39 | 0.666 [0.627, 0.707] | 0.506 [0.467, 0.541] | 0.892 [0.781, 0.976] | 0.170 [0.118, 0.224] |
| No fresh CES-D | LightGBM | 39 | 0.672 [0.634, 0.711] | 0.506 [0.467, 0.541] | 0.892 [0.781, 0.976] | 0.170 [0.118, 0.224] |
| No fresh CES-D | SVM | 39 | 0.618 [0.579, 0.656] | 0.441 [0.386, 0.503] | 0.622 [0.463, 0.793] | 0.163 [0.104, 0.226] |
| Cold start | ElasticNet | 39 | 0.573 ±0.040 | 0.424 ±0.065 | 0.210 ±0.181 | 0.032 ±0.020 |
| Cold start | XGBoost | 39 | 0.821 ±0.049 | 0.720 ±0.085 | 0.569 ±0.260 | 0.224 ±0.133 |
| Cold start | LightGBM | 39 | 0.800 ±0.059 | 0.676 ±0.066 | 0.549 ±0.305 | 0.174 ±0.131 |
| Cold start | SVM | 39 | 0.672 ±0.074 | 0.501 ±0.086 | 0.290 ±0.257 | 0.072 ±0.059 |
| Full model | ElasticNet | 39 | 0.829 [0.792, 0.867] | 0.691 [0.624, 0.762] | 0.730 [0.588, 0.872] | 0.248 [0.168, 0.333] |
| Full model | XGBoost | 39 | 0.906 [0.881, 0.929] | 0.834 [0.784, 0.882] | 0.838 [0.707, 0.947] | 0.356 [0.253, 0.464] |
| Full model | LightGBM | 39 | 0.901 [0.876, 0.925] | 0.842 [0.798, 0.890] | 0.865 [0.744, 0.970] | 0.344 [0.247, 0.439] |
| Full model | SVM | 39 | 0.841 [0.806, 0.876] | 0.696 [0.628, 0.766] | 0.649 [0.488, 0.806] | 0.304 [0.203, 0.405] |

## Cold Start — 5×5-Fold Summary (25 evaluations)


### Summary (mean ± SD across all folds)


| Model | AUC | BalAcc | F1-macro | Sens-W | PPV-W |
|---|---|---|---|---|---|
| ElasticNet | 0.573±0.040 | 0.424±0.065 | 0.294±0.040 | 0.210±0.181 | 0.032±0.020 |
| XGBoost | 0.821±0.049 | 0.720±0.085 | 0.534±0.079 | 0.569±0.260 | 0.224±0.133 |
| LightGBM | 0.800±0.059 | 0.676±0.066 | 0.469±0.105 | 0.549±0.305 | 0.174±0.131 |
| SVM | 0.672±0.074 | 0.501±0.086 | 0.377±0.055 | 0.290±0.257 | 0.072±0.059 |

### Per-Fold Details


**ElasticNet**:

| Repeat | Fold | Held PIDs | N obs | AUC | BalAcc | F1-macro | Sens-W | PPV-W |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 20 | 89 | 0.562 | 0.439 | 0.322 | 0.200 | 0.044 |
| 0 | 1 | 19 | 81 | 0.569 | 0.480 | 0.278 | 0.333 | 0.044 |
| 0 | 2 | 19 | 75 | 0.540 | 0.321 | 0.218 | 0.000 | 0.000 |
| 0 | 3 | 19 | 84 | 0.607 | 0.395 | 0.330 | 0.154 | 0.059 |
| 0 | 4 | 19 | 82 | 0.577 | 0.380 | 0.287 | 0.000 | 0.000 |
| 1 | 0 | 20 | 86 | 0.538 | 0.376 | 0.220 | 0.167 | 0.019 |
| 1 | 1 | 19 | 84 | 0.493 | 0.490 | 0.249 | 0.400 | 0.034 |
| 1 | 2 | 19 | 82 | 0.545 | 0.402 | 0.329 | 0.333 | 0.035 |
| 1 | 3 | 19 | 79 | 0.601 | 0.402 | 0.336 | 0.182 | 0.059 |
| 1 | 4 | 19 | 80 | 0.670 | 0.458 | 0.336 | 0.111 | 0.043 |
| 2 | 0 | 20 | 85 | 0.519 | 0.361 | 0.248 | 0.250 | 0.016 |
| 2 | 1 | 19 | 83 | 0.633 | 0.463 | 0.332 | 0.167 | 0.032 |
| 2 | 2 | 19 | 81 | 0.592 | 0.444 | 0.339 | 0.100 | 0.028 |
| 2 | 3 | 19 | 79 | 0.520 | 0.376 | 0.317 | 0.375 | 0.064 |
| 2 | 4 | 19 | 83 | 0.587 | 0.395 | 0.255 | 0.111 | 0.027 |
| 3 | 0 | 20 | 89 | 0.547 | 0.338 | 0.257 | 0.125 | 0.018 |
| 3 | 1 | 19 | 80 | 0.552 | 0.487 | 0.263 | 0.600 | 0.060 |
| 3 | 2 | 19 | 80 | 0.585 | 0.369 | 0.292 | 0.000 | 0.000 |
| 3 | 3 | 19 | 81 | 0.637 | 0.474 | 0.355 | 0.182 | 0.057 |
| 3 | 4 | 19 | 81 | 0.545 | 0.483 | 0.290 | 0.167 | 0.023 |
| 4 | 0 | 20 | 82 | 0.579 | 0.422 | 0.318 | 0.000 | 0.000 |
| 4 | 1 | 19 | 86 | 0.576 | 0.360 | 0.294 | 0.100 | 0.026 |
| 4 | 2 | 19 | 83 | 0.599 | 0.611 | 0.280 | 0.750 | 0.048 |
| 4 | 3 | 19 | 83 | 0.554 | 0.373 | 0.257 | 0.167 | 0.024 |
| 4 | 4 | 19 | 77 | 0.591 | 0.503 | 0.339 | 0.286 | 0.045 |

**XGBoost**:

| Repeat | Fold | Held PIDs | N obs | AUC | BalAcc | F1-macro | Sens-W | PPV-W |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 20 | 89 | 0.820 | 0.740 | 0.509 | 0.800 | 0.222 |
| 0 | 1 | 19 | 81 | 0.834 | 0.685 | 0.582 | 0.333 | 0.400 |
| 0 | 2 | 19 | 75 | 0.800 | 0.750 | 0.454 | 0.667 | 0.111 |
| 0 | 3 | 19 | 84 | 0.823 | 0.686 | 0.589 | 0.462 | 0.353 |
| 0 | 4 | 19 | 82 | 0.803 | 0.691 | 0.459 | 0.600 | 0.083 |
| 1 | 0 | 20 | 86 | 0.822 | 0.733 | 0.456 | 0.667 | 0.148 |
| 1 | 1 | 19 | 84 | 0.863 | 0.799 | 0.529 | 0.800 | 0.190 |
| 1 | 2 | 19 | 82 | 0.896 | 0.824 | 0.633 | 0.833 | 0.185 |
| 1 | 3 | 19 | 79 | 0.779 | 0.650 | 0.491 | 0.545 | 0.194 |
| 1 | 4 | 19 | 80 | 0.802 | 0.637 | 0.546 | 0.333 | 0.429 |
| 2 | 0 | 20 | 85 | 0.939 | 0.932 | 0.694 | 1.000 | 0.333 |
| 2 | 1 | 19 | 83 | 0.760 | 0.686 | 0.421 | 0.667 | 0.133 |
| 2 | 2 | 19 | 81 | 0.823 | 0.806 | 0.642 | 0.800 | 0.400 |
| 2 | 3 | 19 | 79 | 0.749 | 0.618 | 0.542 | 0.125 | 0.071 |
| 2 | 4 | 19 | 83 | 0.822 | 0.677 | 0.478 | 0.556 | 0.238 |
| 3 | 0 | 20 | 89 | 0.909 | 0.867 | 0.636 | 1.000 | 0.308 |
| 3 | 1 | 19 | 80 | 0.816 | 0.729 | 0.509 | 0.600 | 0.150 |
| 3 | 2 | 19 | 80 | 0.868 | 0.767 | 0.650 | 0.571 | 0.500 |
| 3 | 3 | 19 | 81 | 0.737 | 0.572 | 0.464 | 0.182 | 0.125 |
| 3 | 4 | 19 | 81 | 0.821 | 0.674 | 0.421 | 0.500 | 0.107 |
| 4 | 0 | 20 | 82 | 0.819 | 0.777 | 0.562 | 0.800 | 0.286 |
| 4 | 1 | 19 | 86 | 0.835 | 0.758 | 0.620 | 0.700 | 0.389 |
| 4 | 2 | 19 | 83 | 0.839 | 0.672 | 0.530 | 0.250 | 0.100 |
| 4 | 3 | 19 | 83 | 0.805 | 0.599 | 0.456 | 0.000 | 0.000 |
| 4 | 4 | 19 | 77 | 0.743 | 0.661 | 0.484 | 0.429 | 0.136 |

**LightGBM**:

| Repeat | Fold | Held PIDs | N obs | AUC | BalAcc | F1-macro | Sens-W | PPV-W |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 20 | 89 | 0.703 | 0.600 | 0.267 | 0.800 | 0.123 |
| 0 | 1 | 19 | 81 | 0.798 | 0.685 | 0.582 | 0.333 | 0.400 |
| 0 | 2 | 19 | 75 | 0.702 | 0.607 | 0.441 | 0.000 | 0.000 |
| 0 | 3 | 19 | 84 | 0.819 | 0.686 | 0.589 | 0.462 | 0.353 |
| 0 | 4 | 19 | 82 | 0.819 | 0.686 | 0.453 | 0.600 | 0.081 |
| 1 | 0 | 20 | 86 | 0.825 | 0.733 | 0.456 | 0.667 | 0.148 |
| 1 | 1 | 19 | 84 | 0.795 | 0.667 | 0.227 | 1.000 | 0.077 |
| 1 | 2 | 19 | 82 | 0.843 | 0.681 | 0.356 | 1.000 | 0.087 |
| 1 | 3 | 19 | 79 | 0.766 | 0.650 | 0.491 | 0.545 | 0.194 |
| 1 | 4 | 19 | 80 | 0.823 | 0.643 | 0.559 | 0.333 | 0.500 |
| 2 | 0 | 20 | 85 | 0.942 | 0.667 | 0.268 | 1.000 | 0.057 |
| 2 | 1 | 19 | 83 | 0.759 | 0.686 | 0.421 | 0.667 | 0.133 |
| 2 | 2 | 19 | 81 | 0.823 | 0.733 | 0.592 | 0.600 | 0.316 |
| 2 | 3 | 19 | 79 | 0.752 | 0.581 | 0.508 | 0.125 | 0.048 |
| 2 | 4 | 19 | 83 | 0.805 | 0.677 | 0.478 | 0.556 | 0.238 |
| 3 | 0 | 20 | 89 | 0.908 | 0.848 | 0.604 | 1.000 | 0.267 |
| 3 | 1 | 19 | 80 | 0.816 | 0.725 | 0.502 | 0.600 | 0.143 |
| 3 | 2 | 19 | 80 | 0.868 | 0.788 | 0.550 | 0.857 | 0.250 |
| 3 | 3 | 19 | 81 | 0.706 | 0.536 | 0.433 | 0.091 | 0.062 |
| 3 | 4 | 19 | 81 | 0.746 | 0.664 | 0.409 | 0.500 | 0.100 |
| 4 | 0 | 20 | 82 | 0.783 | 0.669 | 0.470 | 0.600 | 0.176 |
| 4 | 1 | 19 | 86 | 0.832 | 0.752 | 0.610 | 0.700 | 0.368 |
| 4 | 2 | 19 | 83 | 0.834 | 0.672 | 0.530 | 0.250 | 0.100 |
| 4 | 3 | 19 | 83 | 0.761 | 0.594 | 0.453 | 0.000 | 0.000 |
| 4 | 4 | 19 | 77 | 0.765 | 0.661 | 0.484 | 0.429 | 0.136 |

**SVM**:

| Repeat | Fold | Held PIDs | N obs | AUC | BalAcc | F1-macro | Sens-W | PPV-W |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 20 | 89 | 0.657 | 0.512 | 0.429 | 0.100 | 0.045 |
| 0 | 1 | 19 | 81 | 0.655 | 0.547 | 0.426 | 0.333 | 0.105 |
| 0 | 2 | 19 | 75 | 0.712 | 0.482 | 0.281 | 0.333 | 0.031 |
| 0 | 3 | 19 | 84 | 0.711 | 0.508 | 0.437 | 0.462 | 0.188 |
| 0 | 4 | 19 | 82 | 0.666 | 0.525 | 0.349 | 0.200 | 0.026 |
| 1 | 0 | 20 | 86 | 0.521 | 0.320 | 0.259 | 0.000 | 0.000 |
| 1 | 1 | 19 | 84 | 0.757 | 0.663 | 0.388 | 0.800 | 0.098 |
| 1 | 2 | 19 | 82 | 0.723 | 0.583 | 0.447 | 0.500 | 0.073 |
| 1 | 3 | 19 | 79 | 0.596 | 0.436 | 0.373 | 0.091 | 0.053 |
| 1 | 4 | 19 | 80 | 0.744 | 0.534 | 0.441 | 0.111 | 0.167 |
| 2 | 0 | 20 | 85 | 0.793 | 0.684 | 0.386 | 1.000 | 0.083 |
| 2 | 1 | 19 | 83 | 0.547 | 0.446 | 0.335 | 0.167 | 0.042 |
| 2 | 2 | 19 | 81 | 0.709 | 0.514 | 0.414 | 0.100 | 0.071 |
| 2 | 3 | 19 | 79 | 0.609 | 0.398 | 0.381 | 0.250 | 0.054 |
| 2 | 4 | 19 | 83 | 0.617 | 0.455 | 0.334 | 0.000 | 0.000 |
| 3 | 0 | 20 | 89 | 0.645 | 0.464 | 0.401 | 0.250 | 0.059 |
| 3 | 1 | 19 | 80 | 0.679 | 0.471 | 0.375 | 0.200 | 0.040 |
| 3 | 2 | 19 | 80 | 0.709 | 0.445 | 0.337 | 0.286 | 0.069 |
| 3 | 3 | 19 | 81 | 0.780 | 0.581 | 0.505 | 0.273 | 0.250 |
| 3 | 4 | 19 | 81 | 0.581 | 0.414 | 0.317 | 0.000 | 0.000 |
| 4 | 0 | 20 | 82 | 0.648 | 0.471 | 0.363 | 0.100 | 0.040 |
| 4 | 1 | 19 | 86 | 0.680 | 0.499 | 0.374 | 0.500 | 0.132 |
| 4 | 2 | 19 | 83 | 0.791 | 0.633 | 0.366 | 0.750 | 0.067 |
| 4 | 3 | 19 | 83 | 0.588 | 0.386 | 0.328 | 0.167 | 0.043 |
| 4 | 4 | 19 | 77 | 0.676 | 0.561 | 0.388 | 0.286 | 0.071 |

## Key Insights

**CES-D currency is essential:**
1. **Onboarding → Full model**: ΔAUC = +0.236. The largest single gain in the ladder comes from having a current, updating CES-D — not from adding behavioral features.
2. **Stale 4-week degradation** (Full → Stale 4wk): ΔAUC = −0.171. Even a 4-week-old CES-D causes substantial performance loss, reinforcing that the CES-D anchor must be current.
3. **Onboarding is close to intake form only across models**: best Onboarding is SVM (0.733) vs best Intake is XGBoost (0.720); for XGBoost specifically, Onboarding (0.670) falls below Intake (0.720). Adding 37 behavioral features on top of a frozen first CES-D yields little to no gain over having only age and gender. When the CES-D anchor is stale, behavioral features contribute minimally.

**Behavioral features do carry signal — but need a good anchor:**
4. **Cold start vs Onboarding** (AUC 0.821 vs 0.670): a model evaluated on completely unseen persons (with pmcesd = pop_mean) outperforms the same model with Screenome data but a frozen intake CES-D. This confirms that a stale individual anchor is worse than no individual anchor at all.
5. **Cold start vs Full model**: ΔAUC = −0.085. The gap between never-seen persons and known persons is relatively modest, suggesting behavioral trajectory generalizes reasonably across individuals.

**Interpreting No fresh CES-D:**
6. **No fresh CES-D Sens-W = 0.892 is not a real result.** The model predicts worsening for nearly every observation when prior_cesd = pop_mean ≈ 12 (below most persons' pmcesd), making it a near-universal worsening alarm. AUC = 0.666 is the honest performance summary. PPV-W = 0.170 confirms the prediction is clinically unusable.
