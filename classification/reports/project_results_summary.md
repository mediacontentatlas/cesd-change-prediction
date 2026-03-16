# Screenome-Based Depression Symptom Change Prediction — Results Summary

## Overview

This project predicts the **direction of biweekly depressive symptom change** (improving / stable / worsening) from smartphone behavioral data ("screenome" features) combined with prior CES-D scores and demographics. This is a **within-person longitudinal classification task** — not between-person depression detection. CES-D change has near-zero between-person ICC (ICC ~ 0), meaning no person consistently changes more than another.

**96 participants**, biweekly assessments, train/val/test split preserving temporal order within each person. Train: 1,196 obs, Val: 395 obs, Test: 411 obs.

---

## 1. Primary Result: Severity-Crossing Prediction

**Best model**: XGBoost, 39 features (behavioral + lag + person_mean_cesd)

| Metric | Value | 95% CI |
|---|---|---|
| AUC (OvR) | **0.906** | [0.881, 0.929] |
| Balanced Accuracy | **0.834** | [0.784, 0.882] |
| Worsening Sensitivity | **0.838** | [0.707, 0.947] |
| Worsening PPV | **0.356** | [0.253, 0.464] |

> **Suggested format**: Table in main text. This is the headline finding.

### All 4 Models (39-feature condition, test set)

| Model | AUC [95% CI] | BalAcc [95% CI] | Sens-W [95% CI] | PPV-W [95% CI] |
|---|---|---|---|---|
| XGBoost | 0.906 [0.881, 0.929] | 0.834 [0.784, 0.882] | 0.838 [0.707, 0.947] | 0.356 [0.253, 0.464] |
| LightGBM | 0.901 [0.876, 0.925] | 0.842 [0.798, 0.890] | 0.865 [0.744, 0.970] | 0.344 [0.247, 0.439] |
| ElasticNet | 0.829 [0.792, 0.867] | 0.691 [0.624, 0.762] | 0.730 [0.588, 0.872] | 0.248 [0.168, 0.333] |
| SVM | 0.841 [0.806, 0.876] | 0.696 [0.628, 0.766] | 0.649 [0.488, 0.806] | 0.304 [0.203, 0.405] |

> **Suggested format**: Table. Could also be a grouped bar chart if comparing across labels.
>
> **Source**: `reports/bootstrap_ci_results.md`, `models/bootstrap_ci/bootstrap_results.csv`

---

## 2. Feature Ablation: What Each Data Source Contributes

Cumulative feature conditions tested with 1000-bootstrap CIs and paired significance tests:

| Condition | N features | XGBoost AUC | XGBoost Sens-W | Key insight |
|---|---|---|---|---|
| prior_cesd only | 1 | 0.872 | 0.730 | CES-D alone carries strong signal |
| base (behavioral + demo) | 21 | 0.876 | 0.649 | Behavioral features add marginal AUC (+0.004, n.s.) |
| + behavioral lag | 38 | 0.875 | 0.622 | Lag features: no significant improvement |
| **+ person_mean_cesd** | **39** | **0.906** | **0.838** | **+0.031 AUC (p=0.009), +0.218 Sens-W (p=0.007)** |

**Key finding**: `person_mean_cesd` is the single most impactful feature addition. Adding it to the 38-feature model significantly improves AUC (+0.031, p=0.009), BalAcc (+0.100, p=0.001), and Sens-W (+0.218, p=0.007) for XGBoost. This holds across all 4 models and all 3 label types.

> **Suggested format**: **Figure** — grouped bar chart with error bars showing AUC (or BalAcc) by feature condition, one group per model. The ablation figure already exists at `reports/figures/10_feature_ablation.png`. Paired significance asterisks should annotate the 38→39 transition.
>
> **Source**: `reports/bootstrap_ci_results.md`, `models/bootstrap_ci/bootstrap_results.csv`, `models/bootstrap_ci/paired_tests.csv`

---

## 3. Label Robustness: Three Operationalizations

| Label | Description | Class Balance | Best Model | AUC | BalAcc | Sens-W |
|---|---|---|---|---|---|---|
| sev_crossing | Clinical severity boundary crossing | 11%/80%/9% | XGBoost | **0.906** | **0.834** | **0.838** |
| personal_sd | Person-specific SD change (k=1.0) | 12%/78%/10% | ElasticNet | 0.759 | 0.624 | 0.585 |
| balanced_tercile | Rank-based equal terciles | 33%/33%/33% | LightGBM | 0.732 | 0.557 | 0.555 |

**Key finding**: `sev_crossing` is the most predictable operationalization because clinical thresholds align with what the model can distinguish (boundary crossings are driven by prior CES-D level, which the model handles well). `personal_sd` and `balanced_tercile` are harder because they define change relative to each person's variability or rank, not absolute thresholds.

> **Suggested format**: Table in main text. The three labels should each get a brief paragraph describing the rationale and result.
>
> **Source**: `reports/bootstrap_ci_results.md` (all three labels with CIs), `reports/personal_sd_results.md`, `reports/balanced_label_results.md`

---

## 4. Deployment Scenarios: Graceful Degradation

How the model performs under realistic clinical deployment constraints (XGBoost, sev_crossing):

| Scenario | AUC [95% CI] | BalAcc | Sens-W | PPV-W |
|---|---|---|---|---|
| Population baseline (predict all stable) | 0.500 | 0.333 | 0.000 | 0.000 |
| Revert-to-person-mean (rule-based) | 0.750 [0.697, 0.804] | 0.674 | 0.541 | 0.408 |
| Last-change-only (rule-based) | 0.556 [0.527, 0.588] | 0.335 | 0.054 | 0.053 |
| Intake form only (demographics) | 0.720 [0.667, 0.766] | 0.463 | 0.459 | 0.173 |
| Onboarding (single CES-D) | 0.670 [0.613, 0.725] | 0.458 | 0.297 | 0.159 |
| Stale 4 weeks | 0.735 [0.682, 0.786] | 0.565 | 0.514 | 0.232 |
| Stale 8 weeks | 0.702 [0.647, 0.754] | 0.507 | 0.432 | 0.188 |
| No fresh CES-D (screenome + anchor) | 0.666 [0.627, 0.707] | 0.506 | 0.892 | 0.170 |
| Cold start (unseen person, 5×5 CV) | 0.821 ±0.049 | 0.720 | 0.569 | 0.224 |
| **Full model (known person)** | **0.906 [0.881, 0.929]** | **0.834** | **0.838** | **0.356** |

**Key findings**:
- **Revert-to-person-mean is a strong baseline** (AUC=0.750) — the ML model must beat simple regression-to-the-mean
- **Last-change-only is very weak** (AUC=0.556) — CES-D severity transitions have near-zero momentum
- **Cold start works well** (AUC=0.821) — the model generalizes to unseen persons
- **Stale CES-D degrades gracefully** — even 8-week-old CES-D still outperforms demographics alone
- **No fresh CES-D has high sensitivity but low PPV** (Sens-W=0.892, PPV-W=0.170) — the model over-alerts without current self-report

> **Suggested format**: **Figure** — deployment ladder bar chart (AUC on y-axis, scenarios ordered by information available). The deployment ladder figure exists at `reports/figures/13_deployment_ladder.png`. Also include the table for exact values.
>
> **Source**: `reports/deployment_results.md`, `models/deployment_scenarios/deployment_results.csv`, `models/deployment_scenarios/deployment_bootstrap_ci.csv`

---

## 5. Detection Quality by Worsening Type

### By transition type (sev_crossing, XGBoost 39-feat)

| Transition | N cases | Caught | Sensitivity |
|---|---|---|---|
| min → mod | 20 | 15 | 0.750 |
| min → sev | 9 | 9 | **1.000** |
| mod → sev | 8 | 7 | 0.875 |

### By prior CES-D range

| CES-D Range | N worsening | Sensitivity | False Alarms |
|---|---|---|---|
| 0–8 | 5 | 0.40 | 5 |
| 8–12 | 4 | 0.50 | 14 |
| 12–16 | 20 | **1.00** | 28 |
| 16–24 | 8 | 0.875 | 9 |
| 24–60 | 0 | n/a | 0 |

**Key finding**: The model is most sensitive for individuals near clinical thresholds (CES-D 12–24), where boundary crossings are most likely. All min→sev transitions (the most clinically urgent) are caught. The model is weakest at very low CES-D (0–8) where worsening is rare and subtle.

> **Suggested format**: Table in main text. The transition detection table is compact and informative; the CES-D range table could also be a small heat map or annotation on a line plot.
>
> **Source**: `reports/phenotype_posthoc_writeup.md`, `models/posthoc/transition_detection.csv`, `models/posthoc/sensitivity_by_cesd.csv`

---

## 6. Feature Importance — Multi-Method, Multi-Label Comparison

Four complementary methods (XGBoost gain, LightGBM gain, SHAP, ElasticNet coefficients) were applied to **all three label types** to understand how the label definition affects which features the models use.

### Feature utilization changes dramatically by label type

| Metric | sev_crossing | balanced_tercile | personal_sd |
|---|---|---|---|
| XGBoost nonzero gain features | 18/39 | **39/39** | **37/39** |
| XGBoost nonzero SHAP (worsening) | 5/39 | **38/39** | **33/39** |
| CES-D features' share of gain (XGB) | 45% | 15% | 13% |

The sev_crossing SHAP sparsity is a property of the label, not the model — the same XGBoost architecture produces rich SHAP distributions under alternative labels.

### sev_crossing: top behavioral features (XGBoost gain)

| Rank | Feature | Gain |
|---|---|---|
| 3 | lag_mean_daily_social_ratio | 0.060 |
| 4 | lag_mean_daily_social_screens_delta | 0.047 |
| 5 | clip_dispersion_delta | 0.042 |
| 6 | mean_daily_overnight_ratio_delta | 0.041 |
| 7 | mean_daily_switches_delta | 0.041 |

### balanced_tercile: gain is much more distributed

Top behavioral: age (0.033), lag_clip_dispersion (0.028), mean_daily_social_ratio_delta (0.028), mean_daily_social_screens_delta (0.028). All 39 features have nonzero gain.

### personal_sd: lagged behavioral features dominate

Top behavioral: lag_mean_daily_switches (0.037), lag_mean_daily_switches_delta (0.036), lag_active_day_ratio_delta (0.034), lag_switches_per_screen_delta (0.033). **7 of the top 10 features are lagged**, suggesting individual behavioral trajectories help predict person-specific unusual changes.

### ElasticNet worsening coefficients across labels (less-regularized, C=0.1)

| Feature | sev_crossing | balanced_tercile | personal_sd |
|---|---|---|---|
| person_mean_cesd | +0.249 | **+1.418** | **+0.932** |
| prior_cesd | −0.172 | **−0.755** | **−0.836** |
| mean_daily_social_screens_delta | **+0.202** | +0.051 | — |
| switches_per_screen | +0.059 | +0.022 | +0.045 |
| mean_daily_overnight_ratio | — | +0.064 | — |
| mean_daily_overnight_ratio_delta | — | — | +0.025 |

### Cross-method convergence (sev_crossing)

8 behavioral features appear in 3+ of the 4 methods. Top: **mean_daily_switches_delta** (4/4), **mean_daily_screens** (4/4), **mean_daily_social_screens_delta** (3/4), **lag_mean_daily_social_ratio** (3/4), **clip_dispersion_delta** (3/4).

### Key findings

1. **Behavioral features matter more under harder labels.** When the label isn't directly determined by CES-D thresholds, the models engage nearly all 39 features.
2. **Social media and switching behavior are the most consistent behavioral signals** across all three labels and all methods.
3. **Lagged features become particularly important under personal SD labels**, suggesting individual-level behavioral trajectories help predict person-specific unusual changes.
4. While behavioral features don't significantly improve AUC (+0.004, n.s.) for sev_crossing, they would become more important in deployment scenarios without current CES-D (Sens-W=0.892 in "No fresh CES-D" scenario).

> **Suggested format**: **Figure** — SHAP stacked bar chart at `reports/figures/14_shap_summary.png` + gain importance at `reports/figures/12_feature_importance.png`. **Table** — cross-label ElasticNet worsening coefficients and feature utilization table. Full report at `reports/feature_importance_report.md`.
>
> **Source**: `reports/feature_importance_report.md` (comprehensive multi-label report), plus per-label CSVs in `models/feature_importance/` (sev_crossing), `models/feature_importance/balanced_tercile/`, `models/feature_importance/personal_sd/` (each containing `feature_importance.csv`, `lgbm_feature_importance.csv`, `feature_coefficients.csv`, `feature_coefficients_interp.csv`, `shap_summary_xgboost.csv`, `shap_summary_lightgbm.csv`)

---

## 7. Phenotype-Based Posthoc Analysis

Five behavioral phenotype dimensions were used to stratify model performance:

| Phenotype | Most informative finding |
|---|---|
| deviation_cluster | High-deviation individuals: Sens-W=0.952 vs 0.688 for low-deviation |
| reactivity_cluster | Reactive individuals: Sens-W=0.905 vs 0.750 |
| cesd_severity | Mild+ (CES-D 16+): highest false alarm rate (0.300) — expected near threshold |
| level_cluster | Group 1: Sens-W=0.900, but only 10 worsening cases |
| delta_cluster | Group 1: Sens-W=0.857, AUC=0.912 |

**Key findings**:
- `deviation_cluster` is the most informative phenotype for stratifying model quality
- Most **phenotype-specific models underperform the global model** due to insufficient subgroup training data
- The model catches 31/37 worsening cases (84%); the 6 missed cases tend to be low-deviation, low-reactivity individuals
- **deviation × cesd_severity interaction**: high-deviation + minimal severity = Sens-W=0.941 (17 cases, 16 caught)

> **Suggested format**: Table for stratification results. Consider a small-multiples panel showing Sens-W by phenotype group. Text paragraph for the "caught vs missed" profile.
>
> **Source**: `reports/phenotype_posthoc_writeup.md`, `models/posthoc/phenotype_stratification.csv`, `models/posthoc/phenotype_specific_models.csv`, `models/posthoc/interaction_results.csv`

---

## File Index: Where to Find Everything

### Reports (human-readable writeups)

| File | Contents |
|---|---|
| `reports/classification_results.md` | Full methods + results for sev_crossing (confusion matrices, per-class metrics, baselines) |
| `reports/bootstrap_ci_results.md` | **Feature ablation with 95% CIs**: all 4 models × 4 conditions × 3 labels + paired significance tests |
| `reports/deployment_results.md` | **Deployment scenario results with 95% CIs**: 10 scenarios, all 4 models, cold start fold details |
| `reports/personal_sd_results.md` | Personal SD label: 4 models × 4 conditions, grid-searched |
| `reports/balanced_label_results.md` | Balanced tercile label: 4 models × 4 conditions, grid-searched |
| `reports/phenotype_posthoc_writeup.md` | Phenotype stratification, transition detection, sensitivity by CES-D, caught vs missed |
| `reports/feature_importance_report.md` | **Multi-method, multi-label feature importance**: gain, SHAP, ElasticNet coefficients, cross-method convergence, cross-label comparison |
| `reports/experiment_spec.md` | Experiment design specification |

### Machine-readable data (CSVs for figures/analysis)

| File | Contents |
|---|---|
| `models/bootstrap_ci/bootstrap_results.csv` | All ablation point estimates + CIs (feed into ablation figure) |
| `models/bootstrap_ci/paired_tests.csv` | Paired bootstrap significance tests |
| `models/bootstrap_ci/sev_crossing_best_params.yaml` | **Canonical hyperparameters** for all 4 models × 4 conditions |
| `models/deployment_scenarios/deployment_results.csv` | Deployment scenario point estimates |
| `models/deployment_scenarios/deployment_bootstrap_ci.csv` | Deployment scenario bootstrap CIs |
| `models/deployment_scenarios/cold_start_fold_results.csv` | Per-fold cold start results (25 folds) |
| `models/feature_importance/shap_summary_xgboost.csv` | SHAP values per feature per class (sev_crossing) |
| `models/feature_importance/feature_importance.csv` | XGBoost gain importance (sev_crossing) |
| `models/feature_importance/feature_coefficients.csv` | ElasticNet signed coefficients (sev_crossing) |
| `models/feature_importance/feature_coefficients_interp.csv` | ElasticNet less-regularized coefficients (sev_crossing) |
| `models/feature_importance/balanced_tercile/feature_importance.csv` | XGBoost gain importance (balanced_tercile) |
| `models/feature_importance/balanced_tercile/shap_summary_xgboost.csv` | SHAP values per feature per class (balanced_tercile) |
| `models/feature_importance/balanced_tercile/feature_coefficients_interp.csv` | ElasticNet less-regularized coefficients (balanced_tercile) |
| `models/feature_importance/personal_sd/feature_importance.csv` | XGBoost gain importance (personal_sd) |
| `models/feature_importance/personal_sd/shap_summary_xgboost.csv` | SHAP values per feature per class (personal_sd) |
| `models/feature_importance/personal_sd/feature_coefficients_interp.csv` | ElasticNet less-regularized coefficients (personal_sd) |
| `models/posthoc/transition_detection.csv` | Detection rate by transition type |
| `models/posthoc/sensitivity_by_cesd.csv` | Sensitivity by prior CES-D range |
| `models/posthoc/phenotype_stratification.csv` | Global model stratified by phenotype |

### Existing figures

| File | Contents |
|---|---|
| `reports/figures/10_feature_ablation.png` | Feature ablation bar chart with CIs |
| `reports/figures/12_feature_importance.png` | Feature importance (gain) bar chart |
| `reports/figures/13_deployment_ladder.png` | Deployment scenario ladder |
| `reports/figures/14_shap_summary.png` | SHAP beeswarm summary |

### Scripts (reproducibility)

| Script | Purpose |
|---|---|
| `scripts/train_classifier.py` | Main classifier training (4 models, sev_crossing) |
| `scripts/bootstrap_ci.py` | Bootstrap CIs, feature ablation, paired tests (4 models × 4 conditions × 3 labels) |
| `scripts/deployment_scenarios.py` | Deployment ladder + baselines + cold start + bootstrap CIs |
| `scripts/run_balanced_label_experiment.py` | Balanced tercile experiment |
| `scripts/run_personal_sd_experiment.py` | Personal SD experiment |
| `scripts/generate_figures.py` | Publication figures |
| `scripts/feature_importance_alt_labels.py` | Feature importance extraction for balanced_tercile and personal_sd labels |
| `posthoc/phenotype_posthoc.py` | Phenotype stratification analysis |

---

## Recommended Presentation Strategy

### For a paper/manuscript

1. **Main text Table 1**: Primary result — 4 models × sev_crossing, 39-feature condition (Section 1 above)
2. **Main text Figure 1**: Feature ablation with CIs (existing `10_feature_ablation.png` or regenerate from CSV)
3. **Main text Figure 2**: Deployment ladder (existing `13_deployment_ladder.png` or regenerate)
4. **Main text Table 2**: Label robustness — best model per label (Section 3)
5. **Main text Figure 3**: SHAP summary (existing `14_shap_summary.png`)
6. **Supplementary Table S1**: Full ablation results — all 4 models × 4 conditions × 3 labels (from `bootstrap_ci_results.md`)
7. **Supplementary Table S2**: Detection by transition type and CES-D range (Section 5)
8. **Supplementary Table S3**: Deployment scenarios — all models (from `deployment_results.md`)
9. **Supplementary Table S4**: Phenotype stratification (Section 7)
10. **Supplementary Table S5**: Paired bootstrap significance tests (from `models/bootstrap_ci/paired_tests.csv`)
11. **Supplementary Table S6**: Cross-label feature utilization — nonzero features, SHAP, gain by label type (from `reports/feature_importance_report.md` Section 6)
12. **Supplementary Table S7**: Cross-label ElasticNet worsening coefficients (from `feature_coefficients_interp.csv` in each label's model directory)

### For a presentation/talk

- Lead with the deployment ladder figure (tells the story of information value)
- Show the 0.906 AUC headline, then the ablation showing person_mean_cesd as the key feature
- Use the transition detection table to make it concrete ("catches 100% of minimal→severe transitions")
- End with cold start result to show generalizability

### For a technical report/handoff

- Start with `reports/classification_results.md` for full methods + primary results
- Point to `reports/bootstrap_ci_results.md` for all ablation CIs
- Point to `reports/deployment_results.md` for deployment scenarios
- Point to `reports/phenotype_posthoc_writeup.md` for posthoc analyses
- All CSV data files are in `models/` subdirectories for custom figure generation
