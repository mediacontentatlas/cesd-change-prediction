# Classification: Predicting Direction of Symptom Change

Can we predict whether a person's depressive symptoms will improve, remain stable, or worsen over the next two weeks — using their current CES-D trajectory and Screenome-derived behavioral features?

This is a **within-person longitudinal classification task**, not between-person depression detection. CES-D *levels* have high between-person stability (ICC = 0.74), but CES-D *change* (delta) has near-zero between-person ICC (ICC ~ 0; person means explain only 0.4% of delta variance). No person consistently changes more than another — the prediction target is inherently within-person.

## Label Types

Three operationalizations of "improving / stable / worsening" are evaluated to test robustness of findings across definitions:

| Label | Description | Class balance | Location |
|---|---|---|---|
| sev_crossing | Clinical severity boundary crossing (CES-D thresholds 16, 24) | 11% / 80% / 9% | `labels/sev_crossing/` |
| personal_sd | Person-specific SD-based change (k=1.0) | 12% / 78% / 10% | `labels/personal_sd/` |
| balanced_tercile | Rank-based equal-sized terciles of CES-D delta | 33% / 33% / 33% | `labels/balanced_tercile/` |

All label arrays: 0=improving, 1=stable, 2=worsening. Labels are shared with the regression task for posthoc direction analysis.

## Models

All models use class_weight='balanced' or equivalent sample weighting. Hyperparameters are grid-searched per model x feature condition on the validation set.

- **ElasticNet** logistic regression (L1/L2 regularized) — interpretable, provides signed per-feature coefficients
- **XGBoost** (gradient-boosted trees, sample-weighted)
- **LightGBM** (gradient-boosted trees, class_weight=balanced)
- **SVM** (RBF or linear kernel, class_weight=balanced)

## Feature Conditions

Cumulative feature sets test what each data source contributes:

| Condition | N features | Description |
|---|---|---|
| prior_cesd only | 1 | Baseline CES-D score alone (clinician baseline) |
| base | 21 | prior_cesd + 20 behavioral/demographic features |
| base + behavioral lag | 38 | base + lag-1 of behavioral features (excl. lag_age, lag_gender, lag_prior_cesd, lag_cesd_delta) |
| base + behavioral lag + pmcesd | 39 | 38 + person_mean_cesd (trait-level depression anchor) |

## Metrics (report on train / val / test)

| Metric | Description |
|---|---|
| AUC (OvR) | One-vs-rest macro-averaged area under ROC |
| BalAcc | Balanced accuracy (macro recall) |
| Sens-W | Sensitivity for worsening class (recall of class 2) |
| PPV-W | Positive predictive value for worsening class (precision of class 2) |

## Key Results (39-feature condition, test set, per-condition grid-searched params)

| Model | Label | AUC | BalAcc | Sens-W | PPV-W |
|---|---|---|---|---|---|
| XGBoost | sev_crossing | **0.906** | 0.834 | 0.838 | **0.356** |
| LightGBM | sev_crossing | 0.901 | **0.842** | **0.865** | 0.344 |
| ElasticNet | personal_sd | **0.759** | 0.624 | 0.585 | 0.175 |
| SVM | personal_sd | 0.751 | 0.620 | 0.610 | 0.191 |
| LightGBM | balanced_tercile | **0.732** | **0.557** | 0.555 | 0.481 |
| XGBoost | balanced_tercile | 0.723 | 0.555 | 0.562 | 0.490 |

person_mean_cesd is the single most impactful feature addition across all label types (38->39 features: +0.06-0.08 AUC). Bootstrap paired tests confirm significance (p < 0.01 for all 4 models on sev_crossing).

## Baselines and Deployment Ladder

| Stage | What you know | XGB AUC | XGB BalAcc | XGB Sens-W |
|---|---|---|---|---|
| Population baseline | Predict all stable | 0.500 | 0.333 | 0.000 |
| Revert-to-person-mean | Rule: severity → person mean | 0.750 | 0.674 | 0.541 |
| Last-change-only | Rule: repeat last transition | 0.556 | 0.335 | 0.054 |
| Intake form only | Age + gender (3 feat) | 0.720 | 0.463 | 0.459 |
| Onboarding | Intake CES-D as anchor | 0.670 | 0.458 | 0.297 |
| Stale 4 weeks | prior_cesd from t-1 | 0.735 | 0.565 | 0.514 |
| Stale 8 weeks | prior_cesd from t-2 | 0.702 | 0.507 | 0.432 |
| No fresh CES-D | prior_cesd = pop_mean | 0.666 | 0.506 | 0.892 |
| Cold start (5x5 CV) | Unseen person, pmcesd=pop_mean | 0.821 | 0.720 | 0.569 |
| **Full model** | **All features, known person** | **0.906** | **0.834** | **0.838** |

See `reports/deployment_results.md` for full results across all models and per-fold cold-start details.

## Directory Layout

```
classification/
├── configs/            # Grid search parameters
├── labels/             # Shared label arrays (used by regression posthoc too)
│   ├── sev_crossing/
│   ├── personal_sd/
│   └── balanced_tercile/
├── models/
│   ├── sev_crossing/{elasticnet,xgboost,lightgbm,svm}/
│   ├── personal_sd/{elasticnet,xgboost,lightgbm,svm}/
│   ├── balanced_tercile/{elasticnet,xgboost,lightgbm,svm}/
│   ├── bootstrap_ci/          # Bootstrap CI results per label x model x condition
│   ├── deployment_scenarios/   # Deployment scenario outputs and cold-start folds
│   ├── feature_importance/     # Gain importance, ElasticNet coefficients, SHAP values (sev_crossing at root; balanced_tercile/ and personal_sd/ subdirs)
│   └── posthoc/                # Transition detection, CES-D sensitivity, phenotype results
├── scripts/            # Training and evaluation scripts
├── posthoc/            # Posthoc analyses (phenotype)
└── reports/            # Results, writeups, sensitivity analyses
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/train_classifier.py` | Main classifier training (all 4 models, sev_crossing label) |
| `scripts/run_balanced_label_experiment.py` | Balanced tercile experiment with per-condition grid search |
| `scripts/run_personal_sd_experiment.py` | Personal SD experiment with per-condition grid search |
| `scripts/bootstrap_ci.py` | Bootstrap 95% CIs, feature ablation (4 conditions × 4 models × 3 labels), and paired significance tests |
| `scripts/deployment_scenarios.py` | Deployment ladder + baselines (revert-to-mean, last-change, intake, stale CES-D, cold start) |
| `scripts/generate_figures.py` | Publication figures: ablation with CIs, feature importance, deployment ladder |
| `scripts/feature_importance_alt_labels.py` | Feature importance extraction for balanced_tercile and personal_sd labels (XGBoost gain, LightGBM gain, SHAP, ElasticNet coefficients) |
| `posthoc/phenotype_posthoc.py` | Phenotype-stratified posthoc analysis |

## Where to Find Results

| What | Location |
|---|---|
| **Feature ablation** (all models × conditions × labels, with bootstrap CIs) | `reports/bootstrap_ci_results.md`, `models/bootstrap_ci/bootstrap_results.csv` |
| **sev_crossing full results** (confusion matrices, per-class, baselines) | `reports/classification_results.md` |
| **personal_sd results** (4 models × 4 conditions, grid-searched) | `reports/personal_sd_results.md` |
| **balanced_tercile results** (4 models × 4 conditions, grid-searched) | `reports/balanced_label_results.md` |
| **Deployment scenarios + baselines** | `reports/deployment_results.md`, `models/deployment_scenarios/` |
| **Feature importance** (gain, SHAP, ElasticNet coefficients) | `models/feature_importance/` (sev_crossing), `models/feature_importance/balanced_tercile/`, `models/feature_importance/personal_sd/` |
| **Feature importance report** (multi-method, multi-label comparison) | `reports/feature_importance_report.md` |
| **Phenotype posthoc** (stratification, phenotype-specific models) | `reports/phenotype_posthoc_writeup.md`, `models/posthoc/` |
| **Detection by transition type** (min→mod, min→sev, mod→sev) | `models/posthoc/transition_detection.csv` |
| **Sensitivity by CES-D range** | `models/posthoc/sensitivity_by_cesd.csv` |
| **Phenotype × severity interactions** | `models/posthoc/interaction_results.csv` |
