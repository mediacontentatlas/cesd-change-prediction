# Mixed-Effects Linear Model (MixedLM)

Linear mixed-effects regression for predicting CES-D delta, accounting for repeated measures within participants via random intercepts (and optionally random slopes). Uses `statsmodels.MixedLM` with REML estimation and fallback optimization for convergence robustness.

## Quick Start

```bash
# From repo root:

# 1. Train all four ablation conditions (random intercept only)
python regression/mixedlm/scripts/train_mixedlm.py

# 2. Run posthoc direction analysis using sev_crossing labels
python regression/mixedlm/scripts/posthoc_mixedlm.py
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/train_mixedlm.py` | Train MixedLM across feature ablation conditions. Saves model, predictions, baselines, and metrics per condition. |
| `scripts/posthoc_mixedlm.py` | Post-hoc direction analysis: stratified regression error by direction class, classification metrics (BalAcc, AUC, Sens-W, PPV-W), confusion matrices, and diagnostic plots. |
| `scripts/model.py` | `MixedLMModel` class — fit, predict, random effects extraction, convergence diagnostics, fallback optimization. |
| `scripts/metrics.py` | Evaluation utilities — aggregate metrics (MAE, RMSE, within-person R², between-person R²), five baselines (B0–B4), direction classification, comparison tables. |

## Feature Ablation Conditions

Four conditions matching the classification task for direct comparability:

| Condition | Features | N features | Description |
|---|---|---|---|
| `prior_cesd` | `prior_cesd` | 1 | Baseline: prior symptom score only |
| `base` | All 21 base features | 21 | Full base feature set |
| `base_dev` | Base + within-person deviation | 29 | Base + 8 deviation features (`X_dev_*.npy`) |
| `base_dev_pmcesd` | Base + dev + `person_mean_cesd` | 30 | Adds person-level mean CES-D |

## Training

### Run all four ablation conditions (random intercept only)

```bash
python regression/mixedlm/scripts/train_mixedlm.py
```

### Run the full 9-model sweep (matches screenome_mh_pred repo)

```bash
python regression/mixedlm/scripts/train_mixedlm.py --full-sweep
```

This trains all model variants from the original screenome_mh_pred analysis:

| # | Label | Random Effects | Pooled? |
|---|---|---|---|
| 1 | `1_pooled` | None (each row = own group) | Yes |
| 2 | `2_intercept` | Intercept per person | No |
| 3 | `3_prior_slope` | Intercept + prior_cesd | No |
| 4 | `4_prior_switches` | Intercept + prior_cesd + mean_daily_switches | No |
| 5 | `5_prior_social` | Intercept + prior_cesd + mean_daily_social_ratio | No |
| 6 | `6_prior_soc_ext` | Intercept + prior_cesd + social_screens + social_ratio | No |
| 7 | `7_sw_screens` | Intercept + mean_daily_switches + mean_daily_screens | No |
| 8 | `8_prior_sw_scr` | Intercept + prior_cesd + switches + screens | No |
| 9 | `9_dev_features` | Intercept + prior_cesd (29 features incl. dev) | No |

### Run a single condition

```bash
python regression/mixedlm/scripts/train_mixedlm.py --condition base
```

### Custom random slope variants

```bash
# Train intercept-only AND intercept + random slope for prior_cesd
python regression/mixedlm/scripts/train_mixedlm.py --with-slopes

# Custom random slopes
python regression/mixedlm/scripts/train_mixedlm.py --random-slopes prior_cesd mean_daily_switches

# Pooled model (no person grouping)
python regression/mixedlm/scripts/train_mixedlm.py --no-pid --condition base
```

### Override output directory

```bash
python regression/mixedlm/scripts/train_mixedlm.py --output-dir results/mixedlm_experiment
```

### Output per condition

Each condition saves to `models/<condition>/`:

```
models/<condition>/
├── model.pkl                          # Pickled fitted model
├── y_pred_train.npy                   # Predictions per split
├── y_pred_val.npy
├── y_pred_test.npy
├── random_effects.csv                 # Per-person random intercepts/slopes
├── convergence_info.json              # Optimizer attempts and convergence status
├── model_summary.txt                  # Full statsmodels summary
├── training_results.json              # Model config, feature names, fit metrics
├── {train,val,test}_aggregate_comparison.csv    # Model vs 5 baselines (MAE, RMSE, W-R², B-R²)
└── {train,val,test}_direction_classification.csv # Direction prediction metrics
```

## Post-Hoc Direction Analysis

Uses classification labels from `classification/labels/` to evaluate whether the regression model captures the *direction* of symptom change.

### Run posthoc for all trained models

```bash
python regression/mixedlm/scripts/posthoc_mixedlm.py
```

### Use multiple label types

```bash
python regression/mixedlm/scripts/posthoc_mixedlm.py \
    --label-types sev_crossing personal_sd balanced_tercile
```

### Analyze a single model

```bash
python regression/mixedlm/scripts/posthoc_mixedlm.py \
    --model-dir regression/mixedlm/models/base_dev_pmcesd
```

### Posthoc output

```
reports/posthoc/
├── all_posthoc_results.csv            # Combined classification metrics (all models x splits)
├── posthoc_summary.md                 # Markdown summary table
└── <condition>/
    ├── posthoc_classification_sev_crossing.csv   # BalAcc, AUC, Sens-W, PPV-W per split
    ├── {train,val,test}_sev_crossing_stratified_error.csv  # MAE/RMSE/Bias by direction class
    ├── {train,val,test}_sev_crossing_confusion_matrix.png  # 3x3 direction confusion matrix
    ├── test_pred_vs_actual.png                   # Scatter with identity line
    ├── test_residual_vs_predicted.png             # Residual diagnostics
    └── test_person_trajectories.png               # Per-person actual vs predicted over time
```

## Metrics

All four metrics are reported on train, val, and test:

| Metric | Description |
|---|---|
| MAE | Mean absolute error (CES-D points) |
| RMSE | Root mean squared error |
| R² | Overall variance explained |
| Within-person R² (median) | **Primary R² metric.** Median of per-person R² values. Uses median (not mean) because persons with very few test observations can produce extreme negative R² that distorts the mean. |
| Between-person R² | Variance explained between person-level means. |

## Baselines (B0–B4)

Each model is compared against five baselines (fit on train, applied to val/test):

| Baseline | Description |
|---|---|
| B0: No Change | Predict 0 (no change) for everyone |
| B1: Population Mean | Predict the training-set mean delta |
| B2: Last Value Carried Forward | Each person's last training delta |
| B3: Person-Specific Mean | Each person's mean training delta |
| B4: Regression to Mean | Shrunk person mean (shrinkage = 0.5) |

## Model Details

The `MixedLMModel` class fits a linear mixed-effects model:

```
y_delta ~ X_fixed + (1 | person_id)            # random intercept only
y_delta ~ X_fixed + (1 + prior_cesd | person_id)  # random intercept + slope
```

- **Estimation**: REML (Restricted Maximum Likelihood)
- **Prediction**: Fixed effects for all observations + random effects (intercept/slopes) for known persons. New persons get fixed-effects-only predictions.
- **Convergence**: Tries `lbfgs` → `bfgs` → `powell`. If random slopes fail, simplifies to random intercept only. Full attempt history saved in `convergence_info.json`.

## Dependencies

Standard scientific Python stack:

```
numpy
pandas
statsmodels
scikit-learn
matplotlib
```
