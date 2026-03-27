# ElasticNet Regression

Continuous prediction of CES-D delta (change in depression score) using L1+L2 regularized linear regression.

## Quick Start

```bash
cd regression/elasticnet

# Run ALL 11 conditions end-to-end (train + posthoc + performer tiers):
python scripts/run_all_conditions.py

# Dry run (see commands without executing):
python scripts/run_all_conditions.py --dry-run

# Run a single condition:
python scripts/run_all_conditions.py --only base

# Skip per-person plots (faster):
python scripts/run_all_conditions.py --skip-plots

# Rebuild summary CSV from existing outputs (no retraining):
python scripts/run_all_conditions.py --summary-only

# Generate report figures/tables after all conditions are done:
python scripts/build_report.py
```

## Feature Ablation Conditions

### Required (parity with classification)

| Condition | Features | N |
|---|---|---|
| `prior_cesd` | prior_cesd only | 1 |
| `base` | All 21 base features | 21 |
| `base_lag` | Base + 17 behavioral lag features | 38 |
| `base_lag_pmcesd` | Base + lag + person_mean_cesd | 39 |

### Extra (original screenome variants)

| Condition | Features | N |
|---|---|---|
| `dev` | Base + 8 within-person deviation | 29 |
| `pheno` | Base + 5 phenotype | 26 |
| `pid` | Base + ~96 PID one-hot | ~117 |
| `dev_pheno` | Base + dev + pheno | 34 |
| `dev_pid` | Base + dev + PID OHE | ~125 |
| `pheno_pid` | Base + pheno + PID OHE | ~122 |
| `dev_pheno_pid` | Base + dev + pheno + PID OHE | ~130 |

## Pipeline (per condition)

```
1. Grid search: fit on Train, evaluate on Val -> select best (alpha, l1_ratio)
2. Dev model:   fit on Train with best params -> predict Train + Val
3. Final model:  refit on Train+Val with SAME params (no re-tuning) -> predict Test
```

Hyperparameters are locked after step 1. The orchestrator then runs:
- Posthoc direction classification (using labels from `classification/labels/sev_crossing/`)
- Performer tier analysis (high/medium/low by per-person MAE)

## Scripts

| Script | Purpose |
|---|---|
| `scripts/train_elasticnet.py` | Train a single condition. Called by the orchestrator. |
| `scripts/posthoc_direction.py` | Direction analysis using classification labels. |
| `scripts/run_all_conditions.py` | Top-level orchestrator. Runs train + posthoc + performer for each condition. |
| `scripts/build_report.py` | Generate slide-ready figures and tables from all outputs. |

### Running individual scripts

```bash
# Train a single condition directly:
python scripts/train_elasticnet.py --condition base --run-test

# Train without test evaluation (dev phase only):
python scripts/train_elasticnet.py --condition base

# Posthoc for a specific condition and label type:
python scripts/posthoc_direction.py --condition base --label-type sev_crossing
python scripts/posthoc_direction.py --condition base --label-type personal_sd
python scripts/posthoc_direction.py --condition base --label-type balanced_tercile
```

## Metrics

| Metric | Description |
|---|---|
| MAE | Mean absolute error (CES-D points) |
| RMSE | Root mean squared error |
| Within-person R2 | Variance explained within individuals (primary R2) |
| Between-person R2 | Variance explained between individuals |

Posthoc direction metrics: BalAcc, AUC (OvR macro), Sens-W, PPV-W, confusion matrix.

## Output Structure

```
models/{condition}/
  model.joblib                    # Dev model (fit on Train)
  final_model.joblib              # Final model (fit on Train+Val)
  best_params.yaml                # Locked hyperparameters + dev metrics
  final_params.yaml               # Final model metadata + test metrics
  grid_search_results.csv         # Full grid search results
  feature_coefficients.csv        # Dev model coefficients
  final_feature_coefficients.csv  # Final model coefficients
  y_pred_train.npy                # Dev model predictions
  y_pred_val.npy                  # Dev model predictions
  y_pred_test.npy                 # Final model predictions
  plots/                          # Grid search diagnostic plots
  performer_tiers_{split}.csv
  performer_tier_stats_{split}.csv
  performer_analysis_{split}.csv

models/comparison_summary.csv     # Cross-condition ranking

../../posthoc/elasticnet/{condition}/{label_type}/   # Direction classification
  classification_metrics.csv
  stratified_error_{split}.csv
  direction_per_person_{split}.csv
  confusion_matrix_{split}.png
  plots/per_person/               # Per-person CMs + trajectories

reports/                          # Slide-ready figures and tables
```

## Configuration

`configs/elasticnet.yaml` -- hyperparameter grid: 13 alphas (0.0001-100) x 6 l1_ratios (0.1-0.99) = 78 combinations.
