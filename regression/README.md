# Regression: Predicting Magnitude of Symptom Change

Predict the continuous CES-D delta (change from one biweekly survey to the next) from Screenome-derived behavioral features and prior symptom history. This complements the classification task (direction prediction) by asking: how *much* will symptoms change?

## Models

- `elasticnet/` — ElasticNet regression (scripts, models, reports)
- `mixedlm/` — Mixed-effects linear models (scripts, models, reports)
- `posthoc/` — Direction analysis using classification labels

## Data

Use the same pre-split, scaled feature matrices as classification:

```python
import numpy as np

# Feature matrices
X_train = np.load("../data/processed/X_train.npy")  # (1196, 21)
X_val   = np.load("../data/processed/X_val.npy")    # (395, 21)
X_test  = np.load("../data/processed/X_test.npy")   # (411, 21)

# Continuous CES-D delta (this is your regression target)
y_train = np.load("../data/processed/y_train.npy")
y_val   = np.load("../data/processed/y_val.npy")
y_test  = np.load("../data/processed/y_test.npy")

# Person IDs (needed for mixed-effects models and within-person R²)
pid_train = np.load("../data/processed/pid_train.npy")
pid_val   = np.load("../data/processed/pid_val.npy")
pid_test  = np.load("../data/processed/pid_test.npy")
```

Feature names are listed in `../data/processed/features.txt`. Full data dictionary in `../data/DATA_README.md`.

## Metrics (report on train / val / test)

| Metric | Description | Why it matters |
|---|---|---|
| MAE | Mean absolute error | Interpretable scale (CES-D points) |
| RMSE | Root mean squared error | Penalizes large errors — clinically important |
| Within-person R² | Variance explained within individuals | **Primary R² metric.** CES-D delta has near-zero between-person ICC (ICC ~ 0; person means explain only 0.4% of delta variance), so within-person R² guards against inflated estimates from person-level random effects |
| Between-person R² | Variance explained between individuals | Report for completeness; expect it to be low |

**Important**: Always report all four metrics on train, val, AND test. Report within-person R² as the primary R² — a high overall R² that comes from person-level random effects is misleading for this target.

## Using Classification Labels for Posthoc Direction Analysis

Classification labels are stored in `../classification/labels/`. Each label type has:
- `y_train.npy`, `y_val.npy`, `y_test.npy` — integer arrays (0=improving, 1=stable, 2=worsening)
- `label_info.yaml` — label definition and distribution

Three label types are available: `sev_crossing` (primary, clinical threshold crossing), `personal_sd` (person-specific SD-based), `balanced_tercile` (rank-based equal thirds). Use at least `sev_crossing` for posthoc; the others are optional sensitivity analyses.

### What to report for posthoc direction analysis

Stratify your regression metrics by direction class to reveal whether regression models systematically over- or under-predict for each direction:

```python
import numpy as np

# Load classification labels
y_labels = np.load("../classification/labels/sev_crossing/y_test.npy")

# Load your regression predictions and targets
y_pred = np.load("elasticnet/models/y_pred_test.npy")
y_true = np.load("../data/processed/y_test.npy")

# Analyze prediction error by direction class
for cls, name in enumerate(["improving", "stable", "worsening"]):
    mask = y_labels == cls
    mae = np.abs(y_pred[mask] - y_true[mask]).mean()
    rmse = np.sqrt(np.mean((y_pred[mask] - y_true[mask])**2))
    bias = np.mean(y_pred[mask] - y_true[mask])
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, Bias={bias:+.3f}, N={mask.sum()}")
```

Report a table like:

| Direction class | N | MAE | RMSE | Bias |
|---|---|---|---|---|
| Improving | 44 | ? | ? | ? |
| Stable | 330 | ? | ? | ? |
| Worsening | 37 | ? | ? | ? |

This is critical for understanding whether the regression model can predict the *direction* of change, not just the magnitude.

### Posthoc classification metrics

Beyond stratified regression error, you should also treat the regression-predicted direction as a classifier and report these **classification metrics** on the posthoc direction labels:

| Metric | Description |
|---|---|
| BalAcc | Balanced accuracy (macro recall) |
| AUC (OvR) | One-vs-rest macro-averaged AUC |
| Sens-W | Sensitivity for worsening class (recall of class 2) |
| PPV-W | Positive predictive value for worsening class |
| Confusion matrix | 3×3 matrix (improving / stable / worsening) |

This lets us directly compare how well regression-based direction prediction competes with the dedicated classifiers.

### Feature ablation requirements

Your regression experiments should include **at minimum** the same four feature conditions used in the classification task, so results are directly comparable:

| Condition | Features | N features |
|---|---|---|
| `prior_cesd` | `prior_cesd` | 1 |
| `base` | All 21 base features | 21 |
| `base_dev` | Base + 8 within-person deviation features (`X_dev_*.npy`) | 29 |
| `base_dev_pmcesd` | Base + dev + `person_mean_cesd` | 30 |

You may add additional ablation conditions specific to your model, but always include these four.

## Directory Layout

```
regression/
├── elasticnet/
│   ├── scripts/     # Training and hyperparameter search
│   ├── models/      # Saved models, predictions (y_pred_*.npy)
│   └── reports/     # Results tables, figures
├── mixedlm/
│   ├── scripts/
│   ├── models/
│   └── reports/
└── posthoc/
    ├── README.md    # How to run posthoc direction analysis
    └── ...          # Scripts and results
```

## Adding Your Work

1. Place scripts in `<model>/scripts/`, trained models and predictions in `<model>/models/`, results in `<model>/reports/`
2. Save test predictions as `y_pred_test.npy` so posthoc scripts can use them
3. Report all regression metrics (MAE, RMSE, within-person R², between-person R²) on train/val/test
4. Run all four feature ablation conditions (prior_cesd, base, base_dev, base_dev_pmcesd) — add your own on top
5. Run posthoc direction analysis using classification labels (at minimum `sev_crossing`), reporting both stratified regression error AND classification metrics (BalAcc, AUC, Sens-W, PPV-W, confusion matrix)
6. When you have final results, add summary tables/figures to the top-level `../reports/` folder

## Results

### MixedLM — Feature Ablation (Test Set)

| Condition | N features | MAE | RMSE | R² | Within-R² (median) |
|---|---|---|---|---|---|
| `prior_cesd` | 1 | 4.17 | 6.17 | 0.303 | 0.180 |
| `base` | 21 | 4.24 | 6.23 | 0.290 | 0.211 |
| `base_dev` | 29 | 4.30 | 6.32 | 0.270 | 0.211 |
| `base_dev_pmcesd` | 30 | 3.80 | 5.85 | 0.376 | 0.389 |

### MixedLM — Random Effects Sweep (Test Set, 21 base features)

| # | Model | Random Effects | MAE | RMSE | R² |
|---|---|---|---|---|---|
| 1 | Pooled (no PID) | None | 4.707 | 6.920 | 0.125 |
| 2 | Intercept only | Intercept | 4.240 | 6.235 | 0.290 |
| 3 | + prior_cesd slope | + prior_cesd | 4.235 | 6.255 | 0.285 |
| 4 | + prior + switches | + prior_cesd + switches | 4.242 | 6.190 | 0.300 |
| 5 | + prior + social | + prior_cesd + social_ratio | 4.237 | 6.366 | 0.260 |
| 6 | + prior + social ext | + prior_cesd + social_scr + ratio | 4.236 | 6.270 | 0.282 |
| 7 | switches + screens | + switches + screens | 4.301 | 6.195 | 0.299 |
| 8 | **prior + sw + scr** | + prior_cesd + switches + screens | **4.236** | **6.157** | **0.307** |
| 9 | + dev features | + prior_cesd (29 features) | 4.314 | 6.362 | 0.261 |

All 9 models match the screenome_mh_pred repo results exactly. Best random-effects model: #8 (lowest RMSE, highest R²). Best overall: `base_dev_pmcesd` ablation (MAE=3.80, R²=0.376). See `mixedlm/README.md` for full details.

### ElasticNet — Feature Ablation (Test Set)

| Condition | N features | Retained | MAE | RMSE | R² | W-R² (med) |
|---|---|---|---|---|---|---|
| `prior_cesd` | 1 | 1 | 4.58 | 7.39 | -0.006 | -0.042 |
| `base` | 21 | 1 | 4.58 | 7.39 | -0.006 | -0.042 |
| `base_lag` | 38 | 1 | 4.58 | 7.39 | -0.006 | -0.042 |
| `base_lag_pmcesd` | 39 | 2 | **4.13** | **6.28** | **0.279** | **0.186** |

Six conditions (prior_cesd, base, base_lag, dev, pheno, dev_pheno) collapse to an identical prior_cesd-only model. Best overall: `base_lag_pmcesd` (MAE=4.13, R²=0.279). See `elasticnet/reports/regression_results.md` for full details.
