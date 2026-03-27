# ElasticNet Regression -- Experiment Log

## Overview

We predict the continuous CES-D delta (biweekly change in depression score) from Screenome-derived smartphone behavioral features using ElasticNet regression. This complements the classification task (which predicts direction of change) by asking: **how much** will symptoms change?

ElasticNet was chosen because:
- L1 penalty (Lasso) performs automatic feature selection, zeroing out irrelevant predictors
- L2 penalty (Ridge) handles correlated features gracefully, which is expected with behavioral metrics (e.g., daily screens and daily switches are correlated)
- The mixing parameter `l1_ratio` lets us tune the balance between sparsity and stability

## Data

**Source**: Temporal train/val/test split from the shared screenome dataset.

| Split | N rows | Purpose |
|---|---|---|
| Train | 1,196 | Grid search fitting + dev model fitting |
| Val | 395 | Hyperparameter selection (never used for fitting) |
| Test | 411 | Final held-out evaluation |

- **Participants**: 96, all appearing in all three splits (longitudinal, not cross-sectional)
- **Target**: `target_cesd_delta` -- unscaled change in CES-D score between consecutive biweekly surveys (range approx. -52 to +52)
- **Features**: 21 base features (scaled), loaded from `data/processed/X_{split}.npy`
- **Person IDs**: `data/processed/pid_{split}.npy` (needed for within-person R2 and performer analysis)

The temporal split preserves the longitudinal ordering, preventing future data from leaking into training.

## Feature Ablation Conditions

We run 11 conditions: 4 required (matching the classification task for direct comparison) and 7 extras (from the original feature-set comparison).

### Required conditions (parity with classification)

| Condition | Description | N features | How built |
|---|---|---|---|
| `prior_cesd` | Baseline only | 1 | Column 0 of X_train |
| `base` | All base features | 21 | X_train as-is |
| `base_lag` | Base + lagged features | 38 | Base + lag-1 of 17 behavioral features (excl. static demographics + clinical lags), computed from `*_scaled.csv` |
| `base_lag_pmcesd` | Base + lag + person mean | 39 | base_lag + person_mean_cesd (per-person mean of prior_cesd from training data, population mean fallback for unseen persons) |

**Lag feature construction**: For each of the 17 time-varying behavioral features, we compute the lag-1 value (previous period's value for the same person). Static demographics (age, gender_mode_1, gender_mode_2) are excluded since they are constant across periods. Clinical lags (lag_prior_cesd, lag_cesd_delta) are excluded per ablation (see DATA_README.md). Missing lags (first observation per person) are filled with 0.

**person_mean_cesd**: Computed as the mean of `prior_cesd` across all training observations for each person. For persons unseen in training (none in our data, but handled for robustness), we fall back to the population mean. This also matches the classification pipeline.

### Extra conditions (original variants)

| Condition | Description | N features | Data source |
|---|---|---|---|
| `dev` | Base + within-person deviation | 29 | `X_dev_{split}.npy` (8 cols) appended to base |
| `pheno` | Base + phenotype | 26 | `X_all_phenotype_{split}.npy` (5 cols) appended |
| `pid` | Base + participant ID | ~117 | PID one-hot encoded in memory (OneHotEncoder), prepended to base |
| `dev_pheno` | Base + dev + pheno | 34 | hstack of base + dev + pheno |
| `dev_pid` | Base + dev + PID OHE | ~125 | hstack of PID OHE + base + dev |
| `pheno_pid` | Base + pheno + PID OHE | ~122 | hstack of PID OHE + base + pheno |
| `dev_pheno_pid` | All feature types | ~130 | hstack of PID OHE + base + dev + pheno |

The PID one-hot encoding is fitted on training PIDs only. Unknown PIDs at val/test time get a zero vector (via `handle_unknown="ignore"`).

## Hyperparameter Search

**Grid**: 13 alpha values (0.0001 to 100, log scale) x 6 l1_ratio values (0.1 to 0.99) = **78 combinations** per condition.

| Parameter | Values |
|---|---|
| `alpha` | 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0 |
| `l1_ratio` | 0.1, 0.5, 0.7, 0.9, 0.95, 0.99 |
| `max_iter` | 10,000 |

**Selection criterion**: Minimum validation MAE. The grid search fits each combination on Train and evaluates on Val. The (alpha, l1_ratio) pair with the lowest Val MAE is selected. Hyperparameters are **locked** after this step and they are never re-tuned.

The configuration is stored in `configs/elasticnet.yaml`.

## Training Protocol

The pipeline follows a strict 3-phase protocol to prevent data leakage:

### Phase 1: Grid search
- Fit ElasticNet on **Train** for each of 78 (alpha, l1_ratio) pairs
- Evaluate each on **Val** (MAE)
- Select the pair with the lowest Val MAE
- **Hyperparameters are locked** -- no further tuning

### Phase 2: Dev model
- Refit ElasticNet on **Train only** with the locked hyperparameters
- Generate predictions for Train (in-sample diagnostic) and Val (out-of-sample holdout)
- Val metrics from this model are the valid development-phase numbers

### Phase 3: Final model (test evaluation)
- Refit ElasticNet on **Train+Val combined** with the same locked hyperparameters (no re-tuning)
- Generate predictions for Test
- Test metrics are the final unbiased generalization estimate

**Data leakage prevention**: Val is never used during grid search fitting. Test is completely blind until Phase 3. The Train+Val combination in Phase 3 uses the same locked hyperparameters -- no decisions are made based on Test.

## Evaluation Metrics

### Regression metrics (reported on Train / Val / Test)

| Metric | Description | Why |
|---|---|---|
| MAE | Mean absolute error | Interpretable scale (CES-D points); primary comparison metric |
| RMSE | Root mean squared error | Penalizes large errors, which are clinically important |
| Within-person R2 | Pooled within-person variance explained | **Primary R2 metric.** CES-D delta has near-zero between-person ICC (~0.004), so standard R2 would be misleading |
| Between-person R2 | R2 of person-level mean predictions vs true means | Reported for completeness; expected to be low |

**Within-person R2 computation**: For each person, compute SS_res (sum of squared residuals) and SS_tot (sum of squared deviations from that person's mean). Pool across all persons: `1 - sum(SS_res_i) / sum(SS_tot_i)`.

### Posthoc direction analysis

We use pre-computed classification labels from `classification/labels/` to evaluate whether the regression model can predict the **direction** of symptom change (improving / stable / worsening), not just the magnitude.

**Label types used**:
- `sev_crossing` (primary): Clinical severity boundary crossing based on CES-D thresholds (16 for moderate, 24 for severe). A person is "improving" if their predicted post-period severity is lower than their current severity, "worsening" if higher, "stable" if unchanged.
- `personal_sd` (sensitivity analysis): Person-specific SD-based thresholds. Predictions outside +/- k*SD (k=1.0) of a person's training-period variability are classified as improving/worsening.

**Why these two, not balanced_tercile**: The classification side ran all three label types, but `balanced_tercile` (rank-based equal thirds) is omitted here. For a regression-as-classifier evaluation, balanced_tercile assigns labels by ranking all predictions -- the bottom third becomes "improving," the top third becomes "worsening." Predicting these labels correctly only requires that high predictions tend to go to people who actually worsened and low predictions to people who actually improved; it does not require accurate magnitude estimates. That is a weaker signal than what continuous regression metrics (MAE, within-person R2) already measure directly. It would also artificially inflate direction accuracy without clinical meaning. In contrast, `sev_crossing` tests clinically meaningful boundary crossings and `personal_sd` tests whether the regression captures within-person dynamics relative to each person's typical variability. Both are genuinely informative about the regression model's clinical utility.

**How regression predictions become direction predictions**: We apply the same labeling function used to create the ground-truth labels, but on `y_pred` instead of `y_true`. For sev_crossing, this means computing `severity(prior_cesd + y_pred)` and comparing with `severity(prior_cesd)`. For personal_sd, we compare `y_pred` against per-person SD thresholds derived from training data.

**Metrics reported**:

| Metric | Description |
|---|---|
| Stratified MAE/RMSE/Bias | Regression error broken down by direction class |
| Balanced Accuracy | Macro recall across 3 classes |
| AUC (OvR macro) | One-vs-rest macro-averaged AUC using continuous y_pred as soft scores |
| Sens-W | Sensitivity for the worsening class (recall of class 2) |
| PPV-W | Positive predictive value for the worsening class |
| Confusion matrix | 3x3 (improving / stable / worsening) |

### Performer tier analysis

Participants are classified into performance tiers based on per-person MAE:
- **High performers** (good predictions): Per-person MAE below 25th percentile
- **Medium performers**: Between 25th and 75th percentile
- **Low performers** (poor predictions): Per-person MAE above 75th percentile

For each tier, we report MAE statistics and compare behavioral feature profiles (aggregated from `train_scaled.csv`) to understand **for whom** the model works well vs. poorly.

## Reproducibility

### Dependencies

- Python 3.10+
- numpy, pandas, scikit-learn, matplotlib, seaborn, pyyaml, joblib

### Running

```bash
cd regression/elasticnet

# Full pipeline (all 11 conditions, both posthoc label types):
python scripts/run_all_conditions.py

# Single condition:
python scripts/run_all_conditions.py --only base

# Generate report after all conditions complete:
python scripts/build_report.py
```

### Scripts

| Script | Purpose |
|---|---|
| `scripts/train_elasticnet.py` | Train one condition (grid search + dev model + final model) |
| `scripts/posthoc_direction.py` | Posthoc direction analysis for one condition and label type |
| `scripts/run_all_conditions.py` | Orchestrator: runs train + posthoc (both label types) + performer tiers for all conditions |
| `scripts/build_report.py` | Generate slide-ready figures and tables |

All scripts are self-contained (no external package imports beyond standard ML libraries). Default paths resolve relative to the script location, so running from `regression/elasticnet/` requires no path arguments.