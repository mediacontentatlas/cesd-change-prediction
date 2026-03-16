# Screenome Mental Health Prediction

Depression is not a stable state — symptoms show considerable within-person variability over time (van Eeden et al., 2019), and smartphone behavior fluctuates within individuals in ways that track mood prospectively (Beyens et al., 2024). Most prior work asks "who is depressed?" via cross-sectional or between-person designs (Te Molder et al., 2023). Even longitudinal studies examine associations, not prospective prediction of symptom change within individuals (Stamatis et al., 2024). Studies of 1 year or longer with individual-level prediction remain a critical gap (Amin et al., 2025).

This project uses passive smartphone data from the Human Screenome Project (Reeves et al., 2021) — screen-by-screen activity captured every 5 seconds — to predict how a person's depressive symptoms will change over the next two weeks. Prior work demonstrated high within-person covariation between Screenome-derived media use patterns and mental health metrics (Cerit et al., 2025). Here we move from association to prediction.

## Research Questions

**Can biweekly CES-D trajectories and Screenome-derived behavioral features predict how a person's depressive symptoms will change over the next two weeks — and what does each data source contribute?**

- **Approach 1 — Regression:** Predict the magnitude of CES-D change from one biweek to the next.
- **Approach 2 — Classification:** Predict the direction of change — is this person improving, stable, or worsening?

Secondary questions:
- What does prior symptom history vs. behavioral data contribute?
- For whom do models work well, and for whom do they fail?
- What behavioral signatures precede symptom deterioration?

## Sample

- 96 participants, ~21 surveys per person (min 10, median 22, up to 25)
- 64% ever cross clinical threshold (CES-D >= 16)
- 84% have >= 1 change of 5+ points
- 1 year of data per participant

## Evaluation Design

Temporal generalization: 60% train / 20% val / 20% test for each person. Every participant appears in all three splits — the test set evaluates whether the model generalizes to future time points for the same individuals. A separate cold-start analysis (repeated leave-group-out CV) evaluates generalization to new, unseen individuals.

---

## Project Structure

```
├── data/                   # Raw parquet tables + processed train/val/test arrays
│   ├── DATA_README.md      # Comprehensive data dictionary
│   └── README.md           # Quick reference for splits, files, and feature matrices
├── classification/         # Classification task (mood direction prediction)
│   ├── labels/             # Shared label definitions (sev_crossing, personal_sd, balanced_tercile)
│   ├── models/             # Trained models organized by label x model type
│   │   ├── sev_crossing/       # {elasticnet, xgboost, lightgbm, svm}/
│   │   ├── personal_sd/        # {elasticnet, xgboost, lightgbm, svm}/
│   │   ├── balanced_tercile/   # {elasticnet, xgboost, lightgbm, svm}/
│   │   ├── bootstrap_ci/       # Bootstrap CI results
│   │   ├── deployment_scenarios/# Deployment ladder results
│   │   ├── feature_importance/  # Gain, SHAP, ElasticNet coefficients (per label)
│   │   └── posthoc/            # Transition detection, phenotype analyses
│   ├── scripts/            # Training and evaluation scripts
│   ├── posthoc/            # Posthoc analysis scripts
│   └── reports/            # Results and writeups
├── regression/             # Regression task (CES-D delta prediction)
│   ├── elasticnet/         # ElasticNet regression (scripts, models, reports)
│   ├── mixedlm/            # Mixed-effects linear models (scripts, models, reports)
│   └── posthoc/            # Direction analysis using classification labels
└── reports/                # Cross-task figures and summaries
```

---

## For Teammates: How to Contribute

### Where to put your code

| Task | Scripts go in | Models go in | Results go in |
|---|---|---|---|
| **Classification** (new models, analyses) | `classification/scripts/` | `classification/models/<label_type>/<model_name>/` | `classification/reports/` |
| **Regression** (ElasticNet) | `regression/elasticnet/scripts/` | `regression/elasticnet/models/` | `regression/elasticnet/reports/` |
| **Regression** (Mixed-effects LM) | `regression/mixedlm/scripts/` | `regression/mixedlm/models/` | `regression/mixedlm/reports/` |
| **Regression posthoc** (direction analysis) | `regression/posthoc/` | `regression/posthoc/` | `regression/posthoc/` |
| **Cross-task figures** | project root or `reports/` | — | `reports/` |

### Using shared data

All teammates use the same pre-split, scaled feature matrices and targets:

```python
import numpy as np

# Feature matrices (already scaled)
X_train = np.load("data/processed/X_train.npy")  # (1196, 21) base features
X_val   = np.load("data/processed/X_val.npy")    # (395, 21)
X_test  = np.load("data/processed/X_test.npy")   # (411, 21)

# Continuous CES-D delta (regression target)
y_train = np.load("data/processed/y_train.npy")
y_val   = np.load("data/processed/y_val.npy")
y_test  = np.load("data/processed/y_test.npy")

# Person IDs (for grouping, mixed-effects models)
pid_train = np.load("data/processed/pid_train.npy")
pid_val   = np.load("data/processed/pid_val.npy")
pid_test  = np.load("data/processed/pid_test.npy")
```

Feature names are listed in `data/processed/features.txt`. Full data dictionary in `data/DATA_README.md`.

### Using classification labels

Classification labels are stored in `classification/labels/`. Each label type has `y_train.npy`, `y_val.npy`, `y_test.npy` (integers: 0=improving, 1=stable, 2=worsening) plus a `label_info.yaml` with distribution info.

```python
# Load classification labels
y_cls = np.load("classification/labels/sev_crossing/y_test.npy")  # or personal_sd/ or balanced_tercile/
```

**Regression teammates**: Use these labels for posthoc direction analysis — e.g., stratify your regression errors by direction class. See `regression/posthoc/README.md` for an example.

### Three label types

| Label | What it captures | Class balance | When to use |
|---|---|---|---|
| **sev_crossing** | Clinical severity boundary crossing (CES-D 16/24 thresholds) | 11% / 80% / 9% | Primary label. Clinically meaningful change. |
| **personal_sd** | Person-specific SD-based change (k=1.0) | 12% / 78% / 10% | Sensitivity analysis. Person-centered definition of unusual change. |
| **balanced_tercile** | Rank-based equal-sized terciles of CES-D delta | 33% / 33% / 33% | Sensitivity analysis. Forces balanced classes. |

### What metrics to report

Always report on **train / val / test** splits (not just test).

**Classification:**

| Metric | Description | Why |
|---|---|---|
| AUC (OvR) | One-vs-rest macro-averaged area under ROC | Primary discrimination metric |
| BalAcc | Balanced accuracy (macro recall) | Handles class imbalance |
| Sens-W | Sensitivity for worsening class (recall of class 2) | Clinical priority: catch deterioration |
| PPV-W | Positive predictive value for worsening class | False alarm rate context |

**Regression:**

| Metric | Description | Why |
|---|---|---|
| MAE | Mean absolute error | Interpretable scale |
| RMSE | Root mean squared error | Penalizes large errors |
| Within-person R² | Variance explained within individuals | CES-D delta has near-zero between-person ICC, so this is the relevant R² |
| Between-person R² | Variance explained between individuals | Report for completeness but expect it to be low |

**Posthoc direction analysis** (regression → classification labels):

Report regression error (MAE, RMSE, Bias) stratified by direction class (improving/stable/worsening). This reveals whether regression models systematically fail for one direction of change.

In addition, report **posthoc classification metrics** — treat the regression-predicted direction as a classifier and report:

| Metric | Description |
|---|---|
| BalAcc | Balanced accuracy (macro recall) |
| AUC (OvR) | One-vs-rest macro-averaged AUC |
| Sens-W | Sensitivity for worsening class |
| PPV-W | Positive predictive value for worsening class |
| Confusion matrix | 3×3 matrix (improving / stable / worsening) |

See `regression/posthoc/README.md` for details and example code.

### Feature ablation requirements

Your analyses should include **at minimum** the same feature conditions used in classification, so results are directly comparable:

| Condition | Features | N features |
|---|---|---|
| prior_cesd only | `prior_cesd` | 1 |
| base | All 21 base features | 21 |
| base + behavioral lag | Base + 17 lagged behavioral features | 38 |
| base + behavioral lag + pmcesd | Base + lag + `person_mean_cesd` | 39 |

You may add additional ablation conditions specific to your model (e.g., different regularization, feature subsets), but always include these four so we can compare across tasks and models.

---

## Key Results (Classification)

**Best model**: XGBoost, 39 features (behavioral + lag + person_mean_cesd), sev_crossing label

| Metric | Value | 95% CI |
|---|---|---|
| AUC (OvR) | **0.906** | [0.881, 0.929] |
| Balanced Accuracy | **0.834** | [0.784, 0.882] |
| Worsening Sensitivity | **0.838** | [0.707, 0.947] |
| Worsening PPV | **0.356** | [0.253, 0.464] |

All min→sev transitions caught (9/9 = 100%). Cold-start AUC = 0.821 (generalizes to unseen persons). `person_mean_cesd` is the single most impactful feature addition (+0.031 AUC, p=0.009).

See [classification/README.md](classification/README.md) for full details.

## Key Results (Regression)

*(To be filled — regression results will be added when available.)*

See [regression/README.md](regression/README.md).

---

## Quick Links

| What | Where |
|---|---|
| **Data dictionary** | [data/DATA_README.md](data/DATA_README.md) |
| **Classification overview** | [classification/README.md](classification/README.md) |
| **Regression overview** | [regression/README.md](regression/README.md) |
| **Classification results summary** | [classification/reports/project_results_summary.md](classification/reports/project_results_summary.md) |
| **Feature importance (multi-method, multi-label)** | [classification/reports/feature_importance_report.md](classification/reports/feature_importance_report.md) |
| **Bootstrap CIs and feature ablation** | [classification/reports/bootstrap_ci_results.md](classification/reports/bootstrap_ci_results.md) |
| **Deployment scenarios** | [classification/reports/deployment_results.md](classification/reports/deployment_results.md) |
| **Phenotype posthoc analysis** | [classification/reports/phenotype_posthoc_writeup.md](classification/reports/phenotype_posthoc_writeup.md) |
