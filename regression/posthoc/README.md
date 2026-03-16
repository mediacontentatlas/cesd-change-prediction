# Regression Posthoc: Direction Analysis

Use classification labels from `../../classification/labels/` to analyze regression model performance by predicted mood direction.

## Available Labels

| Label | Path | Description |
|---|---|---|
| sev_crossing | `../../classification/labels/sev_crossing/` | Clinical severity boundary crossing (CESD thresholds 16, 24) |
| personal_sd | `../../classification/labels/personal_sd/` | Person-specific SD-based change (k=1.0) |
| balanced_tercile | `../../classification/labels/balanced_tercile/` | Rank-based equal-sized terciles |

Each folder contains `y_train.npy`, `y_val.npy`, `y_test.npy` with values 0 (improving), 1 (stable), 2 (worsening).
