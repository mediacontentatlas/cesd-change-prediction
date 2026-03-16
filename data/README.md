# Data

## Sample

96 participants monitored over 1 year with biweekly CES-D surveys and continuous Screenome capture (screen-by-screen activity every 5 seconds). Behavioral features are derived from Screenome data: dosage, content diversity, fragmentation, social engagement, and overnight usage.

- ~21 surveys per person (min 10, median 22, up to 25)
- 2,002 total observations (person x biweekly period)
- 64% ever cross clinical threshold (CES-D >= 16)
- 84% have >= 1 change of 5+ points

## Splits

Temporal generalization: 60% train / 20% val / 20% test for each person. Every participant appears in all three splits — the design evaluates whether models generalize to future time points. Person IDs do not leak across splits (same persons, later time).

## Raw (`raw/`)

- `delta_table_final.parquet` — Source table with per-person-period behavioral features, CES-D scores, and computed deltas
- `modeling_table.parquet` — Filtered and prepared modeling table (QC-filtered, lagged)

## Processed (`processed/`)

Pre-split, scaled feature matrices and targets ready for modeling. All paths below are relative to `processed/`.

### Feature Matrices

| File | Shape | Description |
|---|---|---|
| `X_{train,val,test}.npy` | (N, 21) | Base features: prior_cesd + 20 behavioral/demographic |
| `X_dev_{train,val,test}.npy` | (N, ?) | Development feature set |
| `X_all_phenotype_{train,val,test}.npy` | (N, ?) | All features including phenotype clusters |

### Targets and IDs

| File | Description |
|---|---|
| `y_{train,val,test}.npy` | Raw CES-D delta (continuous change from prior survey — target for regression) |
| `pid_{train,val,test}.npy` | Participant IDs per observation |

### Metadata

| File | Description |
|---|---|
| `features.txt` | Ordered list of base feature names |
| `train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv` | Full scaled dataframes with all columns |
| `phenotype_assignments.csv` | Cluster assignments per participant |
| `reactivity_scores.csv` | Reactivity phenotype scores |

## Classification Labels

Classification-specific labels (sev_crossing, personal_sd, balanced_tercile) are derived from `y_*.npy` and stored in `../classification/labels/`. The raw `y_*.npy` files here contain continuous CES-D deltas used by both classification (after labeling) and regression.
