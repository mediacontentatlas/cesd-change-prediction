# Balanced Tercile Labeling — Classification Results


## Label Definition


Observations are **rank-ordered by CESD delta** and assigned to 3 equal-sized bins:
the bottom third (largest decreases) = improving, middle third = stable, top third (largest increases) = worsening.
Ties at bin boundaries are broken randomly (seeded for reproducibility).

Training-set boundary midpoints: lo = -2.00, hi = 2.00

| Class | Assignment rule |
|---|---|
| Improving (0) | Bottom third of ranked CESD deltas |
| Stable (1) | Middle third |
| Worsening (2) | Top third |

### Label Distribution


| Split | N obs | Improving | Stable | Worsening |
|---|---|---|---|---|
| Train | 1196 | 398 (33%) | 398 (33%) | 400 (33%) |
| Val | 395 | 131 (33%) | 131 (33%) | 133 (34%) |
| Test | 411 | 137 (33%) | 137 (33%) | 137 (33%) |

---


## Models — Grid-Searched Per Feature Condition


Hyperparameters were **grid-searched on the validation set per model × feature condition**, using the same search spaces as the original sev_crossing experiments. This ensures each condition gets its own optimal params.


### Best hyperparameters per condition


**prior_cesd only:**


| Model | Best Params | Val BalAcc |
|---|---|---|
| ElasticNet | C=0.001, l1_ratio=0.1 | 0.5052 |
| XGBoost | n_estimators=100, max_depth=3, learning_rate=0.05, min_child_weight=1, subsample=1.0, colsample_bytree=0.8 | 0.5161 |
| LightGBM | n_estimators=100, max_depth=3, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0 | 0.5159 |
| SVM | C=10.0, gamma=0.1, kernel=rbf | 0.5258 |

**base (21):**


| Model | Best Params | Val BalAcc |
|---|---|---|
| ElasticNet | C=0.1, l1_ratio=0.1 | 0.5149 |
| XGBoost | n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=5, subsample=0.8, colsample_bytree=1.0 | 0.5453 |
| LightGBM | n_estimators=100, max_depth=3, learning_rate=0.1, num_leaves=63, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.1 | 0.5478 |
| SVM | C=0.5, gamma=0.0001, kernel=linear | 0.4999 |

**base + behavioral lag (38):**


| Model | Best Params | Val BalAcc |
|---|---|---|
| ElasticNet | C=5.0, l1_ratio=0.1 | 0.5176 |
| XGBoost | n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=3, subsample=0.8, colsample_bytree=1.0 | 0.5478 |
| LightGBM | n_estimators=50, max_depth=7, learning_rate=0.1, num_leaves=63, min_child_samples=30, subsample=0.6, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0 | 0.5477 |
| SVM | C=5.0, gamma=0.01, kernel=rbf | 0.5122 |

**base + behavioral lag + pmcesd (39):**


| Model | Best Params | Val BalAcc |
|---|---|---|
| ElasticNet | C=0.5, l1_ratio=0.99 | 0.5850 |
| XGBoost | n_estimators=100, max_depth=5, learning_rate=0.05, min_child_weight=3, subsample=1.0, colsample_bytree=1.0 | 0.6102 |
| LightGBM | n_estimators=50, max_depth=5, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8, colsample_bytree=1.0, reg_alpha=1.0, reg_lambda=0.1 | 0.6025 |
| SVM | C=0.5, gamma=0.0001, kernel=linear | 0.6055 |


---


## Results — Test Set


### prior_cesd only (1 features)


| Model | AUC | BalAcc | Sens-W | F1-macro | PPV-W |
|---|---|---|---|---|---|
| ElasticNet | 0.665 | 0.474 | 0.226 | 0.456 | 0.388 |
| XGBoost | 0.647 | 0.482 | 0.051 | 0.415 | 0.259 |
| LightGBM | 0.642 | 0.472 | 0.095 | 0.425 | 0.283 |
| SVM | 0.658 | 0.499 | 0.161 | 0.464 | 0.423 |

### base (21) (21 features)


| Model | AUC | BalAcc | Sens-W | F1-macro | PPV-W |
|---|---|---|---|---|---|
| ElasticNet | 0.657 | 0.455 | 0.234 | 0.442 | 0.330 |
| XGBoost | 0.655 | 0.462 | 0.256 | 0.452 | 0.361 |
| LightGBM | 0.658 | 0.470 | 0.314 | 0.464 | 0.384 |
| SVM | 0.651 | 0.455 | 0.263 | 0.444 | 0.336 |

### base + behavioral lag (38) (38 features)


| Model | AUC | BalAcc | Sens-W | F1-macro | PPV-W |
|---|---|---|---|---|---|
| ElasticNet | 0.642 | 0.465 | 0.256 | 0.455 | 0.337 |
| XGBoost | 0.658 | 0.489 | 0.299 | 0.479 | 0.414 |
| LightGBM | 0.640 | 0.472 | 0.285 | 0.463 | 0.371 |
| SVM | 0.640 | 0.472 | 0.329 | 0.468 | 0.385 |

### base + behavioral lag + pmcesd (39) (39 features)


| Model | AUC | BalAcc | Sens-W | F1-macro | PPV-W |
|---|---|---|---|---|---|
| ElasticNet | 0.718 | 0.530 | 0.496 | 0.531 | 0.476 |
| XGBoost | 0.723 | 0.555 | 0.562 | 0.556 | 0.490 |
| LightGBM | 0.732 | 0.557 | 0.555 | 0.559 | 0.481 |
| SVM | 0.721 | 0.557 | 0.496 | 0.556 | 0.519 |

---


## Comparison: Balanced Tercile vs Sev_Crossing (39 features, both tuned)


Both label types use hyperparameters independently tuned on their own validation set.


| Model | Label | AUC | BalAcc | Sens-W | PPV-W |
|---|---|---|---|---|---|
| XGBoost | sev_crossing | 0.906 | 0.834 | 0.838 | 0.356 |
| XGBoost | balanced_tercile | 0.723 | 0.555 | 0.562 | 0.490 |
| LightGBM | sev_crossing | 0.901 | 0.842 | 0.865 | 0.344 |
| LightGBM | balanced_tercile | 0.732 | 0.557 | 0.555 | 0.481 |
| ElasticNet | sev_crossing | 0.829 | 0.691 | 0.730 | 0.248 |
| ElasticNet | balanced_tercile | 0.718 | 0.530 | 0.496 | 0.476 |
| SVM | sev_crossing | 0.841 | 0.696 | 0.649 | 0.304 |
| SVM | balanced_tercile | 0.721 | 0.557 | 0.496 | 0.519 |

---


## Confusion Matrices — 39-feature condition (test set)

### ElasticNet (C=0.5, l1_ratio=0.99)

|  | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 68 | 28 | 41 |
| **True: stable** | 21 | 82 | 34 |
| **True: worsening** | 22 | 47 | **68** |

### XGBoost (depth=5, lr=0.05, min_child=3)

|  | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 81 | 17 | 39 |
| **True: stable** | 26 | 70 | 41 |
| **True: worsening** | 36 | 24 | **77** |

### LightGBM (depth=5, lr=0.05, leaves=15)

|  | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 83 | 14 | 40 |
| **True: stable** | 25 | 70 | 42 |
| **True: worsening** | 40 | 21 | **76** |

### SVM (C=0.5, linear)

|  | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 72 | 28 | 37 |
| **True: stable** | 22 | 89 | 26 |
| **True: worsening** | 23 | 46 | **68** |

---

## Interpretation

1. **Classes are exactly equal** (33/33/33% in all splits) via rank-based assignment with random tiebreaking. Chance-level balanced accuracy = 0.333.

2. **Balanced labels are a substantially harder task.** AUC drops 0.10–0.19 across all models vs sev_crossing. The sev_crossing label is largely determined by proximity to fixed clinical thresholds (16, 24), making prior_cesd a powerful predictor. The balanced tercile label asks models to predict the *relative magnitude* of CESD change — a finer-grained discrimination.

3. **Prior CESD remains the dominant single feature** (AUC 0.64–0.67 alone), but its contribution is sharply reduced vs sev_crossing (AUC 0.88). Behavioral features in the base set barely move the needle (AUC ~0.65–0.66), confirming that behaviors cannot independently predict CESD change magnitude even with balanced classes.

4. **Lag features alone don't help much** (38-feature AUC ≈ 21-feature AUC), but **person_mean_cesd is critical** — the jump from 38 to 39 features produces the largest improvement (AUC +0.06–0.08 across all models).

5. **Model ranking with per-condition tuning:** LightGBM (AUC 0.732, BalAcc 0.557) ≈ XGBoost (0.723, 0.555) ≈ SVM (0.721, 0.557) > ElasticNet (0.718, 0.530). SVM with a linear kernel is competitive with tree models under balanced labels — the nonlinear decision boundaries that helped sev_crossing are less important here.

6. **Worsening sensitivity** is 0.50–0.56 for all models at 39 features, with PPV 0.48–0.52. Under balanced labels, precision-recall is more symmetric because the base rate for each class is 33%.

7. **Per-condition grid search did not substantially change results** compared to the fixed-param run — the balanced tercile task is fundamentally harder regardless of tuning.
