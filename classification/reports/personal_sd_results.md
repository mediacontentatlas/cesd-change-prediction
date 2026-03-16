# Personal SD Labeling — Classification Results


## Label Definition


Each person's SD of CESD delta is computed from their training data (floor = 3.0).
An observation is labeled **worsening** if delta > 1.0 x person_SD, 
**improving** if delta < -1.0 x person_SD, and **stable** otherwise.

Population SD (training): 7.43
k multiplier: 1.0

### Label Distribution


| Split | N obs | Improving | Stable | Worsening |
|---|---|---|---|---|
| Train | 1196 | 146 (12%) | 904 (76%) | 146 (12%) |
| Val | 395 | 36 (9%) | 309 (78%) | 50 (13%) |
| Test | 411 | 50 (12%) | 320 (78%) | 41 (10%) |

---


## Models — Grid-Searched Per Feature Condition


Hyperparameters were **grid-searched on the validation set per model x feature condition**, using the same search spaces as sev_crossing and balanced_tercile experiments.


### Best hyperparameters per condition


**prior_cesd only:**


| Model | Best Params | Val BalAcc |
|---|---|---|
| ElasticNet | C=0.005, l1_ratio=0.1 | 0.4683 |
| XGBoost | n_estimators=100, max_depth=3, learning_rate=0.05, min_child_weight=3, subsample=1.0, colsample_bytree=0.8 | 0.4759 |
| LightGBM | n_estimators=50, max_depth=5, learning_rate=0.01, num_leaves=63, min_child_samples=10, subsample=0.6, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=0.1 | 0.4820 |
| SVM | C=0.1, gamma=0.0001, kernel=linear | 0.4626 |

**base (21):**


| Model | Best Params | Val BalAcc |
|---|---|---|
| ElasticNet | C=0.005, l1_ratio=0.1 | 0.4732 |
| XGBoost | n_estimators=100, max_depth=3, learning_rate=0.05, min_child_weight=1, subsample=0.8, colsample_bytree=1.0 | 0.4871 |
| LightGBM | n_estimators=50, max_depth=5, learning_rate=0.01, num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=1.0, reg_alpha=0.1, reg_lambda=0.1 | 0.4941 |
| SVM | C=5.0, gamma=0.001, kernel=rbf | 0.4674 |

**base + behavioral lag (38):**


| Model | Best Params | Val BalAcc |
|---|---|---|
| ElasticNet | C=0.05, l1_ratio=0.99 | 0.4678 |
| XGBoost | n_estimators=100, max_depth=5, learning_rate=0.01, min_child_weight=1, subsample=1.0, colsample_bytree=0.8 | 0.4763 |
| LightGBM | n_estimators=100, max_depth=5, learning_rate=0.01, num_leaves=15, min_child_samples=30, subsample=1.0, colsample_bytree=1.0, reg_alpha=0.1, reg_lambda=0.1 | 0.4713 |
| SVM | C=1.0, gamma=0.0001, kernel=linear | 0.4684 |

**base + behavioral lag + pmcesd (39):**


| Model | Best Params | Val BalAcc |
|---|---|---|
| ElasticNet | C=0.1, l1_ratio=0.5 | 0.5976 |
| XGBoost | n_estimators=100, max_depth=5, learning_rate=0.01, min_child_weight=3, subsample=1.0, colsample_bytree=1.0 | 0.6099 |
| LightGBM | n_estimators=100, max_depth=5, learning_rate=0.01, num_leaves=15, min_child_samples=30, subsample=0.8, colsample_bytree=1.0, reg_alpha=1.0, reg_lambda=0.1 | 0.6235 |
| SVM | C=5.0, gamma=0.0001, kernel=linear | 0.5460 |


---


## Results — Test Set


### prior_cesd only (1 features)


| Model | AUC | BalAcc | Sens-W | F1-macro | PPV-W |
|---|---|---|---|---|---|
| ElasticNet | 0.695 | 0.551 | 0.220 | 0.476 | 0.158 |
| XGBoost | 0.646 | 0.480 | 0.268 | 0.408 | 0.081 |
| LightGBM | 0.637 | 0.444 | 0.122 | 0.345 | 0.061 |
| SVM | 0.695 | 0.522 | 0.000 | 0.436 | 0.000 |

### base (21) (21 features)


| Model | AUC | BalAcc | Sens-W | F1-macro | PPV-W |
|---|---|---|---|---|---|
| ElasticNet | 0.710 | 0.561 | 0.488 | 0.434 | 0.140 |
| XGBoost | 0.693 | 0.535 | 0.244 | 0.435 | 0.122 |
| LightGBM | 0.664 | 0.456 | 0.195 | 0.383 | 0.078 |
| SVM | 0.716 | 0.545 | 0.488 | 0.432 | 0.148 |

### base + behavioral lag (38) (38 features)


| Model | AUC | BalAcc | Sens-W | F1-macro | PPV-W |
|---|---|---|---|---|---|
| ElasticNet | 0.708 | 0.528 | 0.415 | 0.416 | 0.120 |
| XGBoost | 0.690 | 0.493 | 0.244 | 0.412 | 0.115 |
| LightGBM | 0.684 | 0.530 | 0.268 | 0.425 | 0.113 |
| SVM | 0.681 | 0.513 | 0.463 | 0.391 | 0.131 |

### base + behavioral lag + pmcesd (39) (39 features)


| Model | AUC | BalAcc | Sens-W | F1-macro | PPV-W |
|---|---|---|---|---|---|
| ElasticNet | 0.759 | 0.624 | 0.585 | 0.480 | 0.175 |
| XGBoost | 0.750 | 0.620 | 0.634 | 0.457 | 0.166 |
| LightGBM | 0.732 | 0.586 | 0.463 | 0.455 | 0.145 |
| SVM | 0.751 | 0.620 | 0.610 | 0.475 | 0.191 |

---

