# Frozen-CES-D Deployment Scenario — sev_crossing


## Setup

Same trained model as the full 39-feature model (best params from bootstrap analysis).
At evaluation time, for **every val and test observation**:

- `prior_cesd` → person's **first-ever CES-D score** (from their full record)
- `person_mean_cesd` → same first-ever CES-D score
- All 37 behavioral + lag features: **unchanged**, computed from Screenome as usual

This simulates deployment where the CES-D was only administered once (intake)
and never updated, while passive sensing continues.

## Results


| Model | AUC [95% CI] | BalAcc [95% CI] | F1-macro [95% CI] | Sens-W [95% CI] | PPV-W [95% CI] |
|---|---|---|---|---|---|
| ElasticNet | 0.720 [0.676, 0.764] | 0.455 [0.398, 0.516] | 0.408 | 0.054 [0.000, 0.143] | 0.200 [0.000, 0.500] |
| XGBoost | 0.670 [0.613, 0.725] | 0.458 [0.385, 0.530] | 0.420 | 0.297 [0.151, 0.447] | 0.159 [0.074, 0.246] |
| LightGBM | 0.629 [0.566, 0.686] | 0.448 [0.380, 0.518] | 0.412 | 0.297 [0.151, 0.447] | 0.155 [0.070, 0.240] |
| SVM | 0.733 [0.685, 0.780] | 0.505 [0.437, 0.572] | 0.444 | 0.243 [0.111, 0.385] | 0.196 [0.089, 0.310] |

## Comparison to Reference Scenarios (XGBoost)


| Scenario | AUC | BalAcc | Sens-W | PPV-W |
|---|---|---|---|---|
| B4 Regression to Mean (baseline) | 0.750 | 0.674 | 0.541 | 0.408 |
| Cold start (leave-group-out CV) | 0.821 | 0.720 | 0.569 | — |
| Full model (39-feat) | 0.906 | 0.834 | 0.838 | 0.356 |
| **Frozen CES-D (this scenario)** | **0.670** | **0.458** | **0.297** | **0.159** |

## Confusion Matrices


**ElasticNet**

```
                pred: imp   pred: stb   pred: wrs
  true: imp ( 44)        23          20           1
  true: stb (330)        63         260           7
  true: wrs ( 37)        21          14           2
```

**XGBoost**

```
                pred: imp   pred: stb   pred: wrs
  true: imp ( 44)        16          14          14
  true: stb (330)        51         235          44
  true: wrs ( 37)        13          13          11
```

**LightGBM**

```
                pred: imp   pred: stb   pred: wrs
  true: imp ( 44)        15          13          16
  true: stb (330)        53         233          44
  true: wrs ( 37)        12          14          11
```

**SVM**

```
                pred: imp   pred: stb   pred: wrs
  true: imp ( 44)        25          12           7
  true: stb (330)        68         232          30
  true: wrs ( 37)        19           9           9
```

## Interpretation


- **AUC > 0.821 (cold start)**: behavioral trajectory contributes beyond person identity
- **AUC ≈ 0.906 (full model)**: frozen CES-D barely matters; Screenome carries prediction
- **AUC closer to 0.750 (B4 baseline)**: CES-D updates are essential; stale anchor hurts

Frozen CES-D mean (val): 13.55  vs  live prior_cesd mean: 11.92
Frozen CES-D mean (test): 13.48  vs  live prior_cesd mean: 12.05