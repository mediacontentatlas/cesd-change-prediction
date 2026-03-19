# Baseline Results — All Label Types


Classes: **0 = improving**, **1 = stable**, **2 = worsening**

## Baseline Definitions

| ID | Name | Rule |
|---|---|---|
| B0 | No Change | Predict all stable (class 1) — lower bound |
| B1 | Population Mean | Predict majority class from training set |
| B2 | LVCF | Repeat previous period's class label (last value carried forward) |
| B3 | Person-Specific Mean | Predict each person's modal class in training |
| B4 | Regression to Mean | Predict direction CES-D moves toward person's training mean |

**AUC** = one-vs-rest macro. **SensW** = recall for worsening (class 2). **PPVW** = precision for worsening.

---

## sev_crossing

Train distribution: imp=121  stb=956  wrs=119  (total 1196)

Test distribution: imp=44  stb=330  wrs=37  (total 411)

| Baseline | AUC | BalAcc | F1-macro | Sens-W | PPV-W |
|---|---|---|---|---|---|
| **B0** No Change (predict all stable) | 0.500 | 0.333 | 0.297 | 0.000 | 0.000 |
| **B1** Population Mean (majority class from training) | 0.500 | 0.333 | 0.297 | 0.000 | 0.000 |
| **B2** LVCF (last value carried forward) | 0.557 | 0.336 | 0.336 | 0.054 | 0.053 |
| **B3** Person-Specific Mean (modal training class per person) | 0.499 | 0.327 | 0.304 | 0.000 | 0.000 |
| **B4** Regression to Mean (direction toward person mean CES-D) | 0.750 | 0.674 | 0.654 | 0.541 | 0.408 |

### Confusion Matrices (sev_crossing)

**B0** — No Change (predict all stable)

```
              pred_imp  pred_stb  pred_wrs
  true_imp           0        44         0
  true_stb           0       330         0
  true_wrs           0        37         0
```

**B1** — Population Mean (majority class from training)

```
              pred_imp  pred_stb  pred_wrs
  true_imp           0        44         0
  true_stb           0       330         0
  true_wrs           0        37         0
```

**B2** — LVCF (last value carried forward)

```
              pred_imp  pred_stb  pred_wrs
  true_imp           3        16        25
  true_stb          27       292        11
  true_wrs          14        21         2
```

**B3** — Person-Specific Mean (modal training class per person)

```
              pred_imp  pred_stb  pred_wrs
  true_imp           1        40         3
  true_stb           7       316         7
  true_wrs           1        36         0
```

**B4** — Regression to Mean (direction toward person mean CES-D)

```
              pred_imp  pred_stb  pred_wrs
  true_imp          27        15         2
  true_stb          17       286        27
  true_wrs           0        17        20
```

---

## personal_sd

Train distribution: imp=146  stb=904  wrs=146  (total 1196)

Test distribution: imp=50  stb=320  wrs=41  (total 411)

| Baseline | AUC | BalAcc | F1-macro | Sens-W | PPV-W |
|---|---|---|---|---|---|
| **B0** No Change (predict all stable) | 0.500 | 0.333 | 0.292 | 0.000 | 0.000 |
| **B1** Population Mean (majority class from training) | 0.500 | 0.333 | 0.292 | 0.000 | 0.000 |
| **B2** LVCF (last value carried forward) | 0.529 | 0.322 | 0.320 | 0.049 | 0.045 |
| **B3** Person-Specific Mean (modal training class per person) | 0.500 | 0.333 | 0.292 | 0.000 | 0.000 |
| **B4** Regression to Mean (direction toward person mean CES-D) | 0.642 | 0.533 | 0.526 | 0.293 | 0.240 |

### Confusion Matrices (personal_sd)

**B0** — No Change (predict all stable)

```
              pred_imp  pred_stb  pred_wrs
  true_imp           0        50         0
  true_stb           0       320         0
  true_wrs           0        41         0
```

**B1** — Population Mean (majority class from training)

```
              pred_imp  pred_stb  pred_wrs
  true_imp           0        50         0
  true_stb           0       320         0
  true_wrs           0        41         0
```

**B2** — LVCF (last value carried forward)

```
              pred_imp  pred_stb  pred_wrs
  true_imp           3        22        25
  true_stb          29       274        17
  true_wrs           9        30         2
```

**B3** — Person-Specific Mean (modal training class per person)

```
              pred_imp  pred_stb  pred_wrs
  true_imp           0        50         0
  true_stb           0       320         0
  true_wrs           0        41         0
```

**B4** — Regression to Mean (direction toward person mean CES-D)

```
              pred_imp  pred_stb  pred_wrs
  true_imp          25        24         1
  true_stb          25       258        37
  true_wrs           1        28        12
```

---

## balanced_tercile

Train distribution: imp=398  stb=398  wrs=400  (total 1196)

Test distribution: imp=137  stb=137  wrs=137  (total 411)

| Baseline | AUC | BalAcc | F1-macro | Sens-W | PPV-W |
|---|---|---|---|---|---|
| **B0** No Change (predict all stable) | 0.500 | 0.333 | 0.167 | 0.000 | 0.000 |
| **B1** Population Mean (majority class from training) | 0.500 | 0.333 | 0.167 | 1.000 | 0.333 |
| **B2** LVCF (last value carried forward) | 0.482 | 0.309 | 0.307 | 0.197 | 0.201 |
| **B3** Person-Specific Mean (modal training class per person) | 0.580 | 0.440 | 0.432 | 0.241 | 0.363 |
| **B4** Regression to Mean (direction toward person mean CES-D) | 0.646 | 0.528 | 0.530 | 0.584 | 0.506 |

### Confusion Matrices (balanced_tercile)

**B0** — No Change (predict all stable)

```
              pred_imp  pred_stb  pred_wrs
  true_imp           0       137         0
  true_stb           0       137         0
  true_wrs           0       137         0
```

**B1** — Population Mean (majority class from training)

```
              pred_imp  pred_stb  pred_wrs
  true_imp           0         0       137
  true_stb           0         0       137
  true_wrs           0         0       137
```

**B2** — LVCF (last value carried forward)

```
              pred_imp  pred_stb  pred_wrs
  true_imp          35        30        72
  true_stb          37        65        35
  true_wrs          61        49        27
```

**B3** — Person-Specific Mean (modal training class per person)

```
              pred_imp  pred_stb  pred_wrs
  true_imp          77        24        36
  true_stb          44        71        22
  true_wrs          75        29        33
```

**B4** — Regression to Mean (direction toward person mean CES-D)

```
              pred_imp  pred_stb  pred_wrs
  true_imp          67        35        35
  true_stb          24        70        43
  true_wrs          12        45        80
```
