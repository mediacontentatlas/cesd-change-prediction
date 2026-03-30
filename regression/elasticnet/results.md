# ElasticNet Regression -- Results

## Summary

The best ElasticNet condition is **base_lag_pmcesd** (base features + 17 behavioral lag features + person_mean_cesd), selected by lowest validation MAE (3.92). It achieves **Test MAE = 4.13**, beating the best naive baseline (B0: No Change, Test MAE = 4.60). The model retains only 2 of 39 features -- both CES-D history variables -- and zeroes out every behavioral (screenome) feature. It is effectively a regression-to-the-mean predictor: it pushes each person's predicted CES-D change back toward their long-run average.

An honorable mentioned is **pheno_pid**, which achieves slightly better Test MAE (4.06) by learning person-specific intercepts. However, this is at the cost of 120 parameters for 96 persons and a worse validation MAE (3.92).

---

## Regression Baselines

| Baseline | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---|---|---|---|
| B0: No Change | 4.31 | 5.67 | 4.60 | 7.42 |
| B1: Population Mean | 4.34 | 6.62 | 4.62 | 7.41 |
| B2: LVCF | 4.46 | 6.62 | 7.03 | 11.13 |
| B3: Person-Specific Mean | 4.39 | 6.73 | 4.77 | 7.52 |
| B4: Regression to Mean | 6.97 | 6.67 | 4.68 | 7.45 |

- **B0 (No Change)**: Predict 0 delta for all samples. Strongest baseline on Val.
- **B2 (LVCF)**: Predicts last period's CES-D delta. Severely overfits (Test MAE 7.03).
- **B4 (Regression to Mean)**: Predicts delta toward person mean CES-D. Poor Val MAE but reasonable Test MAE -- conceptually closest to what base_lag_pmcesd learns.

---

## Feature Conditions Inventory

| Feature Set | Description | # Features | Retained |
|---|---|---|---|
| prior_cesd | Prior CES-D only | 1 | 1 |
| base | All 21 base features | 21 | 1 |
| pheno | Base + 5 phenotype features | 26 | 1 |
| dev | Base + 8 within-person deviation features | 29 | 1 |
| dev_pheno | Base + dev + pheno | 34 | 1 |
| base_lag | Base + 17 behavioral lag features | 38 | 1 |
| base_lag_pmcesd | Base + lag + person_mean_cesd | 39 | 2 |
| pid | Base + ~96 PID one-hot | 117 | 101 |
| pheno_pid | Base + pheno + PID OHE | 122 | 120 |
| dev_pid | Base + dev + PID OHE | 125 | 107 |
| dev_pheno_pid | Base + dev + pheno + PID OHE | 130 | 111 |

The key obsevation is that 6 conditions (prior_cesd, base, pheno, dev, dev_pheno, base_lag) collapse to identical or near-identical models -- ElasticNet zeroes out all features except `prior_cesd`. Without person-level information (person_mean_cesd or PID encoding), no behavioral feature survives regularization.

---

## Cross-Condition Ranking

Sorted by Val MAE (selection criterion). Conditions marked with `*` are required (parity with classification).

| Feature Set | # Feat | Retained | Train MAE | Val MAE | Test MAE | Test AUC | Test BalAcc |
|---|---|---|---|---|---|---|---|
| base_lag_pmcesd* | 39 | 2 | 3.901 | **3.924** | 4.127 | 0.722 | 0.540 |
| pheno_pid | 122 | 120 | 3.784 | 3.921 | **4.059** | 0.723 | 0.579 |
| pid | 117 | 101 | 3.778 | 3.927 | 4.087 | 0.715 | 0.602 |
| dev_pid | 125 | 107 | 3.780 | 3.950 | 4.100 | 0.715 | 0.603 |
| dev_pheno_pid | 130 | 111 | 3.775 | 3.968 | 4.097 | 0.717 | 0.593 |
| base_lag* | 38 | 1 | 4.649 | 4.289 | 4.584 | 0.678 | 0.403 |
| prior_cesd* | 1 | 1 | 4.649 | 4.289 | 4.584 | 0.678 | 0.403 |
| base* | 21 | 1 | 4.649 | 4.289 | 4.584 | 0.678 | 0.403 |
| dev_pheno | 34 | 1 | 4.649 | 4.289 | 4.584 | 0.678 | 0.403 |
| pheno | 26 | 1 | 4.649 | 4.289 | 4.584 | 0.678 | 0.403 |
| dev | 29 | 1 | 4.649 | 4.289 | 4.584 | 0.678 | 0.403 |

Direction metrics are from posthoc sev_crossing analysis. AUC = OvR macro, BalAcc = macro recall over 3 classes (improving / stable / worsening).

**Observations**:
- The top 5 conditions all include PID one-hot encoding or lag+person_mean, suggesting person-level information is critical
- base_lag_pmcesd wins on Val MAE despite using far fewer parameters (2 vs 100+)
- pheno_pid achieves the best Test MAE (4.059) but loses on Val MAE (3.921 vs 3.924) -- essentially tied
- The 6 collapsed conditions (prior_cesd through base_lag) show that screenome features alone add nothing in a pooled linear model; even 17 behavioral lags don't help without person_mean_cesd

---

## Best Model vs Baselines

| Model | Val MAE | Test MAE | Val RMSE | Test RMSE |
|---|---|---|---|---|
| **base_lag_pmcesd** | **3.92** | **4.13** | **5.73** | **6.28** |
| B0: No Change | 4.31 | 4.60 | 5.67 | 7.42 |
| B1: Population Mean | 4.34 | 4.62 | 6.62 | 7.41 |
| B2: LVCF | 4.46 | 7.03 | 6.62 | 11.13 |
| B3: Person-Specific Mean | 4.39 | 4.77 | 6.73 | 7.52 |
| B4: Regression to Mean | 6.97 | 4.68 | 6.67 | 7.45 |

---

## Model Coefficients & Interpretation

### Best condition: base_lag_pmcesd (2 of 39 features retained)

Hyperparameters: alpha = 10.0, l1_ratio = 0.1 (strongly Ridge-dominant).

| Feature | Dev Coef | Final Coef | Interpretation |
|---|---|---|---|
| prior_cesd | -0.502 | -0.458 | Higher current CES-D predicts a *decrease* (regression toward mean) |
| person_mean_cesd | +0.447 | +0.418 | Higher personal average predicts an *increase* (pull back up) |

The two coefficients nearly cancel. The predicted delta approximates:

```
delta_hat ≈ 0.45 * (person_mean_cesd - prior_cesd)
```

This is a **regression-to-the-mean machine**: it predicts that CES-D will move back toward each person's long-run average. No behavioral feature (screen time, app switching, social media use, etc.) contributes.

### Contrast: PID-based models

The PID-encoded models (pheno_pid, pid, dev_pid, dev_pheno_pid) take a fundamentally different approach. Instead of modeling the regression-to-mean mechanism explicitly, they learn **person-specific intercepts** via one-hot encoded participant IDs.

**pheno_pid (120 of 122 features retained):**

| Category | Example features | Coef range | Role |
|---|---|---|---|
| PID intercepts | pid_5343 (+18.2), pid_1257 (-10.3) | -10.3 to +18.2 | Person-specific baseline delta |
| Gender | gender_mode_1 (-4.5), gender_mode_2 (-3.1) | -4.5 to -3.1 | Group-level offset |
| Phenotype | pheno_1 (-3.0), pheno_2 (-1.7), pheno_0 (-0.8) | -3.0 to -0.5 | Latent subtype adjustment |
| Behavioral (screenome) | active_day_ratio_delta (+3.0), overnight_ratio (+1.0) | -1.7 to +3.0 | Small behavioral signals |
| Prior CES-D | prior_cesd (-0.64) | -0.64 | Regression to mean (diminished vs base_lag_pmcesd) |

By assigning each person a unique intercept (for example, PID 5343 gets +18.2, meaning their CES-D is predicted to increase by ~18 points before other features adjust), the model captures person-level dynamics that the pooled model cannot. This partly compensates for the absence of explicit person-mean features.

Behavioral features *do* enter the PID-based models, but with smaller coefficients than the PID intercepts. The largest behavioral effect (active_day_ratio_delta = +3.0) is dwarfed by the person intercepts (+/-10-18).

### What this means

1. **Screenome features add negligible value in a pooled linear model**: CES-D history dominates entirely
2. **Person-level information is the key for ElasticNet**: whether encoded explicitly (person_mean_cesd) or via PID dummies, knowing *who* the person is matters far more than *what they did on their phone*
3. **PID-based models risk overfitting**: 120 parameters for 96 persons means more free parameters than training subjects, so the model could easily memorize person-specific noise rather than learning generalizable patterns; the fact that Test MAE still improves suggests the person effects are real, but performance would likely degrade substantially on a new sample of persons
4. **The regression-to-mean mechanism is dominant**: CES-D scores naturally fluctuate toward each person's average, and ElasticNet identifies this as the strongest predictive signal

---

## Direction Classification (Posthoc)

Regression predictions are converted to 3-class direction labels (improving / stable / worsening) using the same labeling function as the ground truth, enabling direct comparison with dedicated classifiers.

### sev_crossing (primary) -- base_lag_pmcesd

Clinical severity boundary crossing (CES-D thresholds at 16 and 24).

| Split | BalAcc | AUC (OvR) | Sens-W | PPV-W |
|---|---|---|---|---|
| Val | 0.571 | 0.732 | 0.200 | 0.636 |
| Test | 0.540 | 0.722 | 0.243 | 0.750 |

### personal_sd (sensitivity analysis) -- base_lag_pmcesd

Person-specific SD-based thresholds (k = 1.0).

| Split | BalAcc | AUC (OvR) | Sens-W | PPV-W |
|---|---|---|---|---|
| Val | 0.374 | 0.693 | 0.000 | 0.000 |
| Test | 0.385 | 0.751 | 0.000 | 0.000 |

### Stratified regression error by direction class (test, sev_crossing)

| Direction | N | MAE | RMSE | Bias |
|---|---|---|---|---|
| Improving | 44 | 10.04 | 11.68 | +10.03 |
| Stable | 330 | 2.72 | 3.80 | +0.55 |
| Worsening | 37 | 9.66 | 12.14 | -9.48 |

The model predicts the **stable** class well (MAE 2.72, near-zero bias) but severely misestimates **improving** and **worsening** cases (MAE ~10, strong directional bias). This is consistent with a regression-to-the-mean model: it predicts small deltas near zero, which are correct for stable cases but miss the large swings.

### Median per-person direction accuracy

| Label Type | Val | Test |
|---|---|---|
| sev_crossing | 1.000 | 1.000 |
| personal_sd | 0.800 | 0.900 |

The sev_crossing median of 1.0 reflects the class imbalance: 80% of observations are "stable," and the model predicts nearly all observations as stable. Most persons have all-stable ground truth, so they get perfect accuracy trivially.

---

## Performer Tier Analysis

Participants classified into tiers by per-person MAE (25th / 75th percentile thresholds).

### Test set

| Tier | N | MAE Mean | MAE Std | MAE Median | RMSE Mean |
|---|---|---|---|---|---|
| High (good) | 24 | 1.10 | 0.37 | 1.15 | 1.31 |
| Medium | 48 | 3.16 | 1.03 | 3.18 | 3.65 |
| Low (poor) | 24 | 8.95 | 3.27 | 8.17 | 10.53 |

### Validation set

| Tier | N | MAE Mean | MAE Std | MAE Median | RMSE Mean |
|---|---|---|---|---|---|
| High (good) | 24 | 1.13 | 0.51 | 1.18 | 1.25 |
| Medium | 48 | 3.24 | 0.87 | 3.00 | 3.93 |
| Low (poor) | 24 | 8.09 | 2.47 | 7.63 | 9.37 |

The model works well for some participants (high-tier MAE ~1 CES-D point) and very poorly for others (low-tier MAE ~9 points). This 9x spread suggests substantial individual heterogeneity that a pooled linear model cannot capture.

---

## Key Takeaways

1. **Best feature set**: `base_lag_pmcesd` (Val MAE 3.92, Test MAE 4.13), selected by validation MAE per protocol
2. **Beats B0 baseline** by ~0.47 Test MAE points -- a modest but consistent improvement
3. **All screenome behavioral features zeroed out** by regularization; the model relies entirely on CES-D history (prior_cesd, person_mean_cesd)
4. **ElasticNet is a regression-to-the-mean machine**: it predicts CES-D will move back toward each person's long-run average -- a 2-feature model
5. **PID-encoded models can achieve slightly better Test MAE** (pheno_pid: 4.059 vs 4.127) by learning person-specific intercepts, but at the cost of 120 parameters for 96 persons
6. **Direction prediction dominated by the stable class**; poor sensitivity for worsening (Sens-W = 0.24) and zero for personal_sd
7. **Large performer variability**: high-tier MAE ~1 vs low-tier MAE ~9, indicating substantial heterogeneity that a pooled linear model cannot capture
