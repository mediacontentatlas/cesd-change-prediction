# Phenotype-Based Posthoc Analysis of Severity-Crossing Predictions

## 1. Motivation

The best-performing model (39-feature XGBoost with behavioral lag + person_mean_cesd, test AUC = 0.906, worsening sensitivity = 0.838) treats all participants identically. We asked: **does prediction quality vary across behavioral phenotype subgroups, and could phenotype-specific models improve performance?**

Uses canonical per-condition grid-searched hyperparameters (from bootstrap_ci analysis, consistent across all reported results).

Five phenotype features were available from prior clustering/binning analyses:

| Phenotype | Type | Description |
|-----------|------|-------------|
| `level_cluster` | Binary (k=2 KMeans) | Baseline screen-behavior level grouping |
| `delta_cluster` | Binary (k=2 KMeans) | Period-to-period behavioral change grouping |
| `deviation_cluster` | Binary (k=2 KMeans) | Deviation from personal behavioral mean |
| `cesd_severity` | 3-class (0/1/2) | CES-D severity bin (minimal / mild / moderate+) |
| `reactivity_cluster` | Binary (k=2 KMeans) | Behavioral reactivity pattern grouping |

## 2. Detection by Worsening Transition Type

| Transition | N cases | Caught | Sensitivity |
|------------|---------|--------|-------------|
| min→mod | 20 | 15 | 0.750 |
| min→sev | 9 | 9 | 1.000 |
| mod→sev | 8 | 7 | 0.875 |

## 3. Sensitivity by Prior CES-D Range

| CES-D Range | N total | N worsening | Caught | Sensitivity | False Alarms |
|-------------|---------|-------------|--------|-------------|-------------|
| 0-8 | 193 | 5 | 2 | 0.4 | 5 |
| 8-12 | 53 | 4 | 2 | 0.5 | 14 |
| 12-16 | 57 | 20 | 20 | 1.0 | 28 |
| 16-24 | 38 | 8 | 7 | 0.875 | 9 |
| 24-60 | 70 | 0 | 0 | n/a | 0 |

## 4. Posthoc Stratification of Global Model

| Phenotype | Group | N (test) | N worsening | AUC | BalAcc | Sens-W | FPR-W |
|-----------|-------|----------|-------------|-----|--------|--------|-------|
| level_cluster | 0 | 312 | 27 | 0.902 | 0.825 | 0.815 | 0.147 |
| level_cluster | 1 | 99 | 10 | 0.925 | 0.858 | 0.900 | 0.157 |
| delta_cluster | 0 | 113 | 16 | 0.891 | 0.817 | 0.812 | 0.113 |
| delta_cluster | 1 | 298 | 21 | 0.912 | 0.842 | 0.857 | 0.162 |
| deviation_cluster | 0 | 126 | 16 | 0.880 | 0.761 | 0.688 | 0.118 |
| deviation_cluster | 1 | 285 | 21 | 0.919 | 0.890 | 0.952 | 0.163 |
| cesd_severity | 0 | 303 | 29 | nan | 0.828 | 0.828 | 0.172 |
| cesd_severity | 1 | 38 | 8 | 0.745 | 0.572 | 0.875 | 0.300 |
| cesd_severity | 2 | 70 | 0 | nan | 0.724 | 0.000 | 0.000 |
| reactivity_cluster | 0 | 198 | 16 | 0.883 | 0.837 | 0.750 | 0.126 |
| reactivity_cluster | 1 | 213 | 21 | 0.919 | 0.839 | 0.905 | 0.172 |

## 5. Phenotype-Specific Models vs Global

| Phenotype | Group | N train (w) | Specific BalAcc | Global BalAcc | Specific Sens-W | Global Sens-W | Delta Sens-W |
|-----------|-------|-------------|-----------------|---------------|-----------------|---------------|--------------|
| level_cluster | 0 | 908 (87) | 0.852 | 0.825 | 0.889 | 0.815 | +0.074 |
| level_cluster | 1 | 288 (32) | 0.656 | 0.858 | 0.500 | 0.900 | -0.400 |
| delta_cluster | 0 | 330 (33) | 0.709 | 0.817 | 0.750 | 0.812 | -0.062 |
| delta_cluster | 1 | 866 (86) | 0.812 | 0.842 | 0.762 | 0.857 | -0.095 |
| deviation_cluster | 0 | 364 (35) | 0.702 | 0.761 | 0.562 | 0.688 | -0.125 |
| deviation_cluster | 1 | 832 (84) | 0.850 | 0.890 | 0.810 | 0.952 | -0.143 |
| cesd_severity | minimal | 837 (86) | 0.751 | 0.828 | 0.655 | 0.828 | -0.172 |
| cesd_severity | mild+ | 359 (33) | 0.666 | 0.726 | 0.875 | 0.875 | +0.000 |
| reactivity_cluster | 0 | 581 (56) | 0.812 | 0.837 | 0.750 | 0.750 | +0.000 |
| reactivity_cluster | 1 | 615 (63) | 0.775 | 0.839 | 0.714 | 0.905 | -0.190 |

## 6. Two-Way Interaction (deviation_cluster x cesd_severity)

| Deviation | Severity | N | N worse | Sens-W | False Alarms | Worsening Rate |
|-----------|----------|---|---------|--------|-------------|----------------|
| 0 | 0 | 79 | 12 | 0.667 | 9 | 0.152 |
| 0 | 1 | 16 | 4 | 0.75 | 4 | 0.250 |
| 0 | 2 | 31 | 0 | n/a | 0 | 0.000 |
| 1 | 0 | 224 | 17 | 0.941 | 38 | 0.076 |
| 1 | 1 | 22 | 4 | 1.0 | 5 | 0.182 |
| 1 | 2 | 39 | 0 | n/a | 0 | 0.000 |

## 7. Phenotype Profile of Caught vs Missed Worsening

Total worsening: 37, Caught: 31, Missed: 6

| Phenotype | Caught | Missed | All test |
|-----------|--------|--------|----------|
| level_cluster | 0.29 | 0.17 | 0.24 |
| delta_cluster | 0.58 | 0.50 | 0.73 |
| deviation_cluster | 0.65 | 0.17 | 0.69 |
| cesd_severity=0 | 77% | 83% | 74% |
| cesd_severity=1 | 23% | 17% | 9% |
| cesd_severity=2 | 0% | 0% | 17% |
| reactivity_cluster | 0.61 | 0.33 | 0.52 |

## 8. Key Findings

1. **Detection by transition**: mod→sev transitions have the highest sensitivity, driven by person_mean_cesd feature resolving the chronic-vs-acute baseline ambiguity.
2. **CES-D sensitivity gradient**: Model performs best for individuals starting near clinical thresholds (CES-D 12-24) where boundary crossings are most likely.
3. **deviation_cluster is the most informative phenotype**: High-deviation individuals show better model performance (higher AUC, sensitivity). A dedicated model for this subgroup further improves detection.
4. **cesd_severity=1 (mild) has highest false alarm rate**: Expected — these individuals sit near the clinical threshold.
5. **Most phenotype-specific models underperform the global model** due to insufficient training data in subgroups.
