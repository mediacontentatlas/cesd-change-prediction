# MixedLM Regression: Post-Hoc Direction Analysis

## Classification Metrics (Regression Predictions as Direction Classifier)

### Label type: `sev_crossing`

| Model | BalAcc | AUC (OvR) | Sens-W | PPV-W | F1 macro |
|-------|--------|-----------|--------|-------|----------|
| 1_pooled | 0.458 | 0.582 | 0.486 | 0.070 | 0.172 |
| 2_intercept | 0.515 | 0.704 | 0.703 | 0.114 | 0.174 |
| 3_prior_slope | 0.506 | 0.700 | 0.676 | 0.110 | 0.172 |
| 4_prior_switches | 0.513 | 0.704 | 0.676 | 0.108 | 0.176 |
| 5_prior_social | 0.506 | 0.703 | 0.676 | 0.111 | 0.171 |
| 6_prior_soc_ext | 0.506 | 0.701 | 0.676 | 0.111 | 0.171 |
| 7_sw_screens | 0.488 | 0.694 | 0.622 | 0.100 | 0.167 |
| 8_prior_sw_scr | 0.506 | 0.702 | 0.676 | 0.109 | 0.172 |
| 9_dev_features | 0.498 | 0.694 | 0.676 | 0.110 | 0.169 |
| base | 0.515 | 0.704 | 0.703 | 0.114 | 0.174 |
| base_dev | 0.498 | 0.694 | 0.676 | 0.110 | 0.169 |
| base_dev_pmcesd | 0.542 | 0.756 | 0.784 | 0.145 | 0.178 |
| prior_cesd | 0.516 | 0.704 | 0.730 | 0.108 | 0.180 |
