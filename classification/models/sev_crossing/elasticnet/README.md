# Mood Direction Classifier

ElasticNet logistic regression for within-person, week-to-week mood deterioration prediction.

---

## Task

Predict whether a person's depressive symptoms will **improve**, remain **stable**, or **worsen** in the upcoming observation period, using their current and previous week's smartphone behavioral data.

This is a **within-person longitudinal classification** task — not between-person depression detection. The target variable is the *direction of change* in CESD score, not the current level.

---

## Label: Severity Crossing

The default label (`sev_crossing`) defines the three classes by whether the predicted CESD score crosses a clinical severity boundary:

| Class | Value | Definition |
|---|---|---|
| improving | 0 | Severity category decreased (e.g., moderate → minimal) |
| stable | 1 | Severity category unchanged |
| worsening | 2 | Severity category increased (e.g., minimal → moderate) |

**Severity boundaries** (CESD-20, range 0–60):

| Category | CESD score range |
|---|---|
| Minimal | < 16 |
| Moderate | 16 – 23 |
| Severe | ≥ 24 |

**Label distribution (validation set):** improving = 9%, stable = 82%, worsening = 9%

Two alternative label types are also supported:

| `--label-type` | Definition |
|---|---|
| `sev_crossing` | Severity boundary crossing (default) |
| `thresh_5` | ±5 point CESD change (MCID) |
| `thresh_10` | ±10 point CESD change (conservative) |

---

## Model

**ElasticNet Logistic Regression** (`sklearn.linear_model.LogisticRegression`)

| Setting | Value |
|---|---|
| Penalty | ElasticNet (L1 + L2) |
| Solver | SAGA |
| Class weighting | `balanced` (upweights minority classes) |
| Max iterations | 2000 |
| Best C | 0.1 |
| Best l1_ratio | 0.9 (mostly L1 — sparse solution) |

**Why this model:**
- Regression models (MAE-optimised) predict near-zero for everyone due to regularisation toward the mean, making directional prediction impossible. A classifier trained on class boundaries directly detects direction.
- `class_weight='balanced'` corrects for the 82% stable majority class.
- ElasticNet penalty selects a sparse feature subset while retaining correlated features via the L2 component.

---

## Input Features

**Total: 43 features** (21 base + 22 lag)

### Base features (21)

Computed for the current observation period:

| Feature | Description |
|---|---|
| `prior_cesd` | CESD score at the start of this period |
| `active_day_ratio_delta` | Change in proportion of active days |
| `mean_daily_overnight_ratio` | Proportion of overnight screen use |
| `mean_daily_overnight_ratio_delta` | Change in overnight screen proportion |
| `mean_daily_social_ratio` | Proportion of social app screen time |
| `mean_daily_social_ratio_delta` | Change in social screen proportion |
| `age` | Participant age |
| `mean_daily_screens` | Mean daily screen sessions |
| `mean_daily_screens_delta` | Change in daily screen sessions |
| `mean_daily_unique_apps` | Mean unique apps used daily |
| `mean_daily_unique_apps_delta` | Change in unique apps |
| `mean_daily_switches` | Mean daily app switches |
| `mean_daily_switches_delta` | Change in app switches |
| `switches_per_screen` | App switches per screen session (fragmentation) |
| `switches_per_screen_delta` | Change in fragmentation |
| `mean_daily_social_screens` | Mean daily social app sessions |
| `mean_daily_social_screens_delta` | Change in social app sessions |
| `clip_dispersion` | Dispersion of screen session lengths |
| `clip_dispersion_delta` | Change in session length dispersion |
| `gender_mode_1` | Gender indicator 1 |
| `gender_mode_2` | Gender indicator 2 |

> Note: `_delta` features are changes relative to the *person's own previous period* — they already encode within-person deviation from baseline.

### Lag-1 features (22)

Previous observation period's values for all 21 base features, plus:

| Feature | Description |
|---|---|
| `lag_cesd_delta` | CESD change in the *previous* period |
| `lag_<feature>` | Previous period's value for each base feature |

**Why lag features matter:** Including the previous period's behavioral data and mood change captures temporal momentum. In ablation:

| Features | Val BalAcc | Sens-worsening |
|---|---|---|
| Base only | 0.662 | 0.457 |
| Base + lag | **0.706** | **0.571** |
| Base + lag + PID | 0.706 | 0.571 |
| Base + lag + dev | 0.706 | 0.571 |

Lag features make PID one-hot encoding redundant — they capture the same person-specific temporal context more efficiently.

### Optional features (not used in default run)

| Flag | Features added | Effect |
|---|---|---|
| `--use-pid` | 96 PID OHE columns | No improvement when lag included |
| `--use-dev` | 8 within-person deviation features | No improvement (dev signal too weak) |

---

## Performance (Validation Set, `sev_crossing`)

### Overall

| Metric | Value |
|---|---|
| Balanced accuracy | **0.706** |
| F1-macro | 0.568 |
| F1-worsening | 0.408 |
| Sensitivity (worsening) | **0.571** |
| Accuracy | 0.711 |

### Per-class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| improving | 0.337 | 0.833 | 0.480 | 36 |
| stable | 0.951 | 0.713 | 0.815 | 324 |
| worsening | 0.317 | 0.571 | 0.408 | 35 |

### Confusion matrix

|  | Pred: improving | Pred: stable | Pred: worsening |
|---|---|---|---|
| **True: improving** | 30 | 3 | 3 |
| **True: stable** | 53 | 231 | 40 |
| **True: worsening** | 6 | 9 | **20** |

### Comparison: regression → classification improvement

| Model | BalAcc | Sens-worsening |
|---|---|---|
| Regression + PID (threshold) | 0.621 | 0.229 |
| Classifier (base only) | 0.662 | 0.457 |
| **Classifier (base + lag)** | **0.706** | **0.571** |

---

## Benchmarking note

The 70.6% balanced accuracy should be compared against **within-person longitudinal mood prediction** studies (typical range: 55–68%), not between-person depression detection studies (85–95%). Between-person studies predict whether a person is depressed from their aggregate behavioral profile — a stable trait classification task with much larger effect sizes. Our task predicts week-to-week *change* within a person, where CES-D delta has near-zero between-person ICC (person means explain only 0.4% of delta variance).

---

## Post-hoc: who does the model work for?

### Worsening catch rate by CESD severity

| CESD severity | n with worsening | Catch rate |
|---|---|---|
| Minimal (<16) | 15 | 47% |
| Moderate (16–23) | 6 | 83% |
| Severe (≥24) | 5 | 60% |

The model performs best for the **clinically critical moderate → severe transition**. It misses most worsening in the minimal symptom group, where the CESD change is small (2–5 points to cross the threshold) and behavioral signatures are weaker.

Reactivity cluster (screen-stressor vs soother) does **not** predict catch rate — both groups at ~57–58%.

### Behavioral signature of worsening

Top worsening predictors (current period):

| Feature | Coefficient | Interpretation |
|---|---|---|
| `mean_daily_social_screens_delta` | +0.19 | Increasing social media use |
| `switches_per_screen` | +0.09 | High behavioral fragmentation |
| `mean_daily_screens_delta` | −0.07 | Withdrawing from total screen use |
| `mean_daily_unique_apps_delta` | +0.05 | Using more varied apps |

Top lag predictors (previous period):

| Feature | Coefficient | Interpretation |
|---|---|---|
| `lag_clip_dispersion` | +0.09 | Previously fragmented sessions → risk |
| `lag_cesd_delta` | −0.06 | Worsened last week → less likely again (mean reversion) |
| `lag_mean_daily_switches_delta` | +0.05 | Accelerating switch behaviour → risk |

**Pattern:** worsening is preceded by *withdrawal from productive screen use while escalating social media browsing* — a recognised digital behavioural signature of depression onset.

---

## Training data

| Split | N obs | N persons |
|---|---|---|
| Train | 1196 | 96 |
| Val | 395 | 96 |
| Test | 411 | 96 |

Observation period: ~weekly. Persons contribute 8–20 observations each (median 13 in train, 4 in val).

---

## Output files

| File | Description |
|---|---|
| `model.joblib` | Fitted LogisticRegression |
| `best_params.yaml` | Best hyperparameters and val metrics |
| `grid_search_results.csv` | All C × l1_ratio combinations with val BalAcc |
| `y_pred_train.npy` | Predicted class labels (0/1/2) — train |
| `y_pred_val.npy` | Predicted class labels (0/1/2) — val |
| `y_true_train.npy` | True labels — train |
| `y_true_val.npy` | True labels — val |
| `y_proba_val.npy` | Predicted class probabilities (n×3) — val |
| `feature_coefficients.csv` | Per-class coefficients for all features |

---

## Usage

```bash
# Default: severity crossing + lag features
python3 scripts/train_classifier.py

# 5-point threshold
python3 scripts/train_classifier.py \
    --label-type thresh_5 \
    --output-dir models/sev_crossing/elasticnet_thresh5

# Ablation: no lag features
python3 scripts/train_classifier.py \
    --no-lag \
    --output-dir models/sev_crossing/elasticnet_nolag

# With PID intercepts (no improvement over lag-only)
python3 scripts/train_classifier.py \
    --use-pid \
    --output-dir models/sev_crossing/elasticnet_pid

# Custom data directory and config
python3 scripts/train_classifier.py \
    --data-dir data/processed \
    --config configs/classifier.yaml \
    --output-dir models/sev_crossing/elasticnet_custom
```
