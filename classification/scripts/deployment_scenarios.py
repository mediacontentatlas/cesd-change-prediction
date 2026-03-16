"""Deployment scenario evaluation — sev_crossing label.

Scenarios (ordered from least to most information):
  1. Population baseline — predict all stable, compute actual metrics
  2. Intake form only — demographics (age, gender_mode_1, gender_mode_2), own grid-searched model
  3. Onboarding (first CES-D) — 39-feat model, prior_cesd & pmcesd = intake CES-D
  4a. Stale 4 weeks — 39-feat model, prior_cesd from 1 period ago
  4b. Stale 8 weeks — 39-feat model, prior_cesd from 2 periods ago
  5. No fresh CES-D — 39-feat model, prior_cesd = pop_mean
  6. Cold start — 39-feat model, repeated leave-group-out CV by PID (5x5 = 25 folds)
  7. Full model — 39 features, normal test set

Design:
  - Only scenario 2 gets its own grid-searched model (different feature set).
  - Scenarios 3–5, 7 use the SAME 39-feat model with degraded test inputs.
  - Scenario 6 uses repeated leave-group-out CV (5 repeats × 5-fold = 25 evaluations):
    hold out ~20% of PIDs per fold, shuffle person assignments each repeat,
    test on held-out PIDs' test-period observations with pmcesd = pop_mean.

Outputs:
  - reports/deployment_results.md
  - models/deployment_scenarios/deployment_results.csv
  - models/deployment_scenarios/confusion_matrices.csv
  - models/deployment_scenarios/cold_start_fold_results.csv
  - models/deployment_scenarios/grid_search_params.yaml
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("models/deployment_scenarios")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/deployment_results.md")

MODEL_NAMES = ["ElasticNet", "XGBoost", "LightGBM", "SVM"]
COLD_START_FOLDS = 5
COLD_START_REPEATS = 5
BOOTSTRAP_N = 1000
SEED = 42

# ======================================================================
# Grid definitions (only for intake form scenario)
# ======================================================================
EN_GRID = [
    {"C": C, "l1_ratio": l1r}
    for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    for l1r in [0.1, 0.5, 0.9, 0.99]
]

XGB_GRID = [
    {"learning_rate": lr, "max_depth": md, "min_child_weight": mcw,
     "n_estimators": ne, "subsample": ss, "colsample_bytree": cb}
    for lr in [0.01, 0.05, 0.1]
    for md in [3, 5]
    for mcw in [1, 3]
    for ne in [100, 200]
    for ss in [0.8, 1.0]
    for cb in [0.8, 1.0]
]

LGBM_GRID = [
    {"learning_rate": lr, "max_depth": md, "n_estimators": ne,
     "num_leaves": nl, "min_child_samples": mcs,
     "subsample": ss, "colsample_bytree": cb,
     "reg_alpha": ra, "reg_lambda": rl}
    for lr in [0.01, 0.1]
    for md in [3, 5, 7]
    for ne in [50, 100]
    for nl in [15, 31]
    for mcs in [10, 30]
    for ss in [0.8, 1.0]
    for cb in [0.8, 1.0]
    for ra in [0.0, 1.0]
    for rl in [0.0, 0.1]
]

SVM_GRID = (
    [{"kernel": "linear", "C": C} for C in [0.1, 0.5, 1.0, 5.0]] +
    [{"kernel": "rbf", "C": C, "gamma": g}
     for C in [0.5, 1.0, 5.0, 10.0]
     for g in [0.0001, 0.001, 0.01, 0.1]]
)

GRIDS = {"ElasticNet": EN_GRID, "XGBoost": XGB_GRID,
         "LightGBM": LGBM_GRID, "SVM": SVM_GRID}


# ======================================================================
# Load data
# ======================================================================
print("[1] Loading data...")
X_train = np.load(DATA_DIR / "X_train.npy")
X_val   = np.load(DATA_DIR / "X_val.npy")
X_test  = np.load(DATA_DIR / "X_test.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_val   = np.load(DATA_DIR / "y_val.npy")
y_test  = np.load(DATA_DIR / "y_test.npy")
pid_train = np.load(DATA_DIR / "pid_train.npy")
pid_val   = np.load(DATA_DIR / "pid_val.npy")
pid_test  = np.load(DATA_DIR / "pid_test.npy")

with open("models/feature_names.pkl", "rb") as f:
    base_feature_names = pickle.load(f)

prior_train = X_train[:, 0]
prior_val   = X_val[:, 0]
prior_test  = X_test[:, 0]

print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Load CSVs for lag construction and staleness
all_df = pd.concat([
    pd.read_csv(DATA_DIR / "train_scaled.csv"),
    pd.read_csv(DATA_DIR / "val_scaled.csv"),
    pd.read_csv(DATA_DIR / "test_scaled.csv"),
]).sort_values(["pid", "period_number"]).reset_index(drop=True)

# ======================================================================
# Build lag features (behavioral only, excluding static + clinical)
# ======================================================================
print("\n[2] Building lag features...")

feat_cols = base_feature_names
lag_cols_all = [f"lag_{c}" for c in feat_cols] + ["lag_cesd_delta"]

for col in [c for c in feat_cols if c in all_df.columns]:
    all_df[f"lag_{col}"] = all_df.groupby("pid")[col].shift(1)
for col in feat_cols:
    if f"lag_{col}" not in all_df.columns:
        all_df[f"lag_{col}"] = 0.0
all_df["lag_cesd_delta"] = all_df.groupby("pid")["target_cesd_delta"].shift(1)
all_df[lag_cols_all] = all_df[lag_cols_all].fillna(0)

drop_lags = ["lag_age", "lag_gender_mode_1", "lag_gender_mode_2",
             "lag_prior_cesd", "lag_cesd_delta"]
keep_idx = [i for i, c in enumerate(lag_cols_all) if c not in drop_lags]
lag_cols = [lag_cols_all[i] for i in keep_idx]

lag_tr = all_df[all_df["split"] == "train"][lag_cols_all].values[:, keep_idx]
lag_va = all_df[all_df["split"] == "val"][lag_cols_all].values[:, keep_idx]
lag_te = all_df[all_df["split"] == "test"][lag_cols_all].values[:, keep_idx]

print(f"  Behavioral lag features: {len(lag_cols)}")

# ======================================================================
# person_mean_cesd
# ======================================================================
pmcesd_path = Path("models/classifier_xgb_best39/person_mean_cesd.json")
if pmcesd_path.exists():
    with open(pmcesd_path) as f:
        pmcesd_raw = json.load(f)
    person_mean_cesd = {}
    for k, v in pmcesd_raw.items():
        try:
            person_mean_cesd[int(k)] = v
        except ValueError:
            person_mean_cesd[k] = v
else:
    person_mean_cesd = {}
    for pid in np.unique(pid_train):
        person_mean_cesd[pid] = float(prior_train[pid_train == pid].mean())

pop_mean = np.mean(list(person_mean_cesd.values()))
pop_mean_cesd_val = float(prior_train.mean())

print(f"  pop_mean_cesd = {pop_mean_cesd_val:.2f}")
print(f"  person_mean_cesd range: [{min(person_mean_cesd.values()):.1f}, "
      f"{max(person_mean_cesd.values()):.1f}]")


def get_pmcesd(pids, override_val=None):
    if override_val is not None:
        return np.full((len(pids), 1), override_val)
    return np.array([
        person_mean_cesd.get(int(p) if hasattr(p, "item") else p, pop_mean)
        for p in pids
    ]).reshape(-1, 1)


# ======================================================================
# Build sev_crossing labels
# ======================================================================
print("\n[3] Building sev_crossing labels...")

CONFIG = yaml.safe_load(open("configs/models/classifier.yaml"))
SEV_MINOR = CONFIG["label"].get("sev_minor", 16)
SEV_MOD = CONFIG["label"].get("sev_moderate", 24)


def severity(cesd):
    return np.where(cesd < SEV_MINOR, 0, np.where(cesd < SEV_MOD, 1, 2))


def make_sev_labels(y_delta, prior):
    sb = severity(prior)
    sa = severity(np.clip(prior + y_delta, 0, 60))
    return np.where(sa < sb, 0, np.where(sa > sb, 2, 1))


y_tr = make_sev_labels(y_train, prior_train)
y_va = make_sev_labels(y_val, prior_val)
y_te = make_sev_labels(y_test, prior_test)

for name, labels in [("Train", y_tr), ("Val", y_va), ("Test", y_te)]:
    dist = " | ".join(f"{['imp','stb','wrs'][i]}={int((labels==i).sum())}"
                      for i in range(3))
    print(f"  {name}: {dist}")

# Class weights for XGBoost
classes_, counts_ = np.unique(y_tr, return_counts=True)
class_wt = {c: len(y_tr) / (len(classes_) * cnt) for c, cnt in zip(classes_, counts_)}

# ======================================================================
# Staleness: get stale prior_cesd values
# ======================================================================
print("\n[4] Computing stale CES-D values for test set...")

full_sorted = all_df.sort_values(["pid", "period_number"]).copy()
full_sorted["stale_1"] = full_sorted.groupby("pid")["prior_cesd"].shift(1)
full_sorted["stale_2"] = full_sorted.groupby("pid")["prior_cesd"].shift(2)
test_stale = full_sorted[full_sorted["split"] == "test"].copy()

stale_1_test = test_stale["stale_1"].fillna(test_stale["prior_cesd"]).values
stale_2_test = test_stale["stale_2"].fillna(test_stale["prior_cesd"]).values

print(f"  Stale 1-period: {(~test_stale['stale_1'].isna()).sum()}/{len(test_stale)} "
      f"have true stale values")
print(f"  Stale 2-period: {(~test_stale['stale_2'].isna()).sum()}/{len(test_stale)} "
      f"have true stale values")

# First CES-D per person (onboarding)
first_cesd = (all_df.sort_values(["pid", "period_number"])
              .groupby("pid")["prior_cesd"].first().to_dict())

# ======================================================================
# Build feature matrices
# ======================================================================
print("\n[5] Building feature matrices...")

# Full 39-feature matrices
X38_tr = np.hstack([X_train, lag_tr])
X38_va = np.hstack([X_val, lag_va])
X38_te = np.hstack([X_test, lag_te])
X39_tr = np.hstack([X38_tr, get_pmcesd(pid_train)])
X39_va = np.hstack([X38_va, get_pmcesd(pid_val)])
X39_te = np.hstack([X38_te, get_pmcesd(pid_test)])

# Feature indices
prior_cesd_idx = 0
demo_names = ["age", "gender_mode_1", "gender_mode_2"]
demo_idx = [base_feature_names.index(n) for n in demo_names]

# Intake form only (3 features)
X_demo_tr = X_train[:, demo_idx]
X_demo_va = X_val[:, demo_idx]
X_demo_te = X_test[:, demo_idx]

# Onboarding: 39-feat model but test prior_cesd = intake, pmcesd = intake
X39_te_onboard = X39_te.copy()
for i, pid in enumerate(pid_test):
    pid_key = int(pid) if hasattr(pid, "item") else pid
    intake = first_cesd.get(pid_key, pop_mean_cesd_val)
    X39_te_onboard[i, prior_cesd_idx] = intake
    X39_te_onboard[i, -1] = intake  # pmcesd = intake

# Stale 4 weeks: prior_cesd from t-1
X39_te_stale4 = X39_te.copy()
X39_te_stale4[:, prior_cesd_idx] = stale_1_test

# Stale 8 weeks: prior_cesd from t-2
X39_te_stale8 = X39_te.copy()
X39_te_stale8[:, prior_cesd_idx] = stale_2_test

# No fresh CES-D: prior_cesd = pop_mean
X39_te_nocesd = X39_te.copy()
X39_te_nocesd[:, prior_cesd_idx] = pop_mean_cesd_val

print(f"  Demographics: {X_demo_tr.shape[1]} features")
print(f"  Full model: {X39_tr.shape[1]} features")

# ======================================================================
# Load 39-feature params from bootstrap analysis
# ======================================================================
print("\n[6] Loading 39-feature params...")
with open("models/bootstrap_ci/sev_crossing_best_params.yaml") as f:
    all_params = yaml.safe_load(f)
params_39 = all_params["base + behavioral lag + pmcesd (39)"]
print(f"  Loaded 39-feature params for all 4 models")

# ======================================================================
# Helper functions
# ======================================================================


def make_model(model_name, bp, cw):
    """Create a model instance with given params."""
    if model_name == "ElasticNet":
        return LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=bp["C"], l1_ratio=bp["l1_ratio"],
            class_weight="balanced", max_iter=2000, random_state=42)
    elif model_name == "XGBoost":
        return XGBClassifier(
            **bp, gamma=0, objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=42, verbosity=0)
    elif model_name == "LightGBM":
        return LGBMClassifier(
            **bp, class_weight="balanced", random_state=42, verbose=-1)
    elif model_name == "SVM":
        return SVC(**bp, class_weight="balanced", probability=True, random_state=42)


def train_and_predict(model_name, bp, Xtr, Xva, Xte, ytr, yva, cw):
    """Train model and return (predictions, probabilities)."""
    clf = make_model(model_name, bp, cw)
    if model_name == "SVM":
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xte)
        clf.fit(Xtr_s, ytr)
        return clf.predict(Xte_s), clf.predict_proba(Xte_s)
    elif model_name == "XGBoost":
        sw = np.array([cw[y] for y in ytr])
        clf.fit(Xtr, ytr, sample_weight=sw,
                eval_set=[(Xva, yva)], verbose=False)
        return clf.predict(Xte), clf.predict_proba(Xte)
    elif model_name == "LightGBM":
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        return clf.predict(Xte), clf.predict_proba(Xte)
    else:
        clf.fit(Xtr, ytr)
        return clf.predict(Xte), clf.predict_proba(Xte)


def evaluate(y_true, y_pred, y_proba):
    """Compute all metrics + confusion matrix."""
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    n_wors = (y_true == 2).sum()
    sensw = float((y_pred[y_true == 2] == 2).sum()) / max(n_wors, 1)
    n_pred_wors = (y_pred == 2).sum()
    ppvw = float(((y_pred == 2) & (y_true == 2)).sum()) / max(n_pred_wors, 1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return {"AUC": round(auc, 3), "BalAcc": round(bacc, 3),
            "F1macro": round(f1m, 3), "SensW": round(sensw, 3),
            "PPVW": round(ppvw, 3), "confusion_matrix": cm.tolist()}


def bootstrap_evaluate(y_true, y_pred, y_proba, n_boot=BOOTSTRAP_N, seed=SEED):
    """Percentile bootstrap 95% CIs by resampling the test set."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_metrics = {k: [] for k in ["AUC", "BalAcc", "SensW", "PPVW"]}
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yb_true = y_true[idx]
        yb_pred = y_pred[idx]
        yb_proba = y_proba[idx]
        # Skip degenerate resamples
        if len(np.unique(yb_true)) < 2:
            continue
        bacc = balanced_accuracy_score(yb_true, yb_pred)
        n_wors = (yb_true == 2).sum()
        sensw = float((yb_pred[yb_true == 2] == 2).sum()) / max(n_wors, 1) if n_wors > 0 else np.nan
        n_pred_wors = (yb_pred == 2).sum()
        ppvw = float(((yb_pred == 2) & (yb_true == 2)).sum()) / max(n_pred_wors, 1) if n_pred_wors > 0 else np.nan
        try:
            auc = roc_auc_score(yb_true, yb_proba, multi_class="ovr", average="macro")
        except Exception:
            auc = np.nan
        boot_metrics["AUC"].append(auc)
        boot_metrics["BalAcc"].append(bacc)
        boot_metrics["SensW"].append(sensw)
        boot_metrics["PPVW"].append(ppvw)
    ci = {}
    for k, vals in boot_metrics.items():
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals) >= 10:
            ci[f"{k}_lo"] = round(float(np.percentile(vals, 2.5)), 3)
            ci[f"{k}_hi"] = round(float(np.percentile(vals, 97.5)), 3)
        else:
            ci[f"{k}_lo"] = np.nan
            ci[f"{k}_hi"] = np.nan
    return ci


def grid_search_model(model_name, grid, Xtr, Xva, ytr, yva, cw):
    """Run grid search, return best params dict."""
    best_bacc = -1
    best_params = None
    for params in grid:
        try:
            y_pred, _ = train_and_predict(model_name, params, Xtr, Xva, Xva, ytr, yva, cw)
            bacc = balanced_accuracy_score(yva, y_pred)
            if bacc > best_bacc:
                best_bacc = bacc
                best_params = params.copy()
        except Exception:
            continue
    return best_params, best_bacc


# ======================================================================
# Grid search for intake form (the only scenario needing its own model)
# ======================================================================
print("\n[7] Grid search for intake form scenario...")

intake_params = {}
for model_name in MODEL_NAMES:
    print(f"  Grid search: Intake form / {model_name} "
          f"({len(GRIDS[model_name])} combos)...", end=" ", flush=True)
    bp, best_bacc = grid_search_model(
        model_name, GRIDS[model_name], X_demo_tr, X_demo_va, y_tr, y_va, class_wt)
    intake_params[model_name] = bp
    print(f"best_bacc={best_bacc:.3f}, params={bp}")

with open(OUTPUT_DIR / "grid_search_params.yaml", "w") as f:
    yaml.dump({"Intake form only": intake_params}, f, default_flow_style=False)
print(f"  Saved grid search params to {OUTPUT_DIR / 'grid_search_params.yaml'}")


# ======================================================================
# Evaluate all scenarios
# ======================================================================
print("\n[8] Evaluating all scenarios...")

results = []
# Store predictions for bootstrap CIs: {(scenario, model): (y_pred, y_proba)}
predictions = {}

# --- 1. Population baseline (predict all stable) ---
print("\n  [1] Population baseline (predict all stable)...")
y_pred_stable = np.ones(len(y_te), dtype=int)  # predict class 1 (stable) for all
# Probabilities: 100% stable
y_proba_stable = np.zeros((len(y_te), 3))
y_proba_stable[:, 1] = 1.0
m = evaluate(y_te, y_pred_stable, y_proba_stable)
for model_name in MODEL_NAMES:
    row = m.copy()
    row.update({"scenario": "Population baseline",
                "description": "Predict all stable",
                "n_feat": 0, "n_test": len(y_te), "model": model_name})
    results.append(row)
    predictions[("Population baseline", model_name)] = (y_pred_stable, y_proba_stable)
print(f"    AUC={m['AUC']:.3f} BalAcc={m['BalAcc']:.3f} F1macro={m['F1macro']:.3f} "
      f"SensW={m['SensW']:.3f} PPVW={m['PPVW']:.3f}")

# --- 1b. Revert-to-person-mean baseline (rule-based) ---
# Predicts direction of severity change assuming CES-D moves toward person_mean_cesd.
# If severity(person_mean) > severity(current) → worsening
# If severity(person_mean) < severity(current) → improving
# If same severity band → stable
print("\n  [1b] Revert-to-person-mean baseline...")
sev_current_te = severity(prior_test)
pmcesd_te = np.array([
    person_mean_cesd.get(int(p) if hasattr(p, "item") else p, pop_mean)
    for p in pid_test
])
sev_mean_te = severity(pmcesd_te)
y_pred_rtm = np.where(sev_mean_te > sev_current_te, 2,
                       np.where(sev_mean_te < sev_current_te, 0, 1))
# Probabilities: hard assignment (1.0 for predicted class)
y_proba_rtm = np.zeros((len(y_te), 3))
for i, pred in enumerate(y_pred_rtm):
    y_proba_rtm[i, pred] = 1.0
m = evaluate(y_te, y_pred_rtm, y_proba_rtm)
for model_name in MODEL_NAMES:
    row = m.copy()
    row.update({"scenario": "Revert-to-person-mean",
                "description": "Rule: predict severity moves toward person mean",
                "n_feat": 0, "n_test": len(y_te), "model": model_name})
    results.append(row)
    predictions[("Revert-to-person-mean", model_name)] = (y_pred_rtm, y_proba_rtm)
print(f"    AUC={m['AUC']:.3f} BalAcc={m['BalAcc']:.3f} F1macro={m['F1macro']:.3f} "
      f"SensW={m['SensW']:.3f} PPVW={m['PPVW']:.3f}")
print(f"    Distribution: imp={int((y_pred_rtm==0).sum())} stb={int((y_pred_rtm==1).sum())} "
      f"wrs={int((y_pred_rtm==2).sum())}")

# --- 1c. Last-change-only baseline (rule-based) ---
# Predicts that the previous period's severity transition repeats.
# Uses lag cesd_delta: if prior change crossed severity up → predict worsening again, etc.
print("\n  [1c] Last-change-only baseline...")
test_df = all_df[all_df["split"] == "test"].copy()
lag_delta_test = test_df["lag_cesd_delta"].fillna(0).values
# Compute what last period's severity change was:
# prior_cesd at test time is the end-of-last-period cesd.
# The previous period started at (prior_cesd - lag_delta), ended at prior_cesd.
prev_start = prior_test - lag_delta_test
sev_prev_start = severity(np.clip(prev_start, 0, 60))
sev_prev_end = severity(prior_test)  # = current severity
y_pred_lc = np.where(sev_prev_end > sev_prev_start, 2,
                      np.where(sev_prev_end < sev_prev_start, 0, 1))
y_proba_lc = np.zeros((len(y_te), 3))
for i, pred in enumerate(y_pred_lc):
    y_proba_lc[i, pred] = 1.0
m = evaluate(y_te, y_pred_lc, y_proba_lc)
for model_name in MODEL_NAMES:
    row = m.copy()
    row.update({"scenario": "Last-change-only",
                "description": "Rule: repeat last period's severity transition",
                "n_feat": 0, "n_test": len(y_te), "model": model_name})
    results.append(row)
    predictions[("Last-change-only", model_name)] = (y_pred_lc, y_proba_lc)
print(f"    AUC={m['AUC']:.3f} BalAcc={m['BalAcc']:.3f} F1macro={m['F1macro']:.3f} "
      f"SensW={m['SensW']:.3f} PPVW={m['PPVW']:.3f}")
print(f"    Distribution: imp={int((y_pred_lc==0).sum())} stb={int((y_pred_lc==1).sum())} "
      f"wrs={int((y_pred_lc==2).sum())}")

# --- 2. Intake form only (grid-searched) ---
print("\n  [2] Intake form only (demographics)...")
for model_name in MODEL_NAMES:
    bp = intake_params[model_name]
    y_pred, y_proba = train_and_predict(
        model_name, bp, X_demo_tr, X_demo_va, X_demo_te, y_tr, y_va, class_wt)
    m = evaluate(y_te, y_pred, y_proba)
    m.update({"scenario": "Intake form only", "description": "Age + gender (3 feat)",
              "n_feat": 3, "n_test": len(y_te), "model": model_name})
    results.append(m)
    predictions[("Intake form only", model_name)] = (y_pred, y_proba)
    print(f"    {model_name}: AUC={m['AUC']:.3f} BalAcc={m['BalAcc']:.3f} "
          f"SensW={m['SensW']:.3f} PPVW={m['PPVW']:.3f}")

# --- 3. Onboarding (first CES-D survey) ---
print("\n  [3] Onboarding (first CES-D survey)...")
for model_name in MODEL_NAMES:
    bp = params_39[model_name]
    y_pred, y_proba = train_and_predict(
        model_name, bp, X39_tr, X39_va, X39_te_onboard, y_tr, y_va, class_wt)
    m = evaluate(y_te, y_pred, y_proba)
    m.update({"scenario": "Onboarding", "description": "39-feat, prior_cesd & pmcesd = intake CES-D",
              "n_feat": 39, "n_test": len(y_te), "model": model_name})
    results.append(m)
    predictions[("Onboarding", model_name)] = (y_pred, y_proba)
    print(f"    {model_name}: AUC={m['AUC']:.3f} BalAcc={m['BalAcc']:.3f} "
          f"SensW={m['SensW']:.3f} PPVW={m['PPVW']:.3f}")

# --- 4a. Stale 4 weeks ---
print("\n  [4a] Running — stale 4 weeks...")
for model_name in MODEL_NAMES:
    bp = params_39[model_name]
    y_pred, y_proba = train_and_predict(
        model_name, bp, X39_tr, X39_va, X39_te_stale4, y_tr, y_va, class_wt)
    m = evaluate(y_te, y_pred, y_proba)
    m.update({"scenario": "Stale 4 weeks", "description": "39-feat, prior_cesd from t-1",
              "n_feat": 39, "n_test": len(y_te), "model": model_name})
    results.append(m)
    predictions[("Stale 4 weeks", model_name)] = (y_pred, y_proba)
    print(f"    {model_name}: AUC={m['AUC']:.3f} BalAcc={m['BalAcc']:.3f} "
          f"SensW={m['SensW']:.3f} PPVW={m['PPVW']:.3f}")

# --- 4b. Stale 8 weeks ---
print("\n  [4b] Running — stale 8 weeks...")
for model_name in MODEL_NAMES:
    bp = params_39[model_name]
    y_pred, y_proba = train_and_predict(
        model_name, bp, X39_tr, X39_va, X39_te_stale8, y_tr, y_va, class_wt)
    m = evaluate(y_te, y_pred, y_proba)
    m.update({"scenario": "Stale 8 weeks", "description": "39-feat, prior_cesd from t-2",
              "n_feat": 39, "n_test": len(y_te), "model": model_name})
    results.append(m)
    predictions[("Stale 8 weeks", model_name)] = (y_pred, y_proba)
    print(f"    {model_name}: AUC={m['AUC']:.3f} BalAcc={m['BalAcc']:.3f} "
          f"SensW={m['SensW']:.3f} PPVW={m['PPVW']:.3f}")

# --- 5. No fresh CES-D ---
print("\n  [5] No fresh CES-D (prior_cesd = pop_mean)...")
for model_name in MODEL_NAMES:
    bp = params_39[model_name]
    y_pred, y_proba = train_and_predict(
        model_name, bp, X39_tr, X39_va, X39_te_nocesd, y_tr, y_va, class_wt)
    m = evaluate(y_te, y_pred, y_proba)
    m.update({"scenario": "No fresh CES-D",
              "description": "39-feat, prior_cesd = pop_mean",
              "n_feat": 39, "n_test": len(y_te), "model": model_name})
    results.append(m)
    predictions[("No fresh CES-D", model_name)] = (y_pred, y_proba)
    print(f"    {model_name}: AUC={m['AUC']:.3f} BalAcc={m['BalAcc']:.3f} "
          f"SensW={m['SensW']:.3f} PPVW={m['PPVW']:.3f}")

# --- 6. Cold start (repeated leave-group-out by PID) ---
# Rigorous person-level generalization test:
#   - 5 repeats × 5-fold GroupKFold = 25 evaluations per model
#   - Each repeat shuffles persons into different fold assignments
#   - Train on non-held-out PIDs' TRAIN data only
#   - Val on non-held-out PIDs' VAL data (for early stopping)
#   - Test on held-out PIDs' TEST data, with pmcesd = pop_mean
print(f"\n  [6] Cold start — {COLD_START_REPEATS}x{COLD_START_FOLDS}-fold "
      f"repeated leave-group-out CV ({COLD_START_REPEATS * COLD_START_FOLDS} evaluations)...")

# All unique PIDs
all_pids = np.unique(np.concatenate([pid_train, pid_val, pid_test]))
print(f"    Total unique PIDs: {len(all_pids)}")

def _pid_int(p):
    return int(p) if hasattr(p, "item") else p

cold_start_fold_results = []

for model_name in MODEL_NAMES:
    bp = params_39[model_name]
    # Collect per-fold metrics for summary stats
    fold_aucs = []
    fold_baccs = []
    fold_sensws = []

    for repeat in range(COLD_START_REPEATS):
        # Shuffle PIDs differently each repeat
        rng = np.random.RandomState(SEED + repeat * 100)
        shuffled_pids = all_pids.copy()
        rng.shuffle(shuffled_pids)
        pid_to_fold = {int(pid) if hasattr(pid, "item") else pid: i % COLD_START_FOLDS
                       for i, pid in enumerate(shuffled_pids)}

        for fold_i in range(COLD_START_FOLDS):
            # Held-out PIDs for this fold
            held_pids = {pid for pid, f in pid_to_fold.items() if f == fold_i}

            # Training: TRAIN-split observations from NON-held-out PIDs only
            tr_mask = np.array([_pid_int(p) not in held_pids for p in pid_train])
            Xtr_f = X39_tr[tr_mask]
            ytr_f = y_tr[tr_mask]

            # Validation: VAL-split observations from NON-held-out PIDs only
            va_mask = np.array([_pid_int(p) not in held_pids for p in pid_val])
            Xva_f = X39_va[va_mask]
            yva_f = y_va[va_mask]

            # Test: held-out PIDs' TEST-period observations only
            te_mask = np.array([_pid_int(p) in held_pids for p in pid_test])
            Xte_f = X39_te[te_mask].copy()
            yte_f = y_te[te_mask]
            pid_te_f = pid_test[te_mask]

            if len(yte_f) == 0:
                continue

            # Cold start: set pmcesd = pop_mean (computed from training fold only)
            fold_pop_mean = float(Xtr_f[:, prior_cesd_idx].mean())
            Xte_f[:, -1] = fold_pop_mean

            # Class weights for this fold
            cls_f, cnt_f = np.unique(ytr_f, return_counts=True)
            cw_f = {c: len(ytr_f) / (len(cls_f) * cnt) for c, cnt in zip(cls_f, cnt_f)}

            y_pred_f, y_proba_f = train_and_predict(
                model_name, bp, Xtr_f, Xva_f, Xte_f, ytr_f, yva_f, cw_f)

            # Per-fold metrics
            fold_m = evaluate(yte_f, y_pred_f, y_proba_f)
            fold_held = np.unique(pid_te_f)
            cold_start_fold_results.append({
                "model": model_name, "repeat": repeat, "fold": fold_i,
                "n_held_pids": len(fold_held), "n_obs": len(yte_f),
                "n_train": len(ytr_f), "n_val": len(yva_f),
                **{k: v for k, v in fold_m.items() if k != "confusion_matrix"}
            })
            fold_aucs.append(fold_m["AUC"])
            fold_baccs.append(fold_m["BalAcc"])
            fold_sensws.append(fold_m["SensW"])

        if repeat == COLD_START_REPEATS - 1:
            print(f"    {model_name}: repeat {repeat+1}/{COLD_START_REPEATS} "
                  f"fold {fold_i+1}/{COLD_START_FOLDS} done")

    # Summary stats across all 25 folds
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    mean_bacc = np.mean(fold_baccs)
    mean_sensw = np.mean(fold_sensws)
    mean_ppvw = np.mean([r["PPVW"] for r in cold_start_fold_results
                         if r["model"] == model_name])

    m = {"AUC": round(mean_auc, 3), "BalAcc": round(mean_bacc, 3),
         "F1macro": round(np.mean([r["F1macro"] for r in cold_start_fold_results
                                   if r["model"] == model_name]), 3),
         "SensW": round(mean_sensw, 3), "PPVW": round(mean_ppvw, 3),
         "confusion_matrix": None}
    m.update({"scenario": "Cold start",
              "description": f"Leave-group-out ({COLD_START_REPEATS}x{COLD_START_FOLDS} folds), "
                             f"train/val/test split, pmcesd=pop_mean",
              "n_feat": 39, "n_test": len(pid_test), "model": model_name})
    results.append(m)
    print(f"    {model_name}: AUC={mean_auc:.3f}±{std_auc:.3f} "
          f"BalAcc={mean_bacc:.3f} SensW={mean_sensw:.3f} PPVW={mean_ppvw:.3f} "
          f"({COLD_START_REPEATS * COLD_START_FOLDS} folds)")

# Save per-fold cold start results
pd.DataFrame(cold_start_fold_results).to_csv(
    OUTPUT_DIR / "cold_start_fold_results.csv", index=False)

# --- 7. Full model ---
print("\n  [7] Full model...")
for model_name in MODEL_NAMES:
    bp = params_39[model_name]
    y_pred, y_proba = train_and_predict(
        model_name, bp, X39_tr, X39_va, X39_te, y_tr, y_va, class_wt)
    m = evaluate(y_te, y_pred, y_proba)
    m.update({"scenario": "Full model", "description": "All 39 features, known person",
              "n_feat": 39, "n_test": len(y_te), "model": model_name})
    results.append(m)
    predictions[("Full model", model_name)] = (y_pred, y_proba)
    print(f"    {model_name}: AUC={m['AUC']:.3f} BalAcc={m['BalAcc']:.3f} "
          f"SensW={m['SensW']:.3f} PPVW={m['PPVW']:.3f}")

# ======================================================================
# Bootstrap CIs for all scenarios (except cold start which uses fold variance)
# ======================================================================
print(f"\n[8b] Computing bootstrap 95% CIs ({BOOTSTRAP_N} resamples)...")

bootstrap_rows = []
BOOTSTRAP_SCENARIOS = [
    "Population baseline", "Revert-to-person-mean", "Last-change-only",
    "Intake form only", "Onboarding", "Stale 4 weeks", "Stale 8 weeks",
    "No fresh CES-D", "Full model",
]

for sc in BOOTSTRAP_SCENARIOS:
    for model_name in MODEL_NAMES:
        key = (sc, model_name)
        if key not in predictions:
            continue
        y_pred_b, y_proba_b = predictions[key]
        ci = bootstrap_evaluate(y_te, y_pred_b, y_proba_b)
        ci.update({"scenario": sc, "model": model_name})
        bootstrap_rows.append(ci)

bootstrap_df = pd.DataFrame(bootstrap_rows)
bootstrap_df.to_csv(OUTPUT_DIR / "deployment_bootstrap_ci.csv", index=False)
print(f"  Saved bootstrap CIs to {OUTPUT_DIR / 'deployment_bootstrap_ci.csv'}")

# Merge CIs into results for report
ci_lookup = {}
for _, row in bootstrap_df.iterrows():
    ci_lookup[(row["scenario"], row["model"])] = row

# ======================================================================
# Save results
# ======================================================================
print("\n[9] Saving results...")

# Save confusion matrices separately
cm_rows = []
for r in results:
    if r.get("confusion_matrix") is not None:
        cm = r["confusion_matrix"]
        cm_rows.append({
            "scenario": r["scenario"], "model": r["model"],
            "cm_00": cm[0][0], "cm_01": cm[0][1], "cm_02": cm[0][2],
            "cm_10": cm[1][0], "cm_11": cm[1][1], "cm_12": cm[1][2],
            "cm_20": cm[2][0], "cm_21": cm[2][1], "cm_22": cm[2][2],
        })
if cm_rows:
    pd.DataFrame(cm_rows).to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)
    print(f"  Saved confusion matrices to {OUTPUT_DIR / 'confusion_matrices.csv'}")

# Drop confusion_matrix from results before saving CSV
results_clean = [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in results]
results_df = pd.DataFrame(results_clean)
results_df.to_csv(OUTPUT_DIR / "deployment_results.csv", index=False)

# ======================================================================
# Generate markdown report
# ======================================================================
print(f"\n[10] Writing report to {REPORT_PATH}...")

SCENARIO_ORDER = [
    "Population baseline",
    "Revert-to-person-mean",
    "Last-change-only",
    "Intake form only",
    "Onboarding",
    "Stale 4 weeks",
    "Stale 8 weeks",
    "No fresh CES-D",
    "Cold start",
    "Full model",
]

cs_df = pd.DataFrame(cold_start_fold_results)

lines = []
lines.append("# Deployment Scenario Results — sev_crossing label\n")
lines.append("")
lines.append("## Design\n")
lines.append("")
lines.append("- **Population baseline**: predict all stable (no model)")
lines.append("- **Revert-to-person-mean**: rule-based — predict severity moves toward "
             "each person's training-period mean CES-D (regression-to-the-mean null hypothesis)")
lines.append("- **Last-change-only**: rule-based — predict the previous period's severity "
             "transition repeats (momentum baseline)")
lines.append("- **Intake form only**: 3 demographic features, grid-searched model")
lines.append("- **All other scenarios**: same 39-feature model (grid-searched params from "
             "bootstrap analysis), evaluated with degraded test inputs")
lines.append("- **Cold start**: repeated leave-group-out CV by PID — held-out people never seen "
             f"during training, pmcesd = pop_mean ({COLD_START_REPEATS} repeats × "
             f"{COLD_START_FOLDS} folds = {COLD_START_REPEATS * COLD_START_FOLDS} evaluations, "
             f"{len(all_pids)} PIDs total)")
lines.append("")

# Deployment ladder (XGBoost)
lines.append("## Deployment Ladder (XGBoost)\n")
lines.append("")
lines.append("| Stage | What you know | N feat | AUC [95% CI] | BalAcc [95% CI] | Sens-W [95% CI] | PPV-W [95% CI] |")
lines.append("|---|---|---|---|---|---|---|")


def fmt_ci(val, lo, hi):
    """Format value with CI, handling NaN."""
    if np.isnan(lo) or np.isnan(hi):
        return f"{val:.3f}"
    return f"{val:.3f} [{lo:.3f}, {hi:.3f}]"


for sc in SCENARIO_ORDER:
    row = results_df[(results_df["scenario"] == sc) & (results_df["model"] == "XGBoost")].iloc[0]
    ci_key = (sc, "XGBoost")
    if sc == "Cold start":
        # Use fold-level variance for cold start
        cs_xgb = cs_df[cs_df["model"] == "XGBoost"]
        auc_str = f"{row['AUC']:.3f} [{cs_xgb['AUC'].mean() - 1.96*cs_xgb['AUC'].std():.3f}, {cs_xgb['AUC'].mean() + 1.96*cs_xgb['AUC'].std():.3f}]"
        bacc_str = f"{row['BalAcc']:.3f} [{cs_xgb['BalAcc'].mean() - 1.96*cs_xgb['BalAcc'].std():.3f}, {cs_xgb['BalAcc'].mean() + 1.96*cs_xgb['BalAcc'].std():.3f}]"
        sensw_str = f"{row['SensW']:.3f} [{cs_xgb['SensW'].mean() - 1.96*cs_xgb['SensW'].std():.3f}, {cs_xgb['SensW'].mean() + 1.96*cs_xgb['SensW'].std():.3f}]"
        ppvw_str = f"{row['PPVW']:.3f} [{cs_xgb['PPVW'].mean() - 1.96*cs_xgb['PPVW'].std():.3f}, {cs_xgb['PPVW'].mean() + 1.96*cs_xgb['PPVW'].std():.3f}]"
    elif ci_key in ci_lookup:
        ci = ci_lookup[ci_key]
        auc_str = fmt_ci(row['AUC'], ci['AUC_lo'], ci['AUC_hi'])
        bacc_str = fmt_ci(row['BalAcc'], ci['BalAcc_lo'], ci['BalAcc_hi'])
        sensw_str = fmt_ci(row['SensW'], ci['SensW_lo'], ci['SensW_hi'])
        ppvw_str = fmt_ci(row['PPVW'], ci['PPVW_lo'], ci['PPVW_hi'])
    else:
        auc_str = f"{row['AUC']:.3f}"
        bacc_str = f"{row['BalAcc']:.3f}"
        sensw_str = f"{row['SensW']:.3f}"
        ppvw_str = f"{row['PPVW']:.3f}"
    lines.append(f"| {sc} | {row['description']} | {row['n_feat']} | "
                 f"{auc_str} | {bacc_str} | {sensw_str} | {ppvw_str} |")

lines.append("")

# Full results (all models)
lines.append("## Full Results (All Models)\n")
lines.append("")
lines.append("| Scenario | Model | N feat | AUC [95% CI] | BalAcc [95% CI] | Sens-W [95% CI] | PPV-W [95% CI] |")
lines.append("|---|---|---|---|---|---|---|")

for sc in SCENARIO_ORDER:
    for model_name in MODEL_NAMES:
        row = results_df[(results_df["scenario"] == sc) &
                         (results_df["model"] == model_name)].iloc[0]
        ci_key = (sc, model_name)
        if sc == "Cold start":
            cs_m = cs_df[cs_df["model"] == model_name]
            auc_str = f"{row['AUC']:.3f} ±{cs_m['AUC'].std():.3f}"
            bacc_str = f"{row['BalAcc']:.3f} ±{cs_m['BalAcc'].std():.3f}"
            sensw_str = f"{row['SensW']:.3f} ±{cs_m['SensW'].std():.3f}"
            ppvw_str = f"{row['PPVW']:.3f} ±{cs_m['PPVW'].std():.3f}"
        elif ci_key in ci_lookup:
            ci = ci_lookup[ci_key]
            auc_str = fmt_ci(row['AUC'], ci['AUC_lo'], ci['AUC_hi'])
            bacc_str = fmt_ci(row['BalAcc'], ci['BalAcc_lo'], ci['BalAcc_hi'])
            sensw_str = fmt_ci(row['SensW'], ci['SensW_lo'], ci['SensW_hi'])
            ppvw_str = fmt_ci(row['PPVW'], ci['PPVW_lo'], ci['PPVW_hi'])
        else:
            auc_str = f"{row['AUC']:.3f}"
            bacc_str = f"{row['BalAcc']:.3f}"
            sensw_str = f"{row['SensW']:.3f}"
            ppvw_str = f"{row['PPVW']:.3f}"
        lines.append(f"| {sc} | {model_name} | {row['n_feat']} | "
                     f"{auc_str} | {bacc_str} | {sensw_str} | {ppvw_str} |")

lines.append("")

# Cold start summary and fold details
lines.append(f"## Cold Start — {COLD_START_REPEATS}×{COLD_START_FOLDS}-Fold "
             f"Summary ({COLD_START_REPEATS * COLD_START_FOLDS} evaluations)\n")
lines.append("")

# Summary table: mean ± SD across all folds
lines.append("### Summary (mean ± SD across all folds)\n")
lines.append("")
lines.append("| Model | AUC | BalAcc | F1-macro | Sens-W | PPV-W |")
lines.append("|---|---|---|---|---|---|")
for model_name in MODEL_NAMES:
    mdf = cs_df[cs_df["model"] == model_name]
    lines.append(
        f"| {model_name} | "
        f"{mdf['AUC'].mean():.3f}±{mdf['AUC'].std():.3f} | "
        f"{mdf['BalAcc'].mean():.3f}±{mdf['BalAcc'].std():.3f} | "
        f"{mdf['F1macro'].mean():.3f}±{mdf['F1macro'].std():.3f} | "
        f"{mdf['SensW'].mean():.3f}±{mdf['SensW'].std():.3f} | "
        f"{mdf['PPVW'].mean():.3f}±{mdf['PPVW'].std():.3f} |")
lines.append("")

# Per-fold details
lines.append("### Per-Fold Details\n")
lines.append("")
for model_name in MODEL_NAMES:
    mdf = cs_df[cs_df["model"] == model_name]
    lines.append(f"**{model_name}**:")
    lines.append("")
    lines.append("| Repeat | Fold | Held PIDs | N obs | AUC | BalAcc | F1-macro | Sens-W | PPV-W |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in mdf.iterrows():
        lines.append(f"| {int(r['repeat'])} | {int(r['fold'])} | {int(r['n_held_pids'])} | "
                     f"{int(r['n_obs'])} | "
                     f"{r['AUC']:.3f} | {r['BalAcc']:.3f} | {r['F1macro']:.3f} | "
                     f"{r['SensW']:.3f} | {r['PPVW']:.3f} |")
    lines.append("")

# Key insights
lines.append("## Key Insights\n")
lines.append("")

xgb_results = {sc: results_df[(results_df["scenario"] == sc) &
                               (results_df["model"] == "XGBoost")].iloc[0]
                for sc in SCENARIO_ORDER}

lines.append(f"1. **Intake → Onboarding**: "
             f"ΔAUC = {xgb_results['Onboarding']['AUC'] - xgb_results['Intake form only']['AUC']:+.3f}")
lines.append(f"2. **Onboarding → Full model**: "
             f"ΔAUC = {xgb_results['Full model']['AUC'] - xgb_results['Onboarding']['AUC']:+.3f}")
lines.append(f"3. **Stale 4wk degradation**: "
             f"ΔAUC = {xgb_results['Stale 4 weeks']['AUC'] - xgb_results['Full model']['AUC']:+.3f}")
lines.append(f"4. **Cold start vs full**: "
             f"ΔAUC = {xgb_results['Cold start']['AUC'] - xgb_results['Full model']['AUC']:+.3f}")
lines.append(f"5. **No fresh CES-D vs full**: "
             f"ΔAUC = {xgb_results['No fresh CES-D']['AUC'] - xgb_results['Full model']['AUC']:+.3f}")
lines.append("")

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(lines))

print(f"  Report saved to {REPORT_PATH}")
print(f"  CSV saved to {OUTPUT_DIR / 'deployment_results.csv'}")

print("\n" + "=" * 80)
print("DEPLOYMENT SCENARIO EVALUATION COMPLETE")
print("=" * 80)
