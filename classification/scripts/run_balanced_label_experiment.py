"""Balanced tercile labeling experiment with per-label grid search.

Splits training-set CESD delta into 3 equal-sized bins via rank-based
assignment, then grid-searches hyperparameters for each of 4 models on the
39-feature condition (val set). Best params are then used to evaluate all
feature conditions on the test set.

Models: ElasticNet, XGBoost, LightGBM, SVM
Feature conditions:
  1. prior_cesd only (1 feature)
  2. base: prior_cesd + behavioral + demographics (21 features)
  3. base + behavioral lags excl age/gender (38 features)
  4. base + behavioral lags + person_mean_cesd (39 features)

Grid search spaces match those used for sev_crossing to ensure a fair
cross-label comparison.
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
CONFIG = yaml.safe_load(open("configs/models/classifier.yaml"))
OUTPUT_DIR = Path("models/classifier_balanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/balanced_label_results.md")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Balanced tercile labels
# ---------------------------------------------------------------------------
print("\n[2] Building balanced tercile labels (rank-based)...")

def make_rank_balanced_labels(y_delta, rng=None):
    """Assign labels by rank: bottom third=improving, middle=stable, top=worsening.
    Ties at bin boundaries are broken randomly (seeded for reproducibility).
    """
    n = len(y_delta)
    n_per = n // 3
    # Stable random tiebreaker so identical delta values get shuffled
    if rng is None:
        rng = np.random.RandomState(42)
    order = np.lexsort((rng.random(n), y_delta))  # sort by delta, break ties randomly
    labels = np.empty(n, dtype=int)
    labels[order[:n_per]] = 0            # improving (lowest deltas)
    labels[order[n_per:2 * n_per]] = 1   # stable (middle)
    labels[order[2 * n_per:]] = 2        # worsening (highest deltas)
    return labels

label_names = ["improving", "stable", "worsening"]

ytr = make_rank_balanced_labels(y_train, rng=np.random.RandomState(42))
# For val/test: use training-set boundary values so labels are deterministic
# Boundaries = midpoint between the max of class k and min of class k+1 in training
sorted_train = np.sort(y_train)
n_per = len(y_train) // 3
tercile_lo = (sorted_train[n_per - 1] + sorted_train[n_per]) / 2.0
tercile_hi = (sorted_train[2 * n_per - 1] + sorted_train[2 * n_per]) / 2.0
# For val/test, ties at boundaries are assigned to the class that keeps balance
# Use rank-based assignment on val/test independently to ensure equal sizes
yva = make_rank_balanced_labels(y_val, rng=np.random.RandomState(42))
yte = make_rank_balanced_labels(y_test, rng=np.random.RandomState(42))

print(f"  Training boundary midpoints: lo={tercile_lo:.2f}, hi={tercile_hi:.2f}")

for split_name, labels in [("Train", ytr), ("Val", yva), ("Test", yte)]:
    dist = " | ".join(
        f"{label_names[i]}={int((labels == i).sum())} ({(labels == i).mean() * 100:.0f}%)"
        for i in range(3)
    )
    print(f"  {split_name}: {dist}")

# ---------------------------------------------------------------------------
# Build lag features (17 behavioral + person_mean_cesd)
# ---------------------------------------------------------------------------
print("\n[3] Building lag features...")

feat_cols = base_feature_names
lag_cols_all = [f"lag_{c}" for c in feat_cols] + ["lag_cesd_delta"]

all_df = pd.concat([
    pd.read_csv(DATA_DIR / "train_scaled.csv"),
    pd.read_csv(DATA_DIR / "val_scaled.csv"),
    pd.read_csv(DATA_DIR / "test_scaled.csv"),
]).sort_values(["pid", "period_number"]).reset_index(drop=True)

for col in [c for c in feat_cols if c in all_df.columns]:
    all_df[f"lag_{col}"] = all_df.groupby("pid")[col].shift(1)
for col in feat_cols:
    if f"lag_{col}" not in all_df.columns:
        all_df[f"lag_{col}"] = 0.0
all_df["lag_cesd_delta"] = all_df.groupby("pid")["target_cesd_delta"].shift(1)
all_df[lag_cols_all] = all_df[lag_cols_all].fillna(0)

# Exclude static + clinical lags (same as best39 model)
drop_lags = [
    "lag_age", "lag_gender_mode_1", "lag_gender_mode_2",
    "lag_prior_cesd", "lag_cesd_delta",
]
keep_idx = [i for i, c in enumerate(lag_cols_all) if c not in drop_lags]
lag_cols = [lag_cols_all[i] for i in keep_idx]

lag_tr = all_df[all_df["split"] == "train"][lag_cols_all].values[:, keep_idx]
lag_va = all_df[all_df["split"] == "val"][lag_cols_all].values[:, keep_idx]
lag_te = all_df[all_df["split"] == "test"][lag_cols_all].values[:, keep_idx]

# person_mean_cesd
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
    # Compute from training data
    person_mean_cesd = {}
    for pid in np.unique(pid_train):
        person_mean_cesd[pid] = float(prior_train[pid_train == pid].mean())

pop_mean = np.mean(list(person_mean_cesd.values()))

def get_pmcesd(pids):
    return np.array([
        person_mean_cesd.get(int(p) if hasattr(p, "item") else p, pop_mean)
        for p in pids
    ]).reshape(-1, 1)

# Assemble 39-feature matrices
X39_tr = np.hstack([X_train, lag_tr, get_pmcesd(pid_train)])
X39_va = np.hstack([X_val, lag_va, get_pmcesd(pid_val)])
X39_te = np.hstack([X_test, lag_te, get_pmcesd(pid_test)])
feat_names_39 = base_feature_names + lag_cols + ["person_mean_cesd"]

print(f"  39-feature matrices: train={X39_tr.shape}, val={X39_va.shape}, test={X39_te.shape}")
print(f"  Lag features: {len(lag_cols)} behavioral")

# ---------------------------------------------------------------------------
# Feature conditions
# ---------------------------------------------------------------------------
# Condition 1: prior_cesd only (1 feature)
prior_idx = [0]

# Condition 2: base = prior_cesd + behavioral + demographics (21 features)
base_idx = list(range(X_train.shape[1]))

# Condition 3: base + behavioral lags (excl age/gender lags) = 38 features
X38_tr = np.hstack([X_train, lag_tr])
X38_va = np.hstack([X_val, lag_va])
X38_te = np.hstack([X_test, lag_te])
feat_names_38 = base_feature_names + lag_cols

# Condition 4: base + behavioral lags + person_mean_cesd = 39 features
# (X39_tr/va/te already built above)

CONDITIONS = {
    "prior_cesd only": {
        "train": X_train[:, prior_idx], "val": X_val[:, prior_idx], "test": X_test[:, prior_idx],
        "n_feat": 1,
    },
    "base (21)": {
        "train": X_train[:, base_idx], "val": X_val[:, base_idx], "test": X_test[:, base_idx],
        "n_feat": len(base_idx),
    },
    "base + behavioral lag (38)": {
        "train": X38_tr, "val": X38_va, "test": X38_te,
        "n_feat": 38,
    },
    "base + behavioral lag + pmcesd (39)": {
        "train": X39_tr, "val": X39_va, "test": X39_te,
        "n_feat": 39,
    },
}

# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------


def eval_metrics(y_true, y_proba, y_pred):
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    sens_w = float((y_pred[y_true == 2] == 2).sum()) / max(int((y_true == 2).sum()), 1)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"BalAcc": bacc, "AUC": auc, "Sens_W": sens_w, "F1_macro": f1m}


# ---------------------------------------------------------------------------
# Grid search spaces — same as used for sev_crossing
# ---------------------------------------------------------------------------

# ElasticNet: 8 C × 4 l1_ratio = 32 combos
EN_CS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
EN_L1S = [0.1, 0.5, 0.9, 0.99]

# XGBoost: match original 64-combo grid
XGB_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# LightGBM: randomized search over same grid (300 samples from ~3888 combos)
LGB_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 20, 30],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.0, 0.1, 1.0],
}
LGB_N_RANDOM = 300

# SVM: 5 C × 4 gamma × 2 kernels = 40 combos
SVM_CS = [0.1, 0.5, 1.0, 5.0, 10.0]
SVM_GAMMAS = [0.0001, 0.001, 0.01, 0.1]
SVM_KERNELS = ["rbf", "linear"]

# ======================================================================
# Step 4: Grid search — per model × per feature condition (val set)
# ======================================================================
from itertools import product as iproduct

print("\n[4] Grid search per model × feature condition...")
print("=" * 80)

# Compute balanced sample weights for XGBoost
classes, counts = np.unique(ytr, return_counts=True)
n_total = len(ytr)
class_wt = {c: n_total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
xgb_sample_wt = np.array([class_wt[y] for y in ytr])

# Pre-compute XGB and LGB combo lists
xgb_keys = list(XGB_GRID.keys())
xgb_combos = list(iproduct(*[XGB_GRID[k] for k in xgb_keys]))

rng_lgb = np.random.RandomState(42)
lgb_keys = list(LGB_GRID.keys())
lgb_all_combos = list(iproduct(*[LGB_GRID[k] for k in lgb_keys]))
lgb_sample_idx = rng_lgb.choice(
    len(lgb_all_combos), size=min(LGB_N_RANDOM, len(lgb_all_combos)), replace=False)
lgb_combos = [lgb_all_combos[i] for i in lgb_sample_idx]

svm_combos = list(iproduct(SVM_CS, SVM_GAMMAS, SVM_KERNELS))

# best_params[cond_name][model_name] = {params_dict}
# best_baccs[cond_name][model_name]  = float
best_params = {}
best_baccs = {}
grid_rows = []

MODEL_NAMES = ["ElasticNet", "XGBoost", "LightGBM", "SVM"]

for cond_name, cond_data in CONDITIONS.items():
    Xtr_c = cond_data["train"]
    Xva_c = cond_data["val"]
    n_feat = cond_data["n_feat"]

    best_params[cond_name] = {}
    best_baccs[cond_name] = {}

    print(f"\n  --- {cond_name} ({n_feat} features) ---")

    # --- ElasticNet ---
    best_b, best_p = -1, None
    for C in EN_CS:
        for l1r in EN_L1S:
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", C=C, l1_ratio=l1r,
                class_weight="balanced", max_iter=2000, random_state=42,
            )
            clf.fit(Xtr_c, ytr)
            bacc = balanced_accuracy_score(yva, clf.predict(Xva_c))
            grid_rows.append({"condition": cond_name, "model": "ElasticNet",
                              "params": f"C={C},l1r={l1r}", "val_bacc": round(bacc, 4)})
            if bacc > best_b:
                best_b, best_p = bacc, {"C": C, "l1_ratio": l1r}
    best_params[cond_name]["ElasticNet"] = best_p
    best_baccs[cond_name]["ElasticNet"] = best_b
    print(f"    ElasticNet  best={best_p}  val_bacc={best_b:.4f}")

    # --- XGBoost ---
    best_b, best_p = -1, None
    for i, vals in enumerate(xgb_combos):
        p = dict(zip(xgb_keys, vals))
        clf = XGBClassifier(
            **p, gamma=0, objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=42, verbosity=0,
        )
        clf.fit(Xtr_c, ytr, sample_weight=xgb_sample_wt,
                eval_set=[(Xva_c, yva)], verbose=False)
        bacc = balanced_accuracy_score(yva, clf.predict(Xva_c))
        grid_rows.append({"condition": cond_name, "model": "XGBoost",
                          "params": str(p), "val_bacc": round(bacc, 4)})
        if bacc > best_b:
            best_b, best_p = bacc, p.copy()
    best_params[cond_name]["XGBoost"] = best_p
    best_baccs[cond_name]["XGBoost"] = best_b
    print(f"    XGBoost     best={best_p}  val_bacc={best_b:.4f}")

    # --- LightGBM ---
    best_b, best_p = -1, None
    for i, vals in enumerate(lgb_combos):
        p = dict(zip(lgb_keys, vals))
        clf = LGBMClassifier(**p, class_weight="balanced", random_state=42, verbose=-1)
        clf.fit(Xtr_c, ytr, eval_set=[(Xva_c, yva)])
        bacc = balanced_accuracy_score(yva, clf.predict(Xva_c))
        grid_rows.append({"condition": cond_name, "model": "LightGBM",
                          "params": str(p), "val_bacc": round(bacc, 4)})
        if bacc > best_b:
            best_b, best_p = bacc, p.copy()
    best_params[cond_name]["LightGBM"] = best_p
    best_baccs[cond_name]["LightGBM"] = best_b
    print(f"    LightGBM    best={best_p}  val_bacc={best_b:.4f}")

    # --- SVM ---
    svm_scaler = StandardScaler()
    Xtr_s = svm_scaler.fit_transform(Xtr_c)
    Xva_s = svm_scaler.transform(Xva_c)
    best_b, best_p = -1, None
    for C, gamma, kernel in svm_combos:
        clf = SVC(C=C, gamma=gamma, kernel=kernel, class_weight="balanced",
                  probability=True, random_state=42)
        clf.fit(Xtr_s, ytr)
        bacc = balanced_accuracy_score(yva, clf.predict(Xva_s))
        grid_rows.append({"condition": cond_name, "model": "SVM",
                          "params": f"C={C},gamma={gamma},kernel={kernel}",
                          "val_bacc": round(bacc, 4)})
        if bacc > best_b:
            best_b, best_p = bacc, {"C": C, "gamma": gamma, "kernel": kernel}
    best_params[cond_name]["SVM"] = best_p
    best_baccs[cond_name]["SVM"] = best_b
    print(f"    SVM         best={best_p}  val_bacc={best_b:.4f}")

# Save all grid search results
pd.DataFrame(grid_rows).to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False)

# Save best params
with open(OUTPUT_DIR / "best_params.yaml", "w") as f:
    yaml.dump({cond: {m: p for m, p in models.items()}
               for cond, models in best_params.items()}, f, default_flow_style=False)

# ======================================================================
# Step 5: Evaluate all conditions with their own tuned params (test set)
# ======================================================================
print("\n\n[5] Evaluating all conditions with per-condition tuned params (test set)...")
print("=" * 80)


def train_and_predict(model_name, bp, Xtr, Xva, Xte, ytr, yva):
    """Train model with given best params."""
    if model_name == "ElasticNet":
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=bp["C"], l1_ratio=bp["l1_ratio"],
            class_weight="balanced", max_iter=2000, random_state=42,
        )
        clf.fit(Xtr, ytr)
        return clf, clf.predict(Xte), clf.predict_proba(Xte)

    elif model_name == "XGBoost":
        clf = XGBClassifier(
            **bp, gamma=0, objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=42, verbosity=0,
        )
        clf.fit(Xtr, ytr, sample_weight=np.array([class_wt[y] for y in ytr]),
                eval_set=[(Xva, yva)], verbose=False)
        return clf, clf.predict(Xte), clf.predict_proba(Xte)

    elif model_name == "LightGBM":
        clf = LGBMClassifier(
            **bp, class_weight="balanced", random_state=42, verbose=-1,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        return clf, clf.predict(Xte), clf.predict_proba(Xte)

    elif model_name == "SVM":
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        clf = SVC(**bp, class_weight="balanced", probability=True, random_state=42)
        clf.fit(Xtr_s, ytr)
        return clf, clf.predict(Xte_s), clf.predict_proba(Xte_s)


rows = []

for cond_name, cond_data in CONDITIONS.items():
    Xtr = cond_data["train"]
    Xva = cond_data["val"]
    Xte = cond_data["test"]
    n_feat = cond_data["n_feat"]

    for model_name in MODEL_NAMES:
        bp = best_params[cond_name][model_name]
        print(f"  [{model_name:12s}] {cond_name:40s}...", end=" ", flush=True)
        try:
            clf, y_pred_te, y_proba_te = train_and_predict(
                model_name, bp, Xtr, Xva, Xte, ytr, yva)
            m = eval_metrics(yte, y_proba_te, y_pred_te)
            print(f"AUC={m['AUC']:.3f}  BalAcc={m['BalAcc']:.3f}  "
                  f"SensW={m['Sens_W']:.3f}  F1={m['F1_macro']:.3f}")
            rows.append({
                "condition": cond_name, "n_features": n_feat,
                "model": model_name,
                "best_params": str(bp),
                "val_BalAcc": round(best_baccs[cond_name][model_name], 4),
                **{k: round(v, 4) for k, v in m.items()},
            })
        except Exception as e:
            print(f"FAILED: {e}")
            rows.append({
                "condition": cond_name, "n_features": n_feat,
                "model": model_name, "best_params": str(bp),
                "val_BalAcc": None,
                "AUC": None, "BalAcc": None, "Sens_W": None, "F1_macro": None,
            })

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTPUT_DIR / "balanced_label_results.csv", index=False)

# ---------------------------------------------------------------------------
# Confusion matrices for 39-feature condition
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CONFUSION MATRICES — 39-feature condition (test set)")
print("=" * 80)

Xtr_39 = CONDITIONS["base + behavioral lag + pmcesd (39)"]["train"]
Xva_39 = CONDITIONS["base + behavioral lag + pmcesd (39)"]["val"]
Xte_39 = CONDITIONS["base + behavioral lag + pmcesd (39)"]["test"]
cond39 = "base + behavioral lag + pmcesd (39)"

for model_name in MODEL_NAMES:
    bp = best_params[cond39][model_name]
    clf, y_pred_te, y_proba_te = train_and_predict(
        model_name, bp, Xtr_39, Xva_39, Xte_39, ytr, yva)
    print(f"\n--- {model_name} (best params: {bp}) ---")
    print(classification_report(yte, y_pred_te, target_names=label_names,
                                 zero_division=0, digits=3))
    cm = confusion_matrix(yte, y_pred_te, labels=[0, 1, 2])
    print(f"  {'':12s}  " + "  ".join(f"pred_{n:8s}" for n in label_names))
    for i, row_cm in enumerate(cm):
        print(f"  true_{label_names[i]:8s}  " + "  ".join(f"{v:12d}" for v in row_cm))

# ---------------------------------------------------------------------------
# Generate markdown report
# ---------------------------------------------------------------------------
print(f"\n[6] Writing report to {REPORT_PATH}...")

lines = []
lines.append("# Balanced Tercile Labeling — Classification Results\n")
lines.append("")
lines.append("## Label Definition\n")
lines.append("")
lines.append("Observations are **rank-ordered by CESD delta** and assigned to 3 equal-sized bins:")
lines.append("the bottom third (largest decreases) = improving, middle third = stable, top third (largest increases) = worsening.")
lines.append("Ties at bin boundaries are broken randomly (seeded for reproducibility).")
lines.append("")
lines.append(f"Training-set boundary midpoints: lo = {tercile_lo:.2f}, hi = {tercile_hi:.2f}")
lines.append("")
lines.append("| Class | Assignment rule |")
lines.append("|---|---|")
lines.append("| Improving (0) | Bottom third of ranked CESD deltas |")
lines.append("| Stable (1) | Middle third |")
lines.append("| Worsening (2) | Top third |")
lines.append("")
lines.append("### Label Distribution\n")
lines.append("")
lines.append("| Split | N obs | Improving | Stable | Worsening |")
lines.append("|---|---|---|---|---|")
for split_name, labels in [("Train", ytr), ("Val", yva), ("Test", yte)]:
    n = len(labels)
    parts = []
    for i in range(3):
        cnt = int((labels == i).sum())
        pct = cnt / n * 100
        parts.append(f"{cnt} ({pct:.0f}%)")
    lines.append(f"| {split_name} | {n} | {' | '.join(parts)} |")

lines.append("")
lines.append("---\n")
lines.append("")
lines.append("## Models — Grid-Searched Per Feature Condition\n")
lines.append("")
lines.append("Hyperparameters were **grid-searched on the validation set per model × feature condition**, using the same search spaces as the original sev_crossing experiments. This ensures each condition gets its own optimal params.\n")
lines.append("")
lines.append("### Best hyperparameters per condition\n")
lines.append("")
for cond_name in CONDITIONS:
    lines.append(f"**{cond_name}:**\n")
    lines.append("")
    lines.append("| Model | Best Params | Val BalAcc |")
    lines.append("|---|---|---|")
    for m in MODEL_NAMES:
        bp = best_params[cond_name][m]
        param_str = ", ".join(f"{k}={v}" for k, v in bp.items())
        vb = best_baccs[cond_name][m]
        lines.append(f"| {m} | {param_str} | {vb:.4f} |")
    lines.append("")
lines.append("")
lines.append("---\n")
lines.append("")
lines.append("## Results — Test Set\n")
lines.append("")

# Pivot table by condition
for cond_name in CONDITIONS:
    cond_rows = results_df[results_df["condition"] == cond_name]
    n_feat = cond_rows["n_features"].iloc[0]
    lines.append(f"### {cond_name} ({n_feat} features)\n")
    lines.append("")
    lines.append("| Model | AUC | BalAcc | Sens-W | F1-macro |")
    lines.append("|---|---|---|---|---|")
    for _, r in cond_rows.iterrows():
        auc_str = f"{r['AUC']:.3f}" if pd.notna(r['AUC']) else "—"
        ba_str = f"{r['BalAcc']:.3f}" if pd.notna(r['BalAcc']) else "—"
        sw_str = f"{r['Sens_W']:.3f}" if pd.notna(r['Sens_W']) else "—"
        f1_str = f"{r['F1_macro']:.3f}" if pd.notna(r['F1_macro']) else "—"
        lines.append(f"| {r['model']} | {auc_str} | {ba_str} | {sw_str} | {f1_str} |")
    lines.append("")

lines.append("---\n")
lines.append("")
lines.append("## Comparison: Balanced Tercile vs Sev_Crossing (39 features, both tuned)\n")
lines.append("")
lines.append("Both label types use hyperparameters independently tuned on their own validation set.\n")
lines.append("")
lines.append("| Model | Label | AUC | BalAcc | Sens-W |")
lines.append("|---|---|---|---|---|")
for m, sev_auc, sev_ba, sev_sw in [
    ("XGBoost", "0.915", "0.792", "0.730"),
    ("ElasticNet", "0.820", "0.668", "0.676"),
    ("LightGBM", "0.870", "0.740", "—"),
    ("SVM", "0.825", "0.669", "—"),
]:
    lines.append(f"| {m} | sev_crossing | {sev_auc} | {sev_ba} | {sev_sw} |")
    r = results_df[(results_df["model"] == m) & (results_df["condition"] == cond39)].iloc[0]
    lines.append(f"| {m} | balanced_tercile | {r['AUC']:.3f} | {r['BalAcc']:.3f} | {r['Sens_W']:.3f} |")
lines.append("")

# Write report
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(lines))

print(f"  Report saved to {REPORT_PATH}")
print(f"  CSV saved to {OUTPUT_DIR / 'balanced_label_results.csv'}")

# ---------------------------------------------------------------------------
# Save label info
# ---------------------------------------------------------------------------
label_info = {
    "label_type": "balanced_tercile",
    "tercile_lo": tercile_lo,
    "tercile_hi": tercile_hi,
    "train_improving": int((ytr == 0).sum()),
    "train_stable": int((ytr == 1).sum()),
    "train_worsening": int((ytr == 2).sum()),
    "best_params": {cond: {model: {str(pk): pv for pk, pv in params.items()}
                          for model, params in models.items()}
                    for cond, models in best_params.items()},
}
with open(OUTPUT_DIR / "label_info.yaml", "w") as f:
    yaml.dump(label_info, f, default_flow_style=False)

np.save(OUTPUT_DIR / "y_true_train.npy", ytr)
np.save(OUTPUT_DIR / "y_true_val.npy", yva)
np.save(OUTPUT_DIR / "y_true_test.npy", yte)

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
