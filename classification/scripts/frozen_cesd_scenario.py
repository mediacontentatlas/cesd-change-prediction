"""Frozen-CES-D deployment scenario.

Simulates having only the intake (first-ever) CES-D survey available.
At evaluation time, for every val and test observation:

    prior_cesd        → person's first-ever CES-D score
    person_mean_cesd  → same first-ever CES-D score

All 37 behavioral + lag features are computed from Screenome as usual.
The same 39-feature model trained on full training data is used — no retraining
beyond what is needed for early stopping (XGBoost/LightGBM use frozen val).

Interpretation:
    AUC << 0.906 (full model)  → CES-D updates matter
    AUC ~  0.906               → behavioral trajectory carries the prediction
    AUC >  0.821 (cold start)  → behavioral features are doing real work

Outputs:
  classification/models/frozen_cesd/frozen_cesd_results.csv
  classification/reports/frozen_cesd_results.md
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ============================================================
# Paths (run from repo root)
# ============================================================
ROOT      = Path(".")
DATA_DIR  = ROOT / "data" / "processed"
LABEL_DIR = ROOT / "classification" / "labels" / "sev_crossing"
OUT_DIR   = ROOT / "classification" / "models" / "frozen_cesd"
REPORT    = ROOT / "classification" / "reports" / "frozen_cesd_results.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_N = 1000
SEED        = 42
MODEL_NAMES = ["ElasticNet", "XGBoost", "LightGBM", "SVM"]

PRIOR_CESD_IDX = 0   # column 0 in X_*
PMCESD_IDX     = -1  # last column in 39-feat matrix

# ============================================================
# Load data
# ============================================================
print("[1] Loading data...")
X_train = np.load(DATA_DIR / "X_train.npy")
X_val   = np.load(DATA_DIR / "X_val.npy")
X_test  = np.load(DATA_DIR / "X_test.npy")

y_delta_train = np.load(DATA_DIR / "y_train.npy")
y_delta_val   = np.load(DATA_DIR / "y_val.npy")

pid_train = np.load(DATA_DIR / "pid_train.npy")
pid_val   = np.load(DATA_DIR / "pid_val.npy")
pid_test  = np.load(DATA_DIR / "pid_test.npy")

prior_train = X_train[:, PRIOR_CESD_IDX]
prior_val   = X_val[:, PRIOR_CESD_IDX]
prior_test  = X_test[:, PRIOR_CESD_IDX]

# sev_crossing labels
y_tr = np.load(LABEL_DIR / "y_train.npy")
y_va = np.load(LABEL_DIR / "y_val.npy")
y_te = np.load(LABEL_DIR / "y_test.npy")

print(f"  Train={len(y_tr)}  Val={len(y_va)}  Test={len(y_te)}")
print(f"  Test: imp={(y_te==0).sum()}  stb={(y_te==1).sum()}  wrs={(y_te==2).sum()}")

# ============================================================
# Load CSVs and build lag features (identical to full model)
# ============================================================
print("\n[2] Building lag features...")

with open(ROOT / "classification" / "models" / "feature_names.pkl", "rb") as f:
    feat_names = pickle.load(f)

train_df = pd.read_csv(DATA_DIR / "train_scaled.csv")
val_df   = pd.read_csv(DATA_DIR / "val_scaled.csv")
test_df  = pd.read_csv(DATA_DIR / "test_scaled.csv")

all_df = (
    pd.concat([train_df, val_df, test_df], ignore_index=True)
    .sort_values(["pid", "period_number"])
    .reset_index(drop=True)
)

lag_cols_all = [f"lag_{c}" for c in feat_names] + ["lag_cesd_delta"]
for col in feat_names:
    if col in all_df.columns:
        all_df[f"lag_{col}"] = all_df.groupby("pid")[col].shift(1)
    else:
        all_df[f"lag_{col}"] = 0.0
all_df["lag_cesd_delta"] = all_df.groupby("pid")["target_cesd_delta"].shift(1)
all_df[lag_cols_all] = all_df[lag_cols_all].fillna(0)

drop_lags = {"lag_age", "lag_gender_mode_1", "lag_gender_mode_2",
             "lag_prior_cesd", "lag_cesd_delta"}
keep_idx  = [i for i, c in enumerate(lag_cols_all) if c not in drop_lags]

lag_tr = all_df[all_df["split"] == "train"][lag_cols_all].values[:, keep_idx]
lag_va = all_df[all_df["split"] == "val"][lag_cols_all].values[:, keep_idx]
lag_te = all_df[all_df["split"] == "test"][lag_cols_all].values[:, keep_idx]
print(f"  Behavioral lag features: {lag_tr.shape[1]}")

# ============================================================
# person_mean_cesd (from training prior_cesd — full model values)
# ============================================================
def _pid_key(p):
    return int(p) if hasattr(p, "item") else p

person_mean_cesd: dict = {}
for pid in np.unique(pid_train):
    person_mean_cesd[_pid_key(pid)] = float(prior_train[pid_train == pid].mean())

pop_mean = float(np.mean(list(person_mean_cesd.values())))

def get_pmcesd(pids):
    return np.array(
        [person_mean_cesd.get(_pid_key(p), pop_mean) for p in pids]
    ).reshape(-1, 1)

# ============================================================
# First-ever CES-D per person
# ============================================================
print("\n[3] Computing first-ever CES-D per person...")

first_cesd: dict = (
    all_df.sort_values(["pid", "period_number"])
    .groupby("pid")["prior_cesd"]
    .first()
    .to_dict()
)

def get_first_cesd(pids):
    return np.array(
        [first_cesd.get(_pid_key(p), pop_mean) for p in pids]
    )

first_va = get_first_cesd(pid_val)
first_te = get_first_cesd(pid_test)

print(f"  First CES-D: min={min(first_cesd.values()):.1f}  "
      f"mean={np.mean(list(first_cesd.values())):.1f}  "
      f"max={max(first_cesd.values()):.1f}")

# ============================================================
# Build 39-feature matrices
# ============================================================
print("\n[4] Building feature matrices...")

# Full model matrices (unchanged training)
X39_tr = np.hstack([X_train, lag_tr, get_pmcesd(pid_train)])

# Full val/test (baseline reference)
X39_va_full = np.hstack([X_val, lag_va, get_pmcesd(pid_val)])
X39_te_full = np.hstack([X_test, lag_te, get_pmcesd(pid_test)])

# Frozen-CES-D val: replace prior_cesd and pmcesd with first_cesd
X39_va_frozen = X39_va_full.copy()
X39_va_frozen[:, PRIOR_CESD_IDX] = first_va
X39_va_frozen[:, PMCESD_IDX]     = first_va

# Frozen-CES-D test: same replacement
X39_te_frozen = X39_te_full.copy()
X39_te_frozen[:, PRIOR_CESD_IDX] = first_te
X39_te_frozen[:, PMCESD_IDX]     = first_te

print(f"  Full model matrix: {X39_tr.shape[1]} features")
print(f"  prior_cesd frozen: val mean {first_va.mean():.2f} "
      f"(vs live {prior_val.mean():.2f})")
print(f"  prior_cesd frozen: test mean {first_te.mean():.2f} "
      f"(vs live {prior_test.mean():.2f})")

# ============================================================
# Load best params (39-feat condition from bootstrap analysis)
# ============================================================
print("\n[5] Loading best params...")
with open(ROOT / "classification" / "models" / "bootstrap_ci" /
          "sev_crossing_best_params.yaml") as f:
    all_params = yaml.safe_load(f)
params_39 = all_params["base + behavioral lag + pmcesd (39)"]
print(f"  Loaded: {list(params_39.keys())}")

# Class weights from training labels
cls_, cnt_ = np.unique(y_tr, return_counts=True)
class_wt = {c: len(y_tr) / (len(cls_) * cnt) for c, cnt in zip(cls_, cnt_)}

# ============================================================
# Model helpers
# ============================================================

def make_model(name, bp):
    if name == "ElasticNet":
        return LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=bp["C"], l1_ratio=bp["l1_ratio"],
            class_weight="balanced", max_iter=2000, random_state=SEED)
    elif name == "XGBoost":
        return XGBClassifier(
            **bp, gamma=0, objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=SEED, verbosity=0)
    elif name == "LightGBM":
        return LGBMClassifier(**bp, class_weight="balanced",
                              random_state=SEED, verbose=-1)
    elif name == "SVM":
        return SVC(**bp, class_weight="balanced", probability=True,
                   random_state=SEED)


def train_predict(name, bp, Xtr, Xva, Xte, ytr, yva):
    clf = make_model(name, bp)
    if name == "SVM":
        scaler = StandardScaler()
        clf.fit(scaler.fit_transform(Xtr), ytr)
        return clf.predict(scaler.transform(Xte)), clf.predict_proba(scaler.transform(Xte))
    elif name == "XGBoost":
        sw = np.array([class_wt[y] for y in ytr])
        clf.fit(Xtr, ytr, sample_weight=sw,
                eval_set=[(Xva, yva)], verbose=False)
    elif name == "LightGBM":
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)])
    else:
        clf.fit(Xtr, ytr)
    return clf.predict(Xte), clf.predict_proba(Xte)


def evaluate(y_true, y_pred, y_proba):
    bacc  = balanced_accuracy_score(y_true, y_pred)
    f1m   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    n_w   = (y_true == 2).sum()
    n_pw  = (y_pred == 2).sum()
    sens_w = float((y_pred[y_true == 2] == 2).sum()) / max(n_w, 1)
    ppv_w  = float(((y_pred == 2) & (y_true == 2)).sum()) / max(n_pw, 1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return {"AUC": round(auc, 3), "BalAcc": round(bacc, 3),
            "F1macro": round(f1m, 3), "SensW": round(sens_w, 3),
            "PPVW": round(ppv_w, 3), "confusion_matrix": cm}


def bootstrap_ci(y_true, y_pred, y_proba, n=BOOTSTRAP_N, seed=SEED):
    rng = np.random.RandomState(seed)
    metrics = {k: [] for k in ["AUC", "BalAcc", "SensW", "PPVW"]}
    for _ in range(n):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        metrics["BalAcc"].append(balanced_accuracy_score(yt, yp))
        nw  = (yt == 2).sum()
        npw = (yp == 2).sum()
        metrics["SensW"].append(float((yp[yt==2]==2).sum()) / max(nw, 1) if nw > 0 else np.nan)
        metrics["PPVW"].append(float(((yp==2)&(yt==2)).sum()) / max(npw, 1) if npw > 0 else np.nan)
        try:
            metrics["AUC"].append(roc_auc_score(yt, ypr, multi_class="ovr", average="macro"))
        except Exception:
            metrics["AUC"].append(np.nan)
    ci = {}
    for k, vals in metrics.items():
        v = np.array([x for x in vals if not np.isnan(x)])
        if len(v) >= 10:
            ci[f"{k}_lo"] = round(float(np.percentile(v, 2.5)), 3)
            ci[f"{k}_hi"] = round(float(np.percentile(v, 97.5)), 3)
        else:
            ci[f"{k}_lo"] = ci[f"{k}_hi"] = float("nan")
    return ci

# ============================================================
# Run scenario
# ============================================================
print("\n[6] Training and evaluating — Frozen-CES-D scenario...")

results  = []
preds    = {}
cms      = {}

for name in MODEL_NAMES:
    bp = params_39[name]
    print(f"  {name}...", end=" ", flush=True)

    # Train on full training data; val and test both have frozen CES-D
    y_pred, y_proba = train_predict(
        name, bp, X39_tr, X39_va_frozen, X39_te_frozen, y_tr, y_va)

    m = evaluate(y_te, y_pred, y_proba)
    ci = bootstrap_ci(y_te, y_pred, y_proba)

    preds[name] = (y_pred, y_proba)
    cms[name]   = m["confusion_matrix"]

    row = {k: v for k, v in m.items() if k != "confusion_matrix"}
    row.update(ci)
    row["model"] = name
    results.append(row)

    print(f"AUC={m['AUC']:.3f} [{ci['AUC_lo']:.3f}, {ci['AUC_hi']:.3f}]  "
          f"BalAcc={m['BalAcc']:.3f}  SensW={m['SensW']:.3f}  PPVW={m['PPVW']:.3f}")

# ============================================================
# Save CSV
# ============================================================
print("\n[7] Saving CSV...")
df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "frozen_cesd_results.csv", index=False)
print(f"  {OUT_DIR / 'frozen_cesd_results.csv'}")

# ============================================================
# Markdown report
# ============================================================
print(f"[8] Writing report...")

# Reference values for comparison table
ref = {
    "B4 Regression to Mean (baseline)": {"AUC": "0.750", "BalAcc": "0.674", "SensW": "0.541", "PPVW": "0.408"},
    "Cold start (leave-group-out CV)":   {"AUC": "0.821", "BalAcc": "0.720", "SensW": "0.569", "PPVW": "—"},
    "Full model (39-feat)":              {"AUC": "0.906", "BalAcc": "0.834", "SensW": "0.838", "PPVW": "0.356"},
}

def fmt(val, lo, hi):
    if np.isnan(lo):
        return f"{val:.3f}"
    return f"{val:.3f} [{lo:.3f}, {hi:.3f}]"

lines = [
    "# Frozen-CES-D Deployment Scenario — sev_crossing\n",
    "",
    "## Setup",
    "",
    "Same trained model as the full 39-feature model (best params from bootstrap analysis).",
    "At evaluation time, for **every val and test observation**:",
    "",
    "- `prior_cesd` → person's **first-ever CES-D score** (from their full record)",
    "- `person_mean_cesd` → same first-ever CES-D score",
    "- All 37 behavioral + lag features: **unchanged**, computed from Screenome as usual",
    "",
    "This simulates deployment where the CES-D was only administered once (intake)",
    "and never updated, while passive sensing continues.",
    "",
    "## Results\n",
    "",
    "| Model | AUC [95% CI] | BalAcc [95% CI] | F1-macro [95% CI] | Sens-W [95% CI] | PPV-W [95% CI] |",
    "|---|---|---|---|---|---|",
]

for r in results:
    lines.append(
        f"| {r['model']} "
        f"| {fmt(r['AUC'], r['AUC_lo'], r['AUC_hi'])} "
        f"| {fmt(r['BalAcc'], r['BalAcc_lo'], r['BalAcc_hi'])} "
        f"| {fmt(r['F1macro'], float('nan'), float('nan'))} "
        f"| {fmt(r['SensW'], r['SensW_lo'], r['SensW_hi'])} "
        f"| {fmt(r['PPVW'], r['PPVW_lo'], r['PPVW_hi'])} |"
    )

lines += [
    "",
    "## Comparison to Reference Scenarios (XGBoost)\n",
    "",
    "| Scenario | AUC | BalAcc | Sens-W | PPV-W |",
    "|---|---|---|---|---|",
]

for sc, vals in ref.items():
    lines.append(f"| {sc} | {vals['AUC']} | {vals['BalAcc']} | {vals['SensW']} | {vals['PPVW']} |")

# Frozen XGBoost row
r_xgb = next(r for r in results if r["model"] == "XGBoost")
lines.append(
    f"| **Frozen CES-D (this scenario)** "
    f"| **{r_xgb['AUC']:.3f}** | **{r_xgb['BalAcc']:.3f}** "
    f"| **{r_xgb['SensW']:.3f}** | **{r_xgb['PPVW']:.3f}** |"
)

lines += [
    "",
    "## Confusion Matrices\n",
    "",
]

for name in MODEL_NAMES:
    cm = cms[name]
    lines += [
        f"**{name}**",
        "",
        "```",
        f"                pred: imp   pred: stb   pred: wrs",
        f"  true: imp ({int((y_te==0).sum()):3d})     {cm[0,0]:5d}       {cm[0,1]:5d}       {cm[0,2]:5d}",
        f"  true: stb ({int((y_te==1).sum()):3d})     {cm[1,0]:5d}       {cm[1,1]:5d}       {cm[1,2]:5d}",
        f"  true: wrs ({int((y_te==2).sum()):3d})     {cm[2,0]:5d}       {cm[2,1]:5d}       {cm[2,2]:5d}",
        "```",
        "",
    ]

lines += [
    "## Interpretation\n",
    "",
    "- **AUC > 0.821 (cold start)**: behavioral trajectory contributes beyond person identity",
    "- **AUC ≈ 0.906 (full model)**: frozen CES-D barely matters; Screenome carries prediction",
    "- **AUC closer to 0.750 (B4 baseline)**: CES-D updates are essential; stale anchor hurts",
    "",
    f"Frozen CES-D mean (val): {first_va.mean():.2f}  vs  live prior_cesd mean: {prior_val.mean():.2f}",
    f"Frozen CES-D mean (test): {first_te.mean():.2f}  vs  live prior_cesd mean: {prior_test.mean():.2f}",
]

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines))
print(f"  {REPORT}")

print("\n" + "=" * 70)
print("FROZEN-CES-D SCENARIO COMPLETE")
print("=" * 70)
