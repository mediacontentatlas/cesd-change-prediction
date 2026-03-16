"""Run feature ablation with grid-searched params for ALL 5 conditions.

Produces val + test metrics for the ablation tables in sections 4.3, 4.4, 13.2.
The 5 conditions are:
  1. prior_cesd only (1 feature)           — "clinician baseline"
  2. behav + lag, no prior_cesd (37 feat)  — "behavioral digital phenotype"
  3. base (21 features, no lag)            — "behaviors + prior_cesd, no lag"
  4. base + behavioral lag (38 feat)       — "combined"
  5. base + behavioral lag + pmcesd (39)   — "combined + trait"

For conditions 1, 3, 4, 5: loads existing grid-searched best params.
For condition 2 (new): runs grid search on val set.

Outputs:
  - ablation_results.csv  (all metrics)
  - ablation_bootstrap.csv (bootstrap CIs for the new condition)
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

SEED = 42
N_BOOT = 1000
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("models/bootstrap_ci")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["ElasticNet", "XGBoost", "LightGBM", "SVM"]
LABEL_TYPES = ["sev_crossing", "personal_sd"]

# ======================================================================
# Load data
# ======================================================================
print("[1] Loading data...")
X_train = np.load(DATA_DIR / "X_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_val.npy")
y_test = np.load(DATA_DIR / "y_test.npy")
pid_train = np.load(DATA_DIR / "pid_train.npy")
pid_val = np.load(DATA_DIR / "pid_val.npy")
pid_test = np.load(DATA_DIR / "pid_test.npy")

with open("models/feature_names.pkl", "rb") as f:
    base_feature_names = pickle.load(f)

prior_train = X_train[:, 0]
prior_val = X_val[:, 0]
prior_test = X_test[:, 0]

print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"  Base features: {len(base_feature_names)}")

# ======================================================================
# Build lag features + person_mean_cesd (identical to bootstrap_ci.py)
# ======================================================================
print("\n[2] Building lag features...")

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

# Exclude static + clinical lags
drop_lags = ["lag_age", "lag_gender_mode_1", "lag_gender_mode_2",
             "lag_prior_cesd", "lag_cesd_delta"]
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
    person_mean_cesd = {}
    for pid in np.unique(pid_train):
        person_mean_cesd[pid] = float(prior_train[pid_train == pid].mean())

pop_mean = np.mean(list(person_mean_cesd.values()))


def get_pmcesd(pids):
    return np.array([
        person_mean_cesd.get(int(p) if hasattr(p, "item") else p, pop_mean)
        for p in pids
    ]).reshape(-1, 1)


# Assemble feature conditions
X38_tr = np.hstack([X_train, lag_tr])
X38_va = np.hstack([X_val, lag_va])
X38_te = np.hstack([X_test, lag_te])
X39_tr = np.hstack([X38_tr, get_pmcesd(pid_train)])
X39_va = np.hstack([X38_va, get_pmcesd(pid_val)])
X39_te = np.hstack([X38_te, get_pmcesd(pid_test)])

# NEW: behavioral features + lag, NO prior_cesd (37 features)
# Remove column 0 (prior_cesd) from the 38-feature set
behav_idx = list(range(1, X38_tr.shape[1]))  # all columns except 0
X37_tr = X38_tr[:, behav_idx]
X37_va = X38_va[:, behav_idx]
X37_te = X38_te[:, behav_idx]

prior_idx = [0]
base_idx = list(range(X_train.shape[1]))

CONDITIONS = {
    "prior_cesd only": {
        "train": X_train[:, prior_idx], "val": X_val[:, prior_idx],
        "test": X_test[:, prior_idx],
    },
    "behav + lag, no prior_cesd (37)": {
        "train": X37_tr, "val": X37_va, "test": X37_te,
    },
    "base (21)": {
        "train": X_train[:, base_idx], "val": X_val[:, base_idx],
        "test": X_test[:, base_idx],
    },
    "base + behavioral lag (38)": {
        "train": X38_tr, "val": X38_va, "test": X38_te,
    },
    "base + behavioral lag + pmcesd (39)": {
        "train": X39_tr, "val": X39_va, "test": X39_te,
    },
}

print(f"  Feature conditions: {[(k, v['train'].shape[1]) for k, v in CONDITIONS.items()]}")

# ======================================================================
# Build labels
# ======================================================================
print("\n[3] Building labels...")

CONFIG = yaml.safe_load(open("configs/models/classifier.yaml"))
SEV_MINOR = CONFIG["label"].get("sev_minor", 16)
SEV_MOD = CONFIG["label"].get("sev_moderate", 24)


def severity(cesd):
    return np.where(cesd < SEV_MINOR, 0, np.where(cesd < SEV_MOD, 1, 2))


def make_sev_crossing_labels(y_delta, prior):
    sb = severity(prior)
    sa = severity(np.clip(prior + y_delta, 0, 60))
    return np.where(sa < sb, 0, np.where(sa > sb, 2, 1))


pop_sd = float(y_train.std())
K_MULT = 1.0
person_sd = {}
for pid in np.unique(pid_train):
    vals = y_train[pid_train == pid]
    person_sd[pid] = max(float(vals.std(ddof=1)) if len(vals) > 1 else pop_sd, 3.0)


def make_personal_sd_labels(y_delta, pids):
    labels = np.ones(len(y_delta), dtype=int)
    for i, (d, p) in enumerate(zip(y_delta, pids)):
        sd = person_sd.get(p, pop_sd)
        thresh = K_MULT * sd
        if d > thresh:
            labels[i] = 2
        elif d < -thresh:
            labels[i] = 0
    return labels


LABELS = {}
for ltype in LABEL_TYPES:
    if ltype == "sev_crossing":
        ytr = make_sev_crossing_labels(y_train, prior_train)
        yva = make_sev_crossing_labels(y_val, prior_val)
        yte = make_sev_crossing_labels(y_test, prior_test)
    elif ltype == "personal_sd":
        ytr = make_personal_sd_labels(y_train, pid_train)
        yva = make_personal_sd_labels(y_val, pid_val)
        yte = make_personal_sd_labels(y_test, pid_test)
    LABELS[ltype] = {"train": ytr, "val": yva, "test": yte}
    dist = " | ".join(f"{['imp','stb','wrs'][i]}={int((yte==i).sum())}" for i in range(3))
    print(f"  {ltype}: test={dist}")

# ======================================================================
# Load existing best params + grid search for new condition
# ======================================================================
print("\n[4] Loading best params and grid-searching new condition...")

BEST_PARAMS = {}

# Load existing params
for ltype, yaml_path in [
    ("personal_sd", Path("models/classifier_personal_sd_all/best_params.yaml")),
    ("sev_crossing", OUTPUT_DIR / "sev_crossing_best_params.yaml"),
]:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    BEST_PARAMS[ltype] = raw
    print(f"  Loaded {ltype} params from {yaml_path}")

# Grid search for the new "behav + lag, no prior_cesd (37)" condition
print("\n  Grid searching 'behav + lag, no prior_cesd (37)'...")

# ElasticNet grid
EN_GRID = [
    {"C": c, "l1_ratio": lr}
    for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    for lr in [0.1, 0.5, 0.9, 0.99]
]

# XGBoost grid
XGB_GRID = [
    {"max_depth": md, "learning_rate": lr, "n_estimators": ne,
     "subsample": ss, "colsample_bytree": cs, "min_child_weight": mcw}
    for md in [3, 5, 7]
    for lr in [0.01, 0.05, 0.1]
    for ne in [50, 100, 200]
    for ss in [0.8, 1.0]
    for cs in [0.8, 1.0]
    for mcw in [1, 3, 5]
]

# LightGBM grid
LGBM_GRID = [
    {"max_depth": md, "learning_rate": lr, "n_estimators": ne,
     "num_leaves": nl, "subsample": ss, "colsample_bytree": cs,
     "reg_alpha": ra, "reg_lambda": rl, "min_child_samples": mcs}
    for md in [3, 5, 7]
    for lr in [0.01, 0.05, 0.1]
    for ne in [50, 100]
    for nl in [15, 31, 63]
    for ss in [0.6, 0.8, 1.0]
    for cs in [0.6, 0.8, 1.0]
    for ra in [0.0, 0.1, 1.0]
    for rl in [0.0, 0.1, 1.0]
    for mcs in [10, 20, 30]
]

# SVM grid
SVM_GRID = [
    {"C": c, "gamma": g, "kernel": k}
    for c in [0.1, 0.5, 1.0, 5.0, 10.0]
    for g in [0.0001, 0.001, 0.01, 0.1]
    for k in ["linear", "rbf"]
]

# Subsample grids for speed
rng_grid = np.random.RandomState(42)
if len(XGB_GRID) > 100:
    XGB_GRID = [XGB_GRID[i] for i in rng_grid.choice(len(XGB_GRID), 100, replace=False)]
if len(LGBM_GRID) > 100:
    LGBM_GRID = [LGBM_GRID[i] for i in rng_grid.choice(len(LGBM_GRID), 100, replace=False)]


def train_model(model_name, params, Xtr, Xva, ytr, yva, class_wt):
    """Train model, return val AUC."""
    try:
        if model_name == "ElasticNet":
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga",
                C=params["C"], l1_ratio=params["l1_ratio"],
                class_weight="balanced", max_iter=2000, random_state=42,
            )
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xva)
        elif model_name == "XGBoost":
            clf = XGBClassifier(
                **params, gamma=0, objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", use_label_encoder=False,
                random_state=42, verbosity=0,
            )
            sw = np.array([class_wt[y] for y in ytr])
            clf.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
            proba = clf.predict_proba(Xva)
        elif model_name == "LightGBM":
            clf = LGBMClassifier(
                **params, class_weight="balanced", random_state=42, verbose=-1,
            )
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)])
            proba = clf.predict_proba(Xva)
        elif model_name == "SVM":
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xva_s = scaler.transform(Xva)
            clf = SVC(**params, class_weight="balanced", probability=True, random_state=42)
            clf.fit(Xtr_s, ytr)
            proba = clf.predict_proba(Xva_s)
        return roc_auc_score(yva, proba, multi_class="ovr", average="macro")
    except Exception as e:
        return 0.0


cond_name_new = "behav + lag, no prior_cesd (37)"

for ltype in LABEL_TYPES:
    ytr = LABELS[ltype]["train"]
    yva = LABELS[ltype]["val"]
    Xtr = CONDITIONS[cond_name_new]["train"]
    Xva = CONDITIONS[cond_name_new]["val"]

    classes, counts = np.unique(ytr, return_counts=True)
    class_wt = {c: len(ytr) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}

    BEST_PARAMS[ltype][cond_name_new] = {}

    for model_name, grid in [("ElasticNet", EN_GRID), ("XGBoost", XGB_GRID),
                              ("LightGBM", LGBM_GRID), ("SVM", SVM_GRID)]:
        print(f"    [{ltype}] {model_name}: searching {len(grid)} configs...", end=" ", flush=True)
        best_auc = -1
        best_p = None
        for p in grid:
            auc = train_model(model_name, p, Xtr, Xva, ytr, yva, class_wt)
            if auc > best_auc:
                best_auc = auc
                best_p = p
        BEST_PARAMS[ltype][cond_name_new][model_name] = best_p
        print(f"best val AUC={best_auc:.3f} params={best_p}")

# ======================================================================
# Train all models, get val + test predictions
# ======================================================================
print("\n[5] Training all models...")

COND_NAMES = list(CONDITIONS.keys())


def train_and_predict_full(model_name, bp, Xtr, Xva, Xte, ytr, yva, class_wt):
    """Train model, return (y_pred_val, y_proba_val, y_pred_test, y_proba_test)."""
    if model_name == "ElasticNet":
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=bp["C"], l1_ratio=bp["l1_ratio"],
            class_weight="balanced", max_iter=2000, random_state=42,
        )
        clf.fit(Xtr, ytr)
        return (clf.predict(Xva), clf.predict_proba(Xva),
                clf.predict(Xte), clf.predict_proba(Xte))
    elif model_name == "XGBoost":
        clf = XGBClassifier(
            **bp, gamma=0, objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=42, verbosity=0,
        )
        sw = np.array([class_wt[y] for y in ytr])
        clf.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
        return (clf.predict(Xva), clf.predict_proba(Xva),
                clf.predict(Xte), clf.predict_proba(Xte))
    elif model_name == "LightGBM":
        clf = LGBMClassifier(
            **bp, class_weight="balanced", random_state=42, verbose=-1,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        return (clf.predict(Xva), clf.predict_proba(Xva),
                clf.predict(Xte), clf.predict_proba(Xte))
    elif model_name == "SVM":
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xte)
        clf = SVC(**bp, class_weight="balanced", probability=True, random_state=42)
        clf.fit(Xtr_s, ytr)
        return (clf.predict(Xva_s), clf.predict_proba(Xva_s),
                clf.predict(Xte_s), clf.predict_proba(Xte_s))


rows = []

for ltype in LABEL_TYPES:
    ytr = LABELS[ltype]["train"]
    yva = LABELS[ltype]["val"]
    yte = LABELS[ltype]["test"]

    classes, counts = np.unique(ytr, return_counts=True)
    class_wt = {c: len(ytr) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}

    for cond_name in COND_NAMES:
        Xtr = CONDITIONS[cond_name]["train"]
        Xva = CONDITIONS[cond_name]["val"]
        Xte = CONDITIONS[cond_name]["test"]

        for model_name in MODEL_NAMES:
            bp = BEST_PARAMS[ltype].get(cond_name, {}).get(model_name)
            if bp is None:
                print(f"  SKIP [{ltype}] [{model_name}] {cond_name} — no params")
                continue

            print(f"  [{ltype:18s}] [{model_name:12s}] {cond_name:40s}...", end=" ", flush=True)
            yp_va, ypr_va, yp_te, ypr_te = train_and_predict_full(
                model_name, bp, Xtr, Xva, Xte, ytr, yva, class_wt)

            # Val metrics
            try:
                val_auc = roc_auc_score(yva, ypr_va, multi_class="ovr", average="macro")
            except Exception:
                val_auc = float("nan")
            val_bacc = balanced_accuracy_score(yva, yp_va)
            n_wors_va = (yva == 2).sum()
            val_sensw = float((yp_va[yva == 2] == 2).sum()) / max(n_wors_va, 1)
            n_pred_wors_va = (yp_va == 2).sum()
            val_ppvw = float(((yp_va == 2) & (yva == 2)).sum()) / max(n_pred_wors_va, 1)

            # Test metrics
            try:
                test_auc = roc_auc_score(yte, ypr_te, multi_class="ovr", average="macro")
            except Exception:
                test_auc = float("nan")
            test_bacc = balanced_accuracy_score(yte, yp_te)
            n_wors_te = (yte == 2).sum()
            test_sensw = float((yp_te[yte == 2] == 2).sum()) / max(n_wors_te, 1)
            n_pred_wors_te = (yp_te == 2).sum()
            test_ppvw = float(((yp_te == 2) & (yte == 2)).sum()) / max(n_pred_wors_te, 1)

            # mod->sev detection (sev_crossing only)
            if ltype == "sev_crossing":
                # Find mod->sev cases in test: prior in moderate (1), next in severe (2)
                sev_prior = severity(prior_test)
                sev_next = severity(np.clip(prior_test + y_test, 0, 60))
                mod_sev_mask = (sev_prior == 1) & (sev_next == 2)
                n_mod_sev = mod_sev_mask.sum()
                if n_mod_sev > 0:
                    mod_sev_detected = (yp_te[mod_sev_mask] == 2).sum()
                    mod_sev_rate = f"{mod_sev_detected}/{n_mod_sev}"
                else:
                    mod_sev_rate = "N/A"
            else:
                mod_sev_rate = "N/A"

            print(f"val_AUC={val_auc:.3f}  test_AUC={test_auc:.3f}  test_SensW={test_sensw:.3f}")

            rows.append({
                "label": ltype,
                "condition": cond_name,
                "model": model_name,
                "n_features": CONDITIONS[cond_name]["train"].shape[1],
                "val_AUC": round(val_auc, 3),
                "val_BalAcc": round(val_bacc, 3),
                "val_SensW": round(val_sensw, 3),
                "test_AUC": round(test_auc, 3),
                "test_BalAcc": round(test_bacc, 3),
                "test_SensW": round(test_sensw, 3),
                "test_PPVW": round(test_ppvw, 3),
                "mod_sev_detection": mod_sev_rate,
            })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
print(f"\n  Saved ablation results to {OUTPUT_DIR / 'ablation_results.csv'}")

# ======================================================================
# Print summary tables for doc update
# ======================================================================
print("\n" + "=" * 80)
print("ABLATION TABLES FOR DOC UPDATE")
print("=" * 80)

for ltype in LABEL_TYPES:
    print(f"\n### {ltype}\n")
    sub = df[df["label"] == ltype]

    # Table matching section 4.3 format
    print("| Condition | N feat | EN val AUC | EN test AUC | EN test Sens-W | XGB val AUC | XGB test AUC | XGB test Sens-W |")
    print("|---|---|---|---|---|---|---|---|")
    for cond in COND_NAMES:
        cs = sub[sub["condition"] == cond]
        en = cs[cs["model"] == "ElasticNet"]
        xgb = cs[cs["model"] == "XGBoost"]
        nf = int(cs["n_features"].iloc[0]) if len(cs) > 0 else "?"
        en_va = f"{en['val_AUC'].iloc[0]:.3f}" if len(en) > 0 else "—"
        en_te = f"{en['test_AUC'].iloc[0]:.3f}" if len(en) > 0 else "—"
        en_sw = f"{en['test_SensW'].iloc[0]:.3f}" if len(en) > 0 else "—"
        xgb_va = f"{xgb['val_AUC'].iloc[0]:.3f}" if len(xgb) > 0 else "—"
        xgb_te = f"{xgb['test_AUC'].iloc[0]:.3f}" if len(xgb) > 0 else "—"
        xgb_sw = f"{xgb['test_SensW'].iloc[0]:.3f}" if len(xgb) > 0 else "—"
        print(f"| {cond} | {nf} | {en_va} | {en_te} | {en_sw} | {xgb_va} | {xgb_te} | {xgb_sw} |")

    # Full 4-model table
    print(f"\n  Full 4-model table:")
    print("| Condition | Model | val AUC | test AUC | test BalAcc | test Sens-W | test PPV-W |")
    print("|---|---|---|---|---|---|---|")
    for cond in COND_NAMES:
        for model in MODEL_NAMES:
            r = sub[(sub["condition"] == cond) & (sub["model"] == model)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            print(f"| {cond} | {model} | {r['val_AUC']:.3f} | {r['test_AUC']:.3f} | {r['test_BalAcc']:.3f} | {r['test_SensW']:.3f} | {r['test_PPVW']:.3f} |")

# Section 4.4 specific: 38 vs 39 features for XGBoost sev_crossing
print("\n### Section 4.4: person_mean_cesd effect (XGBoost, sev_crossing)")
sev = df[df["label"] == "sev_crossing"]
for cond in ["base + behavioral lag (38)", "base + behavioral lag + pmcesd (39)"]:
    r = sev[(sev["condition"] == cond) & (sev["model"] == "XGBoost")]
    if len(r) > 0:
        r = r.iloc[0]
        print(f"  {cond}: AUC={r['test_AUC']:.3f} BalAcc={r['test_BalAcc']:.3f} "
              f"SensW={r['test_SensW']:.3f} mod→sev={r['mod_sev_detection']}")

print("\nDONE")
