"""Bootstrap confidence intervals for all label × model × feature condition.

For each combination:
  1. Train model with best params (loaded from YAML or grid-searched for sev_crossing)
  2. Get test-set predictions
  3. Percentile bootstrap (1000 resamples) → 95% CIs for AUC, BalAcc, Sens-W, F1-macro
  4. Paired bootstrap test between adjacent feature conditions → p-values

Outputs:
  - reports/bootstrap_ci_results.md  (main report)
  - models/bootstrap_ci/bootstrap_results.csv  (raw results)
  - models/bootstrap_ci/paired_tests.csv  (significance of additive value)
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

N_BOOT = 1000
SEED = 42
ALPHA = 0.05  # 95% CI

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("models/bootstrap_ci")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("reports/bootstrap_ci_results.md")

MODEL_NAMES = ["ElasticNet", "XGBoost", "LightGBM", "SVM"]
LABEL_TYPES = ["sev_crossing", "personal_sd", "balanced_tercile"]
COND_NAMES = [
    "prior_cesd only",
    "base (21)",
    "base + behavioral lag (38)",
    "base + behavioral lag + pmcesd (39)",
]

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

# ======================================================================
# Build lag features + person_mean_cesd (same as experiment scripts)
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

prior_idx = [0]
base_idx = list(range(X_train.shape[1]))

CONDITIONS = {
    "prior_cesd only": {
        "train": X_train[:, prior_idx], "val": X_val[:, prior_idx],
        "test": X_test[:, prior_idx],
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

print(f"  Feature conditions: {[f'{k} ({v["train"].shape[1]})' for k, v in CONDITIONS.items()]}")
print(f"  Lag features: {len(lag_cols)} behavioral")

# ======================================================================
# Build labels for all 3 label types
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


# Personal SD
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


def make_balanced_tercile_labels(y_delta, rng=None):
    n = len(y_delta)
    n_per = n // 3
    if rng is None:
        rng = np.random.RandomState(42)
    order = np.lexsort((rng.random(n), y_delta))
    labels = np.empty(n, dtype=int)
    labels[order[:n_per]] = 0
    labels[order[n_per:2 * n_per]] = 1
    labels[order[2 * n_per:]] = 2
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
    elif ltype == "balanced_tercile":
        ytr = make_balanced_tercile_labels(y_train, rng=np.random.RandomState(42))
        yva = make_balanced_tercile_labels(y_val, rng=np.random.RandomState(42))
        yte = make_balanced_tercile_labels(y_test, rng=np.random.RandomState(42))
    LABELS[ltype] = {"train": ytr, "val": yva, "test": yte}
    dist = " | ".join(f"{['imp','stb','wrs'][i]}={int((yte==i).sum())}" for i in range(3))
    print(f"  {ltype}: test={dist}")

# ======================================================================
# Load or compute best params
# ======================================================================
print("\n[4] Loading / computing best params...")

# Load saved params for all three label types
BEST_PARAMS = {}

for ltype, yaml_path in [
    ("personal_sd", Path("models/classifier_personal_sd_all/best_params.yaml")),
    ("balanced_tercile", Path("models/classifier_balanced/best_params.yaml")),
    ("sev_crossing", OUTPUT_DIR / "sev_crossing_best_params.yaml"),
]:
    with open(yaml_path) as f:
        BEST_PARAMS[ltype] = yaml.safe_load(f)
    print(f"  Loaded {ltype} params from {yaml_path}")

# ======================================================================
# Train all models and get test predictions
# ======================================================================
print("\n[5] Training all models and getting test predictions...")


def train_and_predict(model_name, bp, Xtr, Xva, Xte, ytr, yva, class_wt):
    """Train model with given best params, return (y_pred, y_proba) on test."""
    if model_name == "ElasticNet":
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=bp["C"], l1_ratio=bp["l1_ratio"],
            class_weight="balanced", max_iter=2000, random_state=42,
        )
        clf.fit(Xtr, ytr)
        return clf.predict(Xte), clf.predict_proba(Xte)

    elif model_name == "XGBoost":
        clf = XGBClassifier(
            **bp, gamma=0, objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=42, verbosity=0,
        )
        sw = np.array([class_wt[y] for y in ytr])
        clf.fit(Xtr, ytr, sample_weight=sw,
                eval_set=[(Xva, yva)], verbose=False)
        return clf.predict(Xte), clf.predict_proba(Xte)

    elif model_name == "LightGBM":
        clf = LGBMClassifier(
            **bp, class_weight="balanced", random_state=42, verbose=-1,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        return clf.predict(Xte), clf.predict_proba(Xte)

    elif model_name == "SVM":
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        clf = SVC(**bp, class_weight="balanced", probability=True, random_state=42)
        clf.fit(Xtr_s, ytr)
        return clf.predict(Xte_s), clf.predict_proba(Xte_s)


# Store predictions: predictions[label][cond][model] = (y_pred, y_proba)
predictions = {}

for ltype in LABEL_TYPES:
    predictions[ltype] = {}
    ytr_l = LABELS[ltype]["train"]
    yva_l = LABELS[ltype]["val"]
    yte_l = LABELS[ltype]["test"]

    # Compute class weights for XGBoost
    classes_l, counts_l = np.unique(ytr_l, return_counts=True)
    n_total_l = len(ytr_l)
    class_wt_l = {c: n_total_l / (len(classes_l) * cnt)
                  for c, cnt in zip(classes_l, counts_l)}

    for cond_name in COND_NAMES:
        predictions[ltype][cond_name] = {}
        Xtr = CONDITIONS[cond_name]["train"]
        Xva = CONDITIONS[cond_name]["val"]
        Xte = CONDITIONS[cond_name]["test"]

        for model_name in MODEL_NAMES:
            bp = BEST_PARAMS[ltype][cond_name][model_name]
            print(f"  [{ltype:18s}] [{model_name:12s}] {cond_name:40s}...",
                  end=" ", flush=True)
            y_pred, y_proba = train_and_predict(
                model_name, bp, Xtr, Xva, Xte, ytr_l, yva_l, class_wt_l)
            predictions[ltype][cond_name][model_name] = (y_pred, y_proba)
            # Quick point estimate
            bacc = balanced_accuracy_score(yte_l, y_pred)
            try:
                auc = roc_auc_score(yte_l, y_proba, multi_class="ovr", average="macro")
            except Exception:
                auc = float("nan")
            print(f"AUC={auc:.3f}  BalAcc={bacc:.3f}")

# ======================================================================
# Bootstrap CIs
# ======================================================================
print(f"\n[6] Bootstrap CIs ({N_BOOT} resamples)...")

rng = np.random.RandomState(SEED)
n_test = len(y_test)
boot_indices = rng.choice(n_test, size=(N_BOOT, n_test), replace=True)

ci_rows = []
cm_rows = []

for ltype in LABEL_TYPES:
    yte_l = LABELS[ltype]["test"]

    for cond_name in COND_NAMES:
        for model_name in MODEL_NAMES:
            y_pred, y_proba = predictions[ltype][cond_name][model_name]

            boot_aucs = []
            boot_baccs = []
            boot_sensw = []
            boot_ppvw = []
            boot_f1m = []

            for b in range(N_BOOT):
                idx = boot_indices[b]
                yt_b = yte_l[idx]
                yp_b = y_pred[idx]
                ypr_b = y_proba[idx]

                # Skip degenerate resamples (missing a class)
                if len(np.unique(yt_b)) < 3:
                    continue

                boot_baccs.append(balanced_accuracy_score(yt_b, yp_b))
                boot_f1m.append(f1_score(yt_b, yp_b, average="macro"))
                try:
                    boot_aucs.append(
                        roc_auc_score(yt_b, ypr_b, multi_class="ovr", average="macro"))
                except Exception:
                    pass

                n_wors = (yt_b == 2).sum()
                if n_wors > 0:
                    boot_sensw.append(
                        float((yp_b[yt_b == 2] == 2).sum()) / n_wors)

                n_pred_wors = (yp_b == 2).sum()
                if n_pred_wors > 0:
                    boot_ppvw.append(
                        float(((yp_b == 2) & (yt_b == 2)).sum()) / n_pred_wors)

            # Point estimates
            bacc_pt = balanced_accuracy_score(yte_l, y_pred)
            f1m_pt = f1_score(yte_l, y_pred, average="macro")
            try:
                auc_pt = roc_auc_score(yte_l, y_proba, multi_class="ovr", average="macro")
            except Exception:
                auc_pt = float("nan")
            n_wors = (yte_l == 2).sum()
            sensw_pt = float((y_pred[yte_l == 2] == 2).sum()) / max(n_wors, 1)
            n_pred_wors = (y_pred == 2).sum()
            ppvw_pt = float(((y_pred == 2) & (yte_l == 2)).sum()) / max(n_pred_wors, 1)

            # Save confusion matrix
            cm = confusion_matrix(yte_l, y_pred, labels=[0, 1, 2])
            cm_rows.append({
                "label": ltype, "condition": cond_name, "model": model_name,
                "TP_imp": int(cm[0, 0]), "FP_imp": int(cm[1, 0] + cm[2, 0]),
                "TP_stb": int(cm[1, 1]), "FP_stb": int(cm[0, 1] + cm[2, 1]),
                "TP_wrs": int(cm[2, 2]), "FP_wrs": int(cm[0, 2] + cm[1, 2]),
                "cm_00": int(cm[0, 0]), "cm_01": int(cm[0, 1]), "cm_02": int(cm[0, 2]),
                "cm_10": int(cm[1, 0]), "cm_11": int(cm[1, 1]), "cm_12": int(cm[1, 2]),
                "cm_20": int(cm[2, 0]), "cm_21": int(cm[2, 1]), "cm_22": int(cm[2, 2]),
            })

            lo, hi = ALPHA / 2 * 100, (1 - ALPHA / 2) * 100

            def ci(arr):
                if len(arr) < 10:
                    return (float("nan"), float("nan"))
                return (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))

            auc_ci = ci(boot_aucs)
            bacc_ci = ci(boot_baccs)
            sensw_ci = ci(boot_sensw)
            ppvw_ci = ci(boot_ppvw)
            f1m_ci = ci(boot_f1m)

            ci_rows.append({
                "label": ltype,
                "condition": cond_name,
                "model": model_name,
                "AUC": round(auc_pt, 3),
                "AUC_lo": round(auc_ci[0], 3),
                "AUC_hi": round(auc_ci[1], 3),
                "BalAcc": round(bacc_pt, 3),
                "BalAcc_lo": round(bacc_ci[0], 3),
                "BalAcc_hi": round(bacc_ci[1], 3),
                "F1macro": round(f1m_pt, 3),
                "F1macro_lo": round(f1m_ci[0], 3),
                "F1macro_hi": round(f1m_ci[1], 3),
                "SensW": round(sensw_pt, 3),
                "SensW_lo": round(sensw_ci[0], 3),
                "SensW_hi": round(sensw_ci[1], 3),
                "PPVW": round(ppvw_pt, 3),
                "PPVW_lo": round(ppvw_ci[0], 3),
                "PPVW_hi": round(ppvw_ci[1], 3),
            })

    print(f"  {ltype} done")

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(OUTPUT_DIR / "bootstrap_results.csv", index=False)
cm_df = pd.DataFrame(cm_rows)
cm_df.to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)
print(f"  Saved bootstrap results to {OUTPUT_DIR / 'bootstrap_results.csv'}")
print(f"  Saved confusion matrices to {OUTPUT_DIR / 'confusion_matrices.csv'}")

# ======================================================================
# Paired bootstrap tests (adjacent feature conditions)
# ======================================================================
print(f"\n[7] Paired bootstrap tests (adjacent conditions)...")

ADJACENT_PAIRS = [
    ("prior_cesd only", "base (21)"),
    ("base (21)", "base + behavioral lag (38)"),
    ("base + behavioral lag (38)", "base + behavioral lag + pmcesd (39)"),
]

paired_rows = []

for ltype in LABEL_TYPES:
    yte_l = LABELS[ltype]["test"]

    for cond_a, cond_b in ADJACENT_PAIRS:
        for model_name in MODEL_NAMES:
            _, proba_a = predictions[ltype][cond_a][model_name]
            _, proba_b = predictions[ltype][cond_b][model_name]
            pred_a = predictions[ltype][cond_a][model_name][0]
            pred_b = predictions[ltype][cond_b][model_name][0]

            boot_auc_diff = []
            boot_bacc_diff = []
            boot_f1m_diff = []
            boot_sensw_diff = []
            boot_ppvw_diff = []

            for b in range(N_BOOT):
                idx = boot_indices[b]
                yt_b = yte_l[idx]

                if len(np.unique(yt_b)) < 3:
                    continue

                bacc_a = balanced_accuracy_score(yt_b, pred_a[idx])
                bacc_b = balanced_accuracy_score(yt_b, pred_b[idx])
                boot_bacc_diff.append(bacc_b - bacc_a)

                f1m_a = f1_score(yt_b, pred_a[idx], average="macro")
                f1m_b = f1_score(yt_b, pred_b[idx], average="macro")
                boot_f1m_diff.append(f1m_b - f1m_a)

                # Sens-W (worsening class = 2)
                n_wors = (yt_b == 2).sum()
                if n_wors > 0:
                    sw_a = float((pred_a[idx][yt_b == 2] == 2).sum()) / n_wors
                    sw_b = float((pred_b[idx][yt_b == 2] == 2).sum()) / n_wors
                    boot_sensw_diff.append(sw_b - sw_a)

                # PPV-W (precision for worsening class)
                npw_a = (pred_a[idx] == 2).sum()
                npw_b = (pred_b[idx] == 2).sum()
                if npw_a > 0 and npw_b > 0:
                    ppv_a = float(((pred_a[idx] == 2) & (yt_b == 2)).sum()) / npw_a
                    ppv_b = float(((pred_b[idx] == 2) & (yt_b == 2)).sum()) / npw_b
                    boot_ppvw_diff.append(ppv_b - ppv_a)

                try:
                    auc_a = roc_auc_score(yt_b, proba_a[idx],
                                          multi_class="ovr", average="macro")
                    auc_b = roc_auc_score(yt_b, proba_b[idx],
                                          multi_class="ovr", average="macro")
                    boot_auc_diff.append(auc_b - auc_a)
                except Exception:
                    pass

            def pval_and_ci(diffs):
                if len(diffs) < 10:
                    return float("nan"), float("nan"), float("nan"), float("nan")
                diffs = np.array(diffs)
                mean_diff = float(diffs.mean())
                # Two-sided p-value: proportion of resamples where diff <= 0
                p = float(np.mean(diffs <= 0))
                lo_d = float(np.percentile(diffs, ALPHA / 2 * 100))
                hi_d = float(np.percentile(diffs, (1 - ALPHA / 2) * 100))
                return mean_diff, p, lo_d, hi_d

            auc_mean, auc_p, auc_lo, auc_hi = pval_and_ci(boot_auc_diff)
            bacc_mean, bacc_p, bacc_lo, bacc_hi = pval_and_ci(boot_bacc_diff)
            f1m_mean, f1m_p, f1m_lo, f1m_hi = pval_and_ci(boot_f1m_diff)
            sw_mean, sw_p, sw_lo, sw_hi = pval_and_ci(boot_sensw_diff)
            ppvw_mean, ppvw_p, ppvw_lo, ppvw_hi = pval_and_ci(boot_ppvw_diff)

            paired_rows.append({
                "label": ltype,
                "model": model_name,
                "cond_from": cond_a,
                "cond_to": cond_b,
                "AUC_diff": round(auc_mean, 4),
                "AUC_diff_lo": round(auc_lo, 4),
                "AUC_diff_hi": round(auc_hi, 4),
                "AUC_pval": round(auc_p, 4),
                "BalAcc_diff": round(bacc_mean, 4),
                "BalAcc_diff_lo": round(bacc_lo, 4),
                "BalAcc_diff_hi": round(bacc_hi, 4),
                "BalAcc_pval": round(bacc_p, 4),
                "F1macro_diff": round(f1m_mean, 4),
                "F1macro_diff_lo": round(f1m_lo, 4),
                "F1macro_diff_hi": round(f1m_hi, 4),
                "F1macro_pval": round(f1m_p, 4),
                "SensW_diff": round(sw_mean, 4),
                "SensW_diff_lo": round(sw_lo, 4),
                "SensW_diff_hi": round(sw_hi, 4),
                "SensW_pval": round(sw_p, 4),
                "PPVW_diff": round(ppvw_mean, 4),
                "PPVW_diff_lo": round(ppvw_lo, 4),
                "PPVW_diff_hi": round(ppvw_hi, 4),
                "PPVW_pval": round(ppvw_p, 4),
            })

    print(f"  {ltype} done")

paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(OUTPUT_DIR / "paired_tests.csv", index=False)
print(f"  Saved paired tests to {OUTPUT_DIR / 'paired_tests.csv'}")

# ======================================================================
# Generate markdown report
# ======================================================================
print(f"\n[8] Writing report to {REPORT_PATH}...")

lines = []
lines.append("# Bootstrap Confidence Intervals — Feature Ablation Significance\n")
lines.append("")
lines.append(f"Percentile bootstrap, {N_BOOT} resamples, seed={SEED}. "
             f"95% CIs reported. All results on the **test set**.\n")
lines.append("")

# --- Per label type: CI tables ---
for ltype in LABEL_TYPES:
    lines.append(f"## {ltype}\n")
    lines.append("")
    lines.append("### Point estimates with 95% CIs\n")
    lines.append("")
    lines.append("| Condition | Model | AUC [95% CI] | BalAcc [95% CI] | F1-macro [95% CI] | Sens-W [95% CI] | PPV-W [95% CI] |")
    lines.append("|---|---|---|---|---|---|---|")

    sub = ci_df[ci_df["label"] == ltype]
    for cond_name in COND_NAMES:
        cond_sub = sub[sub["condition"] == cond_name]
        for _, r in cond_sub.iterrows():
            auc_str = f"{r['AUC']:.3f} [{r['AUC_lo']:.3f}, {r['AUC_hi']:.3f}]"
            bacc_str = f"{r['BalAcc']:.3f} [{r['BalAcc_lo']:.3f}, {r['BalAcc_hi']:.3f}]"
            f1m_str = f"{r['F1macro']:.3f} [{r['F1macro_lo']:.3f}, {r['F1macro_hi']:.3f}]"
            sensw_str = f"{r['SensW']:.3f} [{r['SensW_lo']:.3f}, {r['SensW_hi']:.3f}]"
            ppvw_str = f"{r['PPVW']:.3f} [{r['PPVW_lo']:.3f}, {r['PPVW_hi']:.3f}]"
            lines.append(f"| {cond_name} | {r['model']} | {auc_str} | {bacc_str} | {f1m_str} | {sensw_str} | {ppvw_str} |")

    lines.append("")
    lines.append("### Paired tests: additive value of each feature set\n")
    lines.append("")
    lines.append("Δ = (condition B) − (condition A). "
                 "p-value = proportion of bootstrap resamples where Δ ≤ 0. "
                 "Significant if p < 0.05 and 95% CI of Δ excludes 0.\n")
    lines.append("")
    lines.append("| Transition | Model | ΔAUC [95% CI] | p | ΔBalAcc [95% CI] | p | ΔF1-macro [95% CI] | p | ΔSens-W [95% CI] | p | ΔPPV-W [95% CI] | p | Sig? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    psub = paired_df[paired_df["label"] == ltype]
    for _, r in psub.iterrows():
        transition = f"{r['cond_from']} → {r['cond_to']}"
        dauc = f"{r['AUC_diff']:+.3f} [{r['AUC_diff_lo']:+.3f}, {r['AUC_diff_hi']:+.3f}]"
        dbacc = f"{r['BalAcc_diff']:+.3f} [{r['BalAcc_diff_lo']:+.3f}, {r['BalAcc_diff_hi']:+.3f}]"
        df1m = f"{r['F1macro_diff']:+.3f} [{r['F1macro_diff_lo']:+.3f}, {r['F1macro_diff_hi']:+.3f}]"
        dsw = f"{r['SensW_diff']:+.3f} [{r['SensW_diff_lo']:+.3f}, {r['SensW_diff_hi']:+.3f}]"
        dppvw = f"{r['PPVW_diff']:+.3f} [{r['PPVW_diff_lo']:+.3f}, {r['PPVW_diff_hi']:+.3f}]"
        sig = "**Yes**" if (r['AUC_pval'] < 0.05 and r['AUC_diff_lo'] > 0) else "No"
        lines.append(f"| {transition} | {r['model']} | {dauc} | {r['AUC_pval']:.3f} | "
                     f"{dbacc} | {r['BalAcc_pval']:.3f} | {df1m} | {r['F1macro_pval']:.3f} | "
                     f"{dsw} | {r['SensW_pval']:.3f} | {dppvw} | {r['PPVW_pval']:.3f} | {sig} |")

    lines.append("")
    lines.append("---\n")
    lines.append("")

# --- Summary: key transitions ---
lines.append("## Summary: Which feature additions are significant?\n")
lines.append("")
lines.append("| Label | Transition | Models with significant ΔAUC (p<0.05, CI>0) |")
lines.append("|---|---|---|")

for ltype in LABEL_TYPES:
    for cond_a, cond_b in ADJACENT_PAIRS:
        psub = paired_df[
            (paired_df["label"] == ltype) &
            (paired_df["cond_from"] == cond_a) &
            (paired_df["cond_to"] == cond_b)
        ]
        sig_models = psub[
            (psub["AUC_pval"] < 0.05) & (psub["AUC_diff_lo"] > 0)
        ]["model"].tolist()
        sig_str = ", ".join(sig_models) if sig_models else "None"
        lines.append(f"| {ltype} | {cond_a} → {cond_b} | {sig_str} |")

lines.append("")

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(lines))

print(f"  Report saved to {REPORT_PATH}")

print("\n" + "=" * 80)
print("BOOTSTRAP CI ANALYSIS COMPLETE")
print("=" * 80)
