"""Train ElasticNet logistic regression classifier for mood direction prediction.

Predicts whether a person's CESD score will improve, stay stable, or worsen
in the next observation period. Three label types are supported:

  sev_crossing  Severity boundary crossing using clinical CESD thresholds
                (minimal <16, moderate 16-23, severe >=24). Most clinically
                meaningful — flags who will cross a severity boundary.

  thresh_5      5-point change threshold (MCID — minimal clinically important
                difference for CESD-20).

  thresh_10     10-point threshold (conservative, large changes only).

  personal_sd   Personalized threshold: each person's own running SD of
                cesd_delta (from training data). Worsening = change >
                k * person_SD. Use --k to set the multiplier (default 1.0).

  balanced_tercile  Tercile-based labeling: splits the training-set CESD delta
                    distribution into 3 equal-sized bins. Thresholds are the
                    33rd and 67th percentiles of the training delta.

Key improvements over regression + threshold:
  - Optimises balanced accuracy directly (class_weight='balanced')
  - Includes lag-1 features: previous period's behaviour + cesd_delta
  - Lag features capture temporal momentum and make PID OHE redundant

Usage:
    # Default: severity crossing, with lag features
    python scripts/train_classifier.py

    # 5-point threshold, with PID intercepts
    python scripts/train_classifier.py --label-type thresh_5 --use-pid

    # Severity crossing, no lag (ablation)
    python scripts/train_classifier.py --no-lag --output-dir models/classifier_nolag

    # With dev features
    python scripts/train_classifier.py --use-dev
"""

import sys
import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train ElasticNet classifier for mood direction prediction",
)
parser.add_argument(
    "--data-dir", default="data/processed",
    help="Directory with X_*.npy, y_*.npy, pid_*.npy, *_scaled.csv files",
)
parser.add_argument(
    "--config", default="configs/models/classifier.yaml",
    help="Model config YAML",
)
parser.add_argument(
    "--output-dir", default="models/classifier",
    help="Output directory for model and results",
)
parser.add_argument(
    "--label-type", default=None,
    choices=["sev_crossing", "thresh_5", "thresh_10", "personal_sd", "balanced_tercile"],
    help="Label type (overrides config)",
)
parser.add_argument(
    "--k", type=float, default=1.0,
    help="Multiplier for personal_sd label (default 1.0 = 1 SD threshold)",
)
parser.add_argument(
    "--no-lag", action="store_true",
    help="Disable lag-1 features (ablation)",
)
parser.add_argument(
    "--use-pid", action="store_true",
    help="Prepend PID one-hot encoding to feature matrix",
)
parser.add_argument(
    "--use-dev", action="store_true",
    help="Append dev (within-person deviation) features",
)
parser.add_argument(
    "--use-person-cesd", action="store_true",
    help="Append person_mean_cesd trait feature (mean prior_cesd over training periods per person)",
)
args = parser.parse_args()

data_dir   = Path(args.data_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

with open(args.config) as f:
    config = yaml.safe_load(f)

label_type   = args.label_type or config["label"]["type"]
sev_minor    = config["label"].get("sev_minor",    16)
sev_moderate = config["label"].get("sev_moderate", 24)
params       = config["params"]
Cs           = params["Cs"]
l1_ratios    = params["l1_ratios"]
max_iter     = params.get("max_iter", 2000)
use_lag      = not args.no_lag

# Balanced tercile: rank-based assignment for equal class sizes
tercile_lo, tercile_hi = None, None

print("=" * 70)
print("ELASTICNET CLASSIFIER — MOOD DIRECTION PREDICTION")
print("=" * 70)
print(f"Config:      {args.config}")
print(f"Output dir:  {output_dir}")
print(f"Label type:  {label_type}")
print(f"Use lag:     {use_lag}")
print(f"Use PID:     {args.use_pid}")
print(f"Use dev:     {args.use_dev}")
print(f"Grid:        {len(Cs)} Cs x {len(l1_ratios)} l1_ratios = "
      f"{len(Cs) * len(l1_ratios)} combos")

# ======================================================================
# Step 1: Load data
# ======================================================================
print("\n[Step 1] Loading data...")

X_train = np.load(data_dir / "X_train.npy")
X_val   = np.load(data_dir / "X_val.npy")
y_train = np.load(data_dir / "y_train.npy")
y_val   = np.load(data_dir / "y_val.npy")
pid_train = np.load(data_dir / "pid_train.npy")
pid_val   = np.load(data_dir / "pid_val.npy")

feature_names_pkl = Path("models") / "feature_names.pkl"
if feature_names_pkl.exists():
    with open(feature_names_pkl, "rb") as f:
        base_feature_names = pickle.load(f)
else:
    base_feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

prior_train = X_train[:, 0]  # prior_cesd is col 0
prior_val   = X_val[:, 0]

print(f"  Train: X={X_train.shape}, y={y_train.shape}")
print(f"  Val:   X={X_val.shape},   y={y_val.shape}")

# Compute balanced tercile via rank-based assignment
if label_type == "balanced_tercile":
    sorted_train = np.sort(y_train)
    n_per = len(y_train) // 3
    tercile_lo = (sorted_train[n_per - 1] + sorted_train[n_per]) / 2.0
    tercile_hi = (sorted_train[2 * n_per - 1] + sorted_train[2 * n_per]) / 2.0
    print(f"  balanced_tercile boundary midpoints: lo={tercile_lo:.2f}, hi={tercile_hi:.2f}")

# Compute per-person mean CESD from training data (trait baseline feature)
person_mean_cesd: dict = {}
pop_mean_cesd = float(prior_train.mean())
if args.use_person_cesd:
    for pid in np.unique(pid_train):
        person_mean_cesd[pid] = float(prior_train[pid_train == pid].mean())
    print(f"  person_mean_cesd: pop_mean={pop_mean_cesd:.1f}, "
          f"person range [{min(person_mean_cesd.values()):.1f}, {max(person_mean_cesd.values()):.1f}]")

# Compute per-person SD from training data (used by personal_sd label)
person_sd: dict = {}
pop_sd = float(y_train.std())
k_mult = args.k
if label_type == "personal_sd":
    for pid in np.unique(pid_train):
        vals = y_train[pid_train == pid]
        person_sd[pid] = max(float(vals.std(ddof=1)) if len(vals) > 1 else pop_sd, 3.0)
    print(f"  personal_sd: k={k_mult}, pop_sd={pop_sd:.2f}, "
          f"person_sd range [{min(person_sd.values()):.2f}, {max(person_sd.values()):.2f}]")

# ======================================================================
# Step 2: Build labels
# ======================================================================
print(f"\n[Step 2] Building labels ({label_type})...")

def severity(cesd):
    return np.where(cesd < sev_minor, 0, np.where(cesd < sev_moderate, 1, 2))

def make_labels(y_delta, prior, pids, ltype):
    if ltype == "sev_crossing":
        sb = severity(prior)
        sa = severity(np.clip(prior + y_delta, 0, 60))
        return np.where(sa < sb, 0, np.where(sa > sb, 2, 1))
    if ltype == "personal_sd":
        labels = np.ones(len(y_delta), dtype=int)
        for i, (d, p) in enumerate(zip(y_delta, pids)):
            sd = person_sd.get(p, pop_sd)
            thresh = k_mult * sd
            if d > thresh:
                labels[i] = 2
            elif d < -thresh:
                labels[i] = 0
        return labels
    if ltype == "balanced_tercile":
        n = len(y_delta)
        n_per = n // 3
        rng = np.random.RandomState(42)
        order = np.lexsort((rng.random(n), y_delta))
        labels = np.empty(n, dtype=int)
        labels[order[:n_per]] = 0
        labels[order[n_per:2 * n_per]] = 1
        labels[order[2 * n_per:]] = 2
        return labels
    thresh = 5 if ltype == "thresh_5" else 10
    return np.where(y_delta < -thresh, 0, np.where(y_delta > thresh, 2, 1))

label_names = ["improving", "stable", "worsening"]

y_tr = make_labels(y_train, prior_train, pid_train, label_type)
y_va = make_labels(y_val,   prior_val,   pid_val,   label_type)

for split, labels in [("Train", y_tr), ("Val", y_va)]:
    dist = " | ".join(
        f"{label_names[i]}={int((labels==i).sum())} ({(labels==i).mean()*100:.0f}%)"
        for i in range(3)
    )
    print(f"  {split}: {dist}")

# ======================================================================
# Step 3: Build lag features
# ======================================================================
feature_names = list(base_feature_names)
X_tr = X_train.copy()
X_va = X_val.copy()

if use_lag:
    print("\n[Step 3] Building lag-1 features from scaled CSVs...")

    feat_cols = base_feature_names
    lag_cols  = [f"lag_{c}" for c in feat_cols] + ["lag_cesd_delta"]

    all_df = pd.concat([
        pd.read_csv(data_dir / "train_scaled.csv"),
        pd.read_csv(data_dir / "val_scaled.csv"),
        pd.read_csv(data_dir / "test_scaled.csv"),
    ]).sort_values(["pid", "period_number"]).reset_index(drop=True)

    available = [c for c in feat_cols if c in all_df.columns]
    for col in available:
        all_df[f"lag_{col}"] = all_df.groupby("pid")[col].shift(1)
    # For features not in CSV keep lag=0
    for col in feat_cols:
        if col not in available and f"lag_{col}" not in all_df.columns:
            all_df[f"lag_{col}"] = 0.0

    all_df["lag_cesd_delta"] = all_df.groupby("pid")["target_cesd_delta"].shift(1)
    all_df[lag_cols] = all_df[lag_cols].fillna(0)

    df_tr = all_df[all_df["split"] == "train"].copy()
    df_va = all_df[all_df["split"] == "val"].copy()

    # Align lag arrays with numpy row order using pid+period_number
    pid_pn_tr = pd.DataFrame({"pid": pid_train,
                               "period_number": df_tr["period_number"].values})
    pid_pn_va = pd.DataFrame({"pid": pid_val,
                               "period_number": df_va["period_number"].values})

    lag_tr = df_tr[lag_cols].values
    lag_va = df_va[lag_cols].values

    X_tr = np.hstack([X_tr, lag_tr])
    X_va = np.hstack([X_va, lag_va])
    feature_names = feature_names + lag_cols
    print(f"  Lag features added: {len(lag_cols)} columns  "
          f"(X_train now {X_tr.shape})")
else:
    print("\n[Step 3] Lag features disabled (--no-lag).")

# ======================================================================
# Step 4: Optionally append dev features
# ======================================================================
if args.use_dev:
    print("\n[Step 4] Appending dev features...")
    X_dev_tr = np.load(data_dir / "X_dev_train.npy")
    X_dev_va = np.load(data_dir / "X_dev_val.npy")
    dev_names = [f"dev_{i}" for i in range(X_dev_tr.shape[1])]
    X_tr = np.hstack([X_tr, X_dev_tr])
    X_va = np.hstack([X_va, X_dev_va])
    feature_names = feature_names + dev_names
    print(f"  Dev features: {len(dev_names)} columns appended")
else:
    print("\n[Step 4] Dev features not used.")

# ======================================================================
# Step 5: Optionally prepend PID OHE
# ======================================================================
if args.use_pid:
    print("\n[Step 5] Prepending PID one-hot encoding...")
    pid_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    pid_enc.fit(pid_train.reshape(-1, 1))
    pid_tr_ohe = pid_enc.transform(pid_train.reshape(-1, 1))
    pid_va_ohe = pid_enc.transform(pid_val.reshape(-1, 1))
    pid_names  = [f"pid_{p}" for p in pid_enc.categories_[0]]
    X_tr = np.hstack([pid_tr_ohe, X_tr])
    X_va = np.hstack([pid_va_ohe, X_va])
    feature_names = pid_names + feature_names
    print(f"  PID OHE: {len(pid_names)} columns prepended")
else:
    print("\n[Step 5] PID OHE not used.")
    pid_enc = None

# ======================================================================
# Step 5b: Optionally append person_mean_cesd trait feature
# ======================================================================
if args.use_person_cesd:
    print("\n[Step 5b] Appending person_mean_cesd trait feature...")
    pmcesd_tr = np.array([person_mean_cesd.get(p, pop_mean_cesd) for p in pid_train]).reshape(-1, 1)
    pmcesd_va = np.array([person_mean_cesd.get(p, pop_mean_cesd) for p in pid_val]).reshape(-1, 1)
    X_tr = np.hstack([X_tr, pmcesd_tr])
    X_va = np.hstack([X_va, pmcesd_va])
    feature_names = feature_names + ["person_mean_cesd"]
    print(f"  person_mean_cesd appended (X_train now {X_tr.shape})")
else:
    print("\n[Step 5b] person_mean_cesd not used.")

print(f"\n  Final feature matrix: {X_tr.shape[1]} features")

# ======================================================================
# Step 6: Grid search
# ======================================================================
print(f"\n[Step 6] Grid search ({len(Cs) * len(l1_ratios)} combos)...")

best_bacc, best_C, best_l1r, best_clf = -1, None, None, None
results = []
n_combos = len(Cs) * len(l1_ratios)
combo = 0

for C in Cs:
    for l1r in l1_ratios:
        combo += 1
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=C,
            l1_ratio=l1r,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=42,
        )
        clf.fit(X_tr, y_tr)
        yp = clf.predict(X_va)
        bacc = balanced_accuracy_score(y_va, yp)
        f1w  = f1_score(y_va, yp, labels=[2], average="macro", zero_division=0)
        results.append({"C": C, "l1_ratio": l1r, "val_bacc": bacc, "val_f1_worse": f1w})
        if bacc > best_bacc:
            best_bacc = bacc
            best_C, best_l1r, best_clf = C, l1r, clf
        if combo % 8 == 0 or combo == n_combos:
            print(f"  {combo}/{n_combos} combos evaluated", end="\r")

print(f"\n  Best C={best_C}  l1_ratio={best_l1r}  val_bacc={best_bacc:.4f}")

results_df = pd.DataFrame(results).sort_values("val_bacc", ascending=False)
results_df.to_csv(output_dir / "grid_search_results.csv", index=False)

# ======================================================================
# Step 7: Evaluate final model
# ======================================================================
print("\n[Step 7] Evaluating final model...")

y_pred_tr = best_clf.predict(X_tr)
y_pred_va = best_clf.predict(X_va)

def metrics(y_true, y_pred, split):
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1w  = f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
    f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    sens = ((y_pred==2) & (y_true==2)).sum() / max((y_true==2).sum(), 1)
    acc  = (y_true == y_pred).mean()
    return {"split": split, "acc": acc, "bal_acc": bacc,
            "f1_macro": f1m, "f1_worsening": f1w, "sens_worsening": sens}

tr_m = metrics(y_tr, y_pred_tr, "Train")
va_m = metrics(y_va, y_pred_va, "Val")

print(f"\n  {'Split':8s}  {'Acc':>6}  {'BalAcc':>7}  {'F1-worse':>9}  "
      f"{'Sens-worse':>11}  {'F1-macro':>9}")
print(f"  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*11}  {'-'*9}")
for m in [tr_m, va_m]:
    print(f"  {m['split']:8s}  {m['acc']:6.3f}  {m['bal_acc']:7.3f}  "
          f"{m['f1_worsening']:9.3f}  {m['sens_worsening']:11.3f}  {m['f1_macro']:9.3f}")

print(f"\n  Validation classification report:")
print(classification_report(y_va, y_pred_va, target_names=label_names,
                             zero_division=0, digits=3))

print("  Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_va, y_pred_va, labels=[0, 1, 2])
print(f"  {'':12s}  " + "  ".join(f"pred_{n:8s}" for n in label_names))
for i, row in enumerate(cm):
    print(f"  true_{label_names[i]:8s}  " + "  ".join(f"{v:12d}" for v in row))

# ======================================================================
# Step 8: Save predictions
# ======================================================================
print("\n[Step 8] Saving predictions...")
np.save(output_dir / "y_pred_train.npy", y_pred_tr)
np.save(output_dir / "y_pred_val.npy",   y_pred_va)
np.save(output_dir / "y_true_train.npy", y_tr)
np.save(output_dir / "y_true_val.npy",   y_va)

# Probabilities
proba_va = best_clf.predict_proba(X_va)
np.save(output_dir / "y_proba_val.npy", proba_va)

print(f"  Saved to {output_dir}")

# ======================================================================
# Step 9: Save model artifact
# ======================================================================
print("\n[Step 9] Saving model artifact...")
joblib.dump(best_clf, output_dir / "model.joblib")
if pid_enc is not None:
    joblib.dump(pid_enc, output_dir / "pid_encoder.joblib")

best_params_out = {
    "label_type":         label_type,
    "k":                  k_mult if label_type == "personal_sd" else None,
    "use_lag":            use_lag,
    "use_pid":            args.use_pid,
    "use_dev":            args.use_dev,
    "use_person_cesd":    args.use_person_cesd,
    "best_C":           best_C,
    "best_l1_ratio":    best_l1r,
    "val_bal_acc":      float(va_m["bal_acc"]),
    "val_f1_worsening": float(va_m["f1_worsening"]),
    "val_sens_worsening": float(va_m["sens_worsening"]),
    "val_f1_macro":     float(va_m["f1_macro"]),
    "n_features":       int(X_tr.shape[1]),
}
import json
if label_type == "personal_sd" and person_sd:
    with open(output_dir / "person_sd.json", "w") as f:
        json.dump({str(k): v for k, v in person_sd.items()}, f, indent=2)
    print(f"  Saved person_sd to {output_dir / 'person_sd.json'}")
if args.use_person_cesd and person_mean_cesd:
    with open(output_dir / "person_mean_cesd.json", "w") as f:
        json.dump({str(k): v for k, v in person_mean_cesd.items()}, f, indent=2)
    print(f"  Saved person_mean_cesd to {output_dir / 'person_mean_cesd.json'}")
with open(output_dir / "best_params.yaml", "w") as f:
    yaml.dump(best_params_out, f, default_flow_style=False)

print(f"  Saved model to {output_dir / 'model.joblib'}")
print(f"  Saved params to {output_dir / 'best_params.yaml'}")

# ======================================================================
# Step 10: Feature coefficients
# ======================================================================
print("\n[Step 10] Saving feature coefficients...")
if len(feature_names) == X_tr.shape[1] and hasattr(best_clf, "coef_"):
    coef_rows = []
    for class_idx, class_name in enumerate(label_names):
        for feat, coef in zip(feature_names, best_clf.coef_[class_idx]):
            coef_rows.append({
                "class": class_name,
                "feature": feat,
                "coefficient": coef,
                "abs_coefficient": abs(coef),
            })
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(output_dir / "feature_coefficients.csv", index=False)

    # Top non-zero worsening features
    worse_coefs = (coef_df[coef_df["class"] == "worsening"]
                   .sort_values("abs_coefficient", ascending=False))
    nonzero = worse_coefs[worse_coefs["abs_coefficient"] > 1e-5]
    n_base = sum(1 for f in nonzero["feature"] if not f.startswith(("pid_", "lag_", "dev_")))
    print(f"  Selected features for worsening class: {len(nonzero)} total, "
          f"{n_base} base behavioral")
    if n_base > 0:
        base_feats = nonzero[~nonzero["feature"].str.startswith(("pid_", "lag_", "dev_"))]
        print(f"\n  Top base behavioral features (worsening class):")
        for _, row in base_feats.head(10).iterrows():
            print(f"    {row['feature']:40s}  {row['coefficient']:+.4f}")

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Label type:         {label_type}")
print(f"Use lag features:   {use_lag}")
print(f"Use PID OHE:        {args.use_pid}")
print(f"Use dev features:   {args.use_dev}")
print(f"Best C:             {best_C}")
print(f"Best l1_ratio:      {best_l1r}")
print(f"Val Balanced Acc:   {va_m['bal_acc']:.4f}")
print(f"Val Sens-worsening: {va_m['sens_worsening']:.4f}")
print(f"Val F1-macro:       {va_m['f1_macro']:.4f}")
print(f"\nAll outputs saved to: {output_dir}")
print("=" * 70)
