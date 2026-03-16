"""Generate publication-quality figures with bootstrap CIs.

Produces:
  1. Feature ablation figure (AUC by condition, all 4 models, with 95% CIs)
  2. Feature importance (XGBoost gain, LightGBM gain, ElasticNet coefficients, SHAP)
  3. Deployment ladder figure

Uses grid-searched params from bootstrap_ci analysis.

Outputs:
  reports/figures/10_feature_ablation.png      (updated with CIs)
  reports/figures/12_feature_importance.png     (gain + coefficients)
  reports/figures/13_deployment_ladder.png      (deployment ladder)
  reports/figures/14_shap_summary.png           (SHAP beeswarm, XGBoost + LightGBM)
  models/classifier/feature_importance.csv      (XGBoost gain-based)
  models/classifier/lgbm_feature_importance.csv (LightGBM gain-based)
  models/classifier/feature_coefficients.csv    (ElasticNet, grid-searched)
  models/classifier/shap_values_xgb.npy        (SHAP values, XGBoost, test set)
  models/classifier/shap_values_lgbm.npy       (SHAP values, LightGBM, test set)
  models/classifier/shap_expected_value_xgb.npy
  models/classifier/shap_expected_value_lgbm.npy
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import MultipleLocator
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/processed")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models/classifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# Load data
# ======================================================================
print("[1] Loading data...")
bootstrap_results = pd.read_csv("models/bootstrap_ci/bootstrap_results.csv")
deployment_results = pd.read_csv("models/deployment_scenarios/deployment_results.csv")

X_train = np.load(DATA_DIR / "X_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_val.npy")
pid_train = np.load(DATA_DIR / "pid_train.npy")

with open("models/feature_names.pkl", "rb") as f:
    base_feature_names = pickle.load(f)

# Build lag features
all_df = pd.concat([
    pd.read_csv(DATA_DIR / "train_scaled.csv"),
    pd.read_csv(DATA_DIR / "val_scaled.csv"),
    pd.read_csv(DATA_DIR / "test_scaled.csv"),
]).sort_values(["pid", "period_number"]).reset_index(drop=True)

feat_cols = base_feature_names
for col in [c for c in feat_cols if c in all_df.columns]:
    all_df[f"lag_{col}"] = all_df.groupby("pid")[col].shift(1)
for col in feat_cols:
    if f"lag_{col}" not in all_df.columns:
        all_df[f"lag_{col}"] = 0.0
all_df["lag_cesd_delta"] = all_df.groupby("pid")["target_cesd_delta"].shift(1)

lag_cols_all = [f"lag_{c}" for c in feat_cols] + ["lag_cesd_delta"]
all_df[lag_cols_all] = all_df[lag_cols_all].fillna(0)

drop_lags = ["lag_age", "lag_gender_mode_1", "lag_gender_mode_2",
             "lag_prior_cesd", "lag_cesd_delta"]
lag_cols = [c for c in lag_cols_all if c not in drop_lags]

lag_tr = all_df[all_df["split"] == "train"][lag_cols].values
lag_va = all_df[all_df["split"] == "val"][lag_cols].values
lag_te = all_df[all_df["split"] == "test"][lag_cols].values

# person_mean_cesd
pmcesd_path = Path("models/classifier_xgb_best39/person_mean_cesd.json")
if pmcesd_path.exists():
    with open(pmcesd_path) as f:
        pmcesd_raw = json.load(f)
    person_mean_cesd = {int(k) if k.isdigit() else k: v for k, v in pmcesd_raw.items()}
else:
    person_mean_cesd = {}
    for pid in np.unique(pid_train):
        person_mean_cesd[pid] = float(X_train[:, 0][pid_train == pid].mean())

pop_mean = np.mean(list(person_mean_cesd.values()))

def get_pmcesd(pids):
    return np.array([
        person_mean_cesd.get(int(p) if hasattr(p, "item") else p, pop_mean)
        for p in pids
    ]).reshape(-1, 1)

pid_val = np.load(DATA_DIR / "pid_val.npy")
pid_test = np.load(DATA_DIR / "pid_test.npy")

X38_tr = np.hstack([X_train, lag_tr])
X38_va = np.hstack([X_val, lag_va])
X39_tr = np.hstack([X38_tr, get_pmcesd(pid_train)])
X39_va = np.hstack([X38_va, get_pmcesd(pid_val)])
X39_te = np.hstack([np.hstack([X_test, lag_te]), get_pmcesd(pid_test)])

# sev_crossing labels
CONFIG = yaml.safe_load(open("configs/models/classifier.yaml"))
SEV_MINOR = CONFIG["label"].get("sev_minor", 16)
SEV_MOD = CONFIG["label"].get("sev_moderate", 24)

def severity(cesd):
    return np.where(cesd < SEV_MINOR, 0, np.where(cesd < SEV_MOD, 1, 2))

def make_sev_labels(y_delta, prior):
    sb = severity(prior)
    sa = severity(np.clip(prior + y_delta, 0, 60))
    return np.where(sa < sb, 0, np.where(sa > sb, 2, 1))

y_tr = make_sev_labels(y_train, X_train[:, 0])
y_va = make_sev_labels(y_val, X_val[:, 0])
y_te = make_sev_labels(np.load(DATA_DIR / "y_test.npy"), X_test[:, 0])

# Load grid-searched params
with open("models/bootstrap_ci/sev_crossing_best_params.yaml") as f:
    all_params = yaml.safe_load(f)
params_39 = all_params["base + behavioral lag + pmcesd (39)"]

feature_names_39 = base_feature_names + lag_cols + ["person_mean_cesd"]

print(f"  {len(feature_names_39)} features, {X39_tr.shape[0]} train obs")

# ======================================================================
# Figure 1: Feature Ablation with Bootstrap CIs
# ======================================================================
print("\n[2] Generating feature ablation figure with CIs...")

sev = bootstrap_results[bootstrap_results["label"] == "sev_crossing"].copy()
conditions = ["prior_cesd only", "base (21)", "base + behavioral lag (38)",
              "base + behavioral lag + pmcesd (39)"]
condition_short = ["Prior CES-D\nonly (1)", "Base\n(21)", "+ Behav.\nlag (38)",
                   "+ pmcesd\n(39)"]
models = ["ElasticNet", "XGBoost", "LightGBM", "SVM"]
colors = {"ElasticNet": "#4C72B0", "XGBoost": "#DD8452",
          "LightGBM": "#55A868", "SVM": "#C44E52"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax_i, metric in enumerate(["AUC", "BalAcc"]):
    ax = axes[ax_i]
    x = np.arange(len(conditions))
    width = 0.18

    for i, model in enumerate(models):
        vals, lo, hi = [], [], []
        for cond in conditions:
            row = sev[(sev["model"] == model) & (sev["condition"] == cond)]
            if len(row) == 0:
                vals.append(np.nan)
                lo.append(0)
                hi.append(0)
            else:
                r = row.iloc[0]
                vals.append(r[metric])
                lo.append(r[metric] - r[f"{metric}_lo"])
                hi.append(r[f"{metric}_hi"] - r[metric])

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=colors[model],
                       alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offset, vals, yerr=[lo, hi], fmt="none",
                    ecolor="black", elinewidth=1, capsize=3, capthick=1)

    ax.set_xticks(x)
    ax.set_xticklabels(condition_short, fontsize=9)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"Test {metric} by Feature Condition — sev_crossing", fontsize=11)
    ax.axhline(y=0.5 if metric == "AUC" else 0.333, color="gray",
               linestyle="--", alpha=0.5, linewidth=1)
    if metric == "AUC":
        ax.set_ylim(0.45, 0.98)
    else:
        ax.set_ylim(0.25, 0.92)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "10_feature_ablation.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved {FIG_DIR / '10_feature_ablation.png'}")

# ======================================================================
# Figure 2: Feature Importance (XGBoost gain + ElasticNet coefficients)
# ======================================================================
print("\n[3] Training models and extracting feature importance...")

# Class weights
classes_, counts_ = np.unique(y_tr, return_counts=True)
class_wt = {c: len(y_tr) / (len(classes_) * cnt) for c, cnt in zip(classes_, counts_)}
sample_weight = np.array([class_wt[y] for y in y_tr])

# --- XGBoost ---
bp_xgb = params_39["XGBoost"]
xgb_clf = XGBClassifier(
    **bp_xgb, gamma=0, objective="multi:softprob", num_class=3,
    eval_metric="mlogloss", use_label_encoder=False, random_state=42, verbosity=0)
xgb_clf.fit(X39_tr, y_tr, sample_weight=sample_weight,
            eval_set=[(X39_va, y_va)], verbose=False)

xgb_importance = xgb_clf.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names_39,
    "importance": xgb_importance
}).sort_values("importance", ascending=False)
importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
print(f"  Saved XGBoost gain importance → {MODEL_DIR / 'feature_importance.csv'}")

# --- LightGBM ---
bp_lgbm = params_39["LightGBM"]
lgbm_clf = LGBMClassifier(
    **bp_lgbm, objective="multiclass", num_class=3,
    class_weight="balanced", random_state=42, verbosity=-1)
lgbm_clf.fit(X39_tr, y_tr,
             eval_set=[(X39_va, y_va)], callbacks=[])

lgbm_importance = lgbm_clf.feature_importances_
lgbm_importance_df = pd.DataFrame({
    "feature": feature_names_39,
    "importance": lgbm_importance
}).sort_values("importance", ascending=False)
lgbm_importance_df.to_csv(MODEL_DIR / "lgbm_feature_importance.csv", index=False)
print(f"  Saved LightGBM gain importance → {MODEL_DIR / 'lgbm_feature_importance.csv'}")

# --- ElasticNet ---
bp_en = params_39["ElasticNet"]
en_clf = LogisticRegression(
    penalty="elasticnet", solver="saga",
    C=bp_en["C"], l1_ratio=bp_en["l1_ratio"],
    class_weight="balanced", max_iter=2000, random_state=42)
en_clf.fit(X39_tr, y_tr)

# Coefficients: shape (3, 39) for 3 classes
coef_rows = []
for cls_i, cls_name in enumerate(["improving", "stable", "worsening"]):
    for feat_i, feat_name in enumerate(feature_names_39):
        coef_rows.append({
            "feature": feat_name,
            "class": cls_name,
            "coefficient": en_clf.coef_[cls_i, feat_i]
        })
coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(MODEL_DIR / "feature_coefficients.csv", index=False)
print(f"  Saved ElasticNet coefficients (grid-searched C={bp_en['C']}, "
      f"l1={bp_en['l1_ratio']}) → {MODEL_DIR / 'feature_coefficients.csv'}")

# --- ElasticNet (less regularized, for interpretability) ---
EN_INTERP_C = 0.1
EN_INTERP_L1 = 0.9
en_interp = LogisticRegression(
    penalty="elasticnet", solver="saga",
    C=EN_INTERP_C, l1_ratio=EN_INTERP_L1,
    class_weight="balanced", max_iter=2000, random_state=42)
en_interp.fit(X39_tr, y_tr)

coef_interp_rows = []
for cls_i, cls_name in enumerate(["improving", "stable", "worsening"]):
    for feat_i, feat_name in enumerate(feature_names_39):
        coef_interp_rows.append({
            "feature": feat_name,
            "class": cls_name,
            "coefficient": en_interp.coef_[cls_i, feat_i]
        })
coef_interp_df = pd.DataFrame(coef_interp_rows)
coef_interp_df.to_csv(MODEL_DIR / "feature_coefficients_interp.csv", index=False)
print(f"  Saved ElasticNet coefficients (interpretable C={EN_INTERP_C}, "
      f"l1={EN_INTERP_L1}) → {MODEL_DIR / 'feature_coefficients_interp.csv'}")

# ======================================================================
# SHAP values (XGBoost + LightGBM, test set)
# ======================================================================
print("\n[3b] Computing SHAP values (TreeExplainer)...")

class_names = ["improving", "stable", "worsening"]

# XGBoost SHAP
xgb_explainer = shap.TreeExplainer(xgb_clf)
xgb_shap_values = xgb_explainer.shap_values(X39_te)  # list of 3 arrays (n_test, 39)
np.save(MODEL_DIR / "shap_values_xgb.npy", np.array(xgb_shap_values))
np.save(MODEL_DIR / "shap_expected_value_xgb.npy", np.array(xgb_explainer.expected_value))
print(f"  Saved XGBoost SHAP values → {MODEL_DIR / 'shap_values_xgb.npy'}")

# LightGBM SHAP
lgbm_explainer = shap.TreeExplainer(lgbm_clf)
lgbm_shap_values = lgbm_explainer.shap_values(X39_te)  # list of 3 arrays
np.save(MODEL_DIR / "shap_values_lgbm.npy", np.array(lgbm_shap_values))
np.save(MODEL_DIR / "shap_expected_value_lgbm.npy", np.array(lgbm_explainer.expected_value))
print(f"  Saved LightGBM SHAP values → {MODEL_DIR / 'shap_values_lgbm.npy'}")

# Normalize SHAP outputs to consistent list-of-arrays format
def normalize_shap(sv, n_classes=3):
    """Ensure shap_values is a list of (n_samples, n_features) arrays."""
    sv = np.array(sv)
    if sv.ndim == 3 and sv.shape[0] == n_classes:
        return [sv[i] for i in range(n_classes)]
    if sv.ndim == 3 and sv.shape[0] != n_classes:
        # shape (n_samples, n_features, n_classes)
        return [sv[:, :, i] for i in range(n_classes)]
    return sv  # already a list

xgb_shap_values = normalize_shap(xgb_shap_values)
lgbm_shap_values = normalize_shap(lgbm_shap_values)

# Save mean |SHAP| per feature per class as CSV for easy reference
for model_name, sv in [("xgboost", xgb_shap_values), ("lightgbm", lgbm_shap_values)]:
    rows = []
    for cls_i, cls_name in enumerate(class_names):
        mean_abs = np.abs(sv[cls_i]).mean(axis=0)
        for feat_i, feat_name in enumerate(feature_names_39):
            rows.append({
                "feature": feat_name,
                "class": cls_name,
                "mean_abs_shap": mean_abs[feat_i]
            })
    shap_df = pd.DataFrame(rows)
    shap_df.to_csv(MODEL_DIR / f"shap_summary_{model_name}.csv", index=False)
    print(f"  Saved {model_name} SHAP summary → {MODEL_DIR / f'shap_summary_{model_name}.csv'}")

# --- SHAP by transition type (worsening cases only) ---
print("\n[3b-ii] Computing SHAP by severity transition type...")

y_test = np.load(DATA_DIR / "y_test.npy")
prior_te = X_test[:, 0]
post_te = np.clip(prior_te + y_test, 0, 60)
sev_before = severity(prior_te)
sev_after = severity(post_te)
sev_labels = {0: "min", 1: "mod", 2: "sev"}

transitions = np.array([
    f"{sev_labels[int(sb)]}→{sev_labels[int(sa)]}" if sa > sb else "not_worsening"
    for sb, sa in zip(sev_before, sev_after)
])
worsening_mask = sev_after > sev_before

for model_name, sv in [("xgboost", xgb_shap_values), ("lightgbm", lgbm_shap_values)]:
    shap_wors = sv[2]  # worsening class SHAP
    rows = []

    # All worsening cases combined
    for subset_name, mask in [
        ("all_worsening", worsening_mask),
        ("min→mod", transitions == "min→mod"),
        ("min→sev", transitions == "min→sev"),
        ("mod→sev", transitions == "mod→sev"),
    ]:
        n = mask.sum()
        if n == 0:
            continue
        mean_abs = np.abs(shap_wors[mask]).mean(axis=0)
        ranked = np.argsort(mean_abs)[::-1]
        for rank, idx in enumerate(ranked[:10]):
            rows.append({
                "transition": subset_name,
                "n_cases": int(n),
                "rank": rank + 1,
                "feature": feature_names_39[idx],
                "mean_abs_shap": float(mean_abs[idx]),
            })

    trans_df = pd.DataFrame(rows)
    trans_df.to_csv(MODEL_DIR / f"shap_by_transition_{model_name}.csv", index=False)
    print(f"  Saved {model_name} transition SHAP → {MODEL_DIR / f'shap_by_transition_{model_name}.csv'}")

# ======================================================================
# Figure 2: Feature Importance (gain + coefficients)
# ======================================================================
print("\n[3c] Generating feature importance figure...")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Panel A: XGBoost gain importance (top 20)
ax = axes[0]
top20 = importance_df.head(20).iloc[::-1]
ax.barh(range(len(top20)), top20["importance"].values,
        color="#DD8452", alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["feature"].values, fontsize=8)
ax.set_xlabel("Feature Importance (gain)", fontsize=10)
ax.set_title("A. XGBoost — Top 20 (gain)", fontsize=11)
ax.grid(axis="x", alpha=0.3)

# Panel B: LightGBM gain importance (top 20)
ax = axes[1]
top20_lgbm = lgbm_importance_df.head(20).iloc[::-1]
ax.barh(range(len(top20_lgbm)), top20_lgbm["importance"].values,
        color="#55A868", alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(top20_lgbm)))
ax.set_yticklabels(top20_lgbm["feature"].values, fontsize=8)
ax.set_xlabel("Feature Importance (gain)", fontsize=10)
ax.set_title("B. LightGBM — Top 20 (gain)", fontsize=11)
ax.grid(axis="x", alpha=0.3)

# Panel C: ElasticNet worsening coefficients (less regularized, for interpretability)
ax = axes[2]
wors_coef = coef_interp_df[coef_interp_df["class"] == "worsening"].copy()
wors_coef["abs_coef"] = wors_coef["coefficient"].abs()
wors_nonzero = wors_coef[wors_coef["coefficient"] != 0].sort_values("abs_coef", ascending=True)
colors_en = ["#C44E52" if v > 0 else "#4C72B0" for v in wors_nonzero["coefficient"].values]
ax.barh(range(len(wors_nonzero)), wors_nonzero["coefficient"].values,
        color=colors_en, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(wors_nonzero)))
ax.set_yticklabels(wors_nonzero["feature"].values, fontsize=8)
ax.set_xlabel("Coefficient (worsening class)", fontsize=10)
ax.set_title(f"C. ElasticNet — Worsening (C={EN_INTERP_C})", fontsize=11)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "12_feature_importance.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved {FIG_DIR / '12_feature_importance.png'}")

# ======================================================================
# Figure 4: SHAP Summary — mean |SHAP| bar charts
# Row 1: All classes combined, all features (top 15)
# Row 2: All classes combined, behavioral features only (excl. CES-D)
# ======================================================================
print("\n[3d] Generating SHAP summary figure...")

from matplotlib.patches import Patch

cesd_features = {"prior_cesd", "person_mean_cesd"}
behavioral_idx = [i for i, f in enumerate(feature_names_39) if f not in cesd_features]
behavioral_names = [feature_names_39[i] for i in behavioral_idx]
class_names_short = ["improving", "stable", "worsening"]
class_colors = {"improving": "#4C72B0", "stable": "#55A868", "worsening": "#C44E52"}

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

for col_i, (model_label, sv) in enumerate([
    ("XGBoost", xgb_shap_values),
    ("LightGBM", lgbm_shap_values),
]):
    # Per-class mean |SHAP| for each feature: shape (39,) per class
    per_class = {}
    for cls_i, cls in enumerate(class_names_short):
        per_class[cls] = np.abs(sv[cls_i]).mean(axis=0)  # (39,)

    # Total mean |SHAP| across all classes
    total_mean_abs = sum(per_class.values())

    # --- Top row: All features, top 15, stacked by class ---
    ax = axes[0, col_i]
    top_idx = np.argsort(total_mean_abs)[::-1][:15]
    top_names = [feature_names_39[i] for i in top_idx]
    y_pos = np.arange(len(top_names))[::-1]

    left = np.zeros(len(top_names))
    for cls in class_names_short:
        vals = per_class[cls][top_idx]
        ax.barh(y_pos, vals, left=left, color=class_colors[cls],
                alpha=0.85, edgecolor="white", height=0.7, label=cls)
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (summed across classes)", fontsize=10)
    ax.set_title(f"{model_label} — All features (top 15)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # --- Bottom row: Behavioral only, top 15, stacked by class ---
    ax = axes[1, col_i]
    per_class_behav = {}
    for cls in class_names_short:
        per_class_behav[cls] = per_class[cls][behavioral_idx]
    total_behav = sum(per_class_behav.values())
    top_behav_sort = np.argsort(total_behav)[::-1][:15]
    behav_top_names = [behavioral_names[i] for i in top_behav_sort]
    y_pos = np.arange(len(behav_top_names))[::-1]

    # Filter to features with nonzero contribution
    nonzero_mask = total_behav[top_behav_sort] > 1e-8
    if nonzero_mask.sum() < len(behav_top_names):
        n_show = max(int(nonzero_mask.sum()), 3)
        top_behav_sort = top_behav_sort[:n_show]
        behav_top_names = behav_top_names[:n_show]
        y_pos = np.arange(len(behav_top_names))[::-1]

    left = np.zeros(len(behav_top_names))
    for cls in class_names_short:
        vals = per_class_behav[cls][top_behav_sort]
        bar_label = cls
        ax.barh(y_pos, vals, left=left, color=class_colors[cls],
                alpha=0.85, edgecolor="white", height=0.7, label=bar_label)
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(behav_top_names, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (summed across classes)", fontsize=10)
    ax.set_title(
        f"{model_label} — Behavioral features only (excl. CES-D, nonzero only)",
        fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

plt.tight_layout(h_pad=3.0)
plt.savefig(FIG_DIR / "14_shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved {FIG_DIR / '14_shap_summary.png'}")

# ======================================================================
# Figure 3: Deployment Ladder
# ======================================================================
print("\n[4] Generating deployment ladder figure...")

scenario_order = [
    "Population baseline", "Last-change-only", "Revert-to-person-mean",
    "Intake form only", "Onboarding",
    "Stale 4 weeks", "Stale 8 weeks", "No fresh CES-D",
    "Cold start", "Full model"
]
scenario_short = [
    "Pop.\nbaseline", "Last\nchange", "Revert\nto mean",
    "Intake\nform", "Onboard-\ning",
    "Stale\n4 wk", "Stale\n8 wk", "No fresh\nCES-D",
    "Cold\nstart", "Full\nmodel"
]

fig, ax = plt.subplots(figsize=(12, 5))

for model in models:
    aucs = []
    for sc in scenario_order:
        row = deployment_results[
            (deployment_results["scenario"] == sc) &
            (deployment_results["model"] == model)]
        if len(row) > 0:
            aucs.append(row.iloc[0]["AUC"])
        else:
            aucs.append(np.nan)
    ax.plot(range(len(scenario_order)), aucs, marker="o", label=model,
            color=colors[model], linewidth=2, markersize=6)

ax.set_xticks(range(len(scenario_order)))
ax.set_xticklabels(scenario_short, fontsize=9)
ax.set_ylabel("Test AUC", fontsize=11)
ax.set_title("Deployment Ladder — AUC by Scenario (sev_crossing)", fontsize=12)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.set_ylim(0.4, 0.95)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "13_deployment_ladder.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved {FIG_DIR / '13_deployment_ladder.png'}")

print("\n" + "=" * 60)
print("FIGURE GENERATION COMPLETE")
print("=" * 60)
