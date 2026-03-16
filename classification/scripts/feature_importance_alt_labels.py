"""Extract feature importance for alternative label types (balanced_tercile, personal_sd).

Produces the same outputs as generate_figures.py does for sev_crossing:
  - XGBoost gain importance
  - LightGBM gain importance
  - ElasticNet coefficients (canonical + less-regularized)
  - SHAP summary values per class (XGBoost + LightGBM)

All outputs saved to models/classifier_balanced/ and models/classifier_personal_sd_all/.
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/processed")
CLASS_NAMES = ["improving", "stable", "worsening"]

# ======================================================================
# Load shared data
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

X39_tr = np.hstack([X_train, lag_tr, get_pmcesd(pid_train)])
X39_va = np.hstack([X_val, lag_va, get_pmcesd(pid_val)])
X39_te = np.hstack([X_test, lag_te, get_pmcesd(pid_test)])

feature_names_39 = base_feature_names + lag_cols + ["person_mean_cesd"]
print(f"  {len(feature_names_39)} features, {X39_tr.shape[0]} train obs")


# ======================================================================
# Label constructors
# ======================================================================
def make_balanced_labels(y_delta, rng=None):
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


def make_personal_sd_labels(y_delta, pids, pid_train_arr, y_train_arr, k=1.0):
    pop_sd = float(y_train_arr.std())
    sd_map = {}
    for pid in np.unique(pid_train_arr):
        vals = y_train_arr[pid_train_arr == pid]
        sd_map[pid] = max(float(vals.std(ddof=1)) if len(vals) > 1 else pop_sd, 3.0)
    labels = np.ones(len(y_delta), dtype=int)
    for i, (d, p) in enumerate(zip(y_delta, pids)):
        sd = sd_map.get(p, pop_sd)
        if d > k * sd:
            labels[i] = 2
        elif d < -k * sd:
            labels[i] = 0
    return labels


# ======================================================================
# Feature importance extraction function
# ======================================================================
def extract_feature_importance(label_name, ytr, yva, yte, params, output_dir):
    """Train models and extract all feature importance for one label type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {label_name}: extracting feature importance")
    print(f"{'='*60}")

    dist = {c: int((yte == i).sum()) for i, c in enumerate(CLASS_NAMES)}
    print(f"  Test distribution: {dist}")

    p39 = params["base + behavioral lag + pmcesd (39)"]

    # --- XGBoost ---
    print("  Training XGBoost...")
    xgb_p = p39["XGBoost"]
    xgb = XGBClassifier(
        n_estimators=xgb_p["n_estimators"],
        max_depth=xgb_p["max_depth"],
        learning_rate=xgb_p["learning_rate"],
        min_child_weight=xgb_p["min_child_weight"],
        subsample=xgb_p["subsample"],
        colsample_bytree=xgb_p["colsample_bytree"],
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X39_tr, ytr)

    # Gain importance
    imp = xgb.feature_importances_
    df_imp = pd.DataFrame({
        "feature": feature_names_39,
        "importance": imp,
    }).sort_values("importance", ascending=False)
    df_imp.to_csv(output_dir / "feature_importance.csv", index=False)
    print(f"    XGBoost gain: {(imp > 0).sum()} nonzero features")

    # SHAP
    print("  Computing XGBoost SHAP...")
    explainer = shap.TreeExplainer(xgb)
    shap_values_raw = explainer.shap_values(X39_te)
    # Normalize shape: could be list of (n,f) or ndarray (n,f,c)
    if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        # (n_samples, n_features, n_classes) → list of (n_samples, n_features)
        shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
    else:
        shap_values = shap_values_raw
    np.save(output_dir / "shap_values_xgb.npy", shap_values)
    np.save(output_dir / "shap_expected_value_xgb.npy", explainer.expected_value)

    rows = []
    for cls_i, cls in enumerate(CLASS_NAMES):
        means = np.abs(shap_values[cls_i]).mean(axis=0)
        for j, feat in enumerate(feature_names_39):
            rows.append({"feature": feat, "class": cls, "mean_abs_shap": float(means[j])})
    pd.DataFrame(rows).to_csv(output_dir / "shap_summary_xgboost.csv", index=False)

    # --- LightGBM ---
    print("  Training LightGBM...")
    lgb_p = p39["LightGBM"]
    lgbm = LGBMClassifier(
        n_estimators=lgb_p["n_estimators"],
        max_depth=lgb_p["max_depth"],
        learning_rate=lgb_p["learning_rate"],
        num_leaves=lgb_p["num_leaves"],
        min_child_samples=lgb_p["min_child_samples"],
        subsample=lgb_p["subsample"],
        colsample_bytree=lgb_p["colsample_bytree"],
        reg_alpha=lgb_p["reg_alpha"],
        reg_lambda=lgb_p["reg_lambda"],
        objective="multiclass",
        num_class=3,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(X39_tr, ytr)

    lgb_imp = lgbm.feature_importances_
    df_lgb = pd.DataFrame({
        "feature": feature_names_39,
        "importance": lgb_imp,
    }).sort_values("importance", ascending=False)
    df_lgb.to_csv(output_dir / "lgbm_feature_importance.csv", index=False)
    print(f"    LightGBM gain: {(lgb_imp > 0).sum()} nonzero features")

    # SHAP
    print("  Computing LightGBM SHAP...")
    explainer_lgbm = shap.TreeExplainer(lgbm)
    shap_values_lgbm_raw = explainer_lgbm.shap_values(X39_te)
    if isinstance(shap_values_lgbm_raw, np.ndarray) and shap_values_lgbm_raw.ndim == 3:
        shap_values_lgbm = [shap_values_lgbm_raw[:, :, i] for i in range(shap_values_lgbm_raw.shape[2])]
    else:
        shap_values_lgbm = shap_values_lgbm_raw
    np.save(output_dir / "shap_values_lgbm.npy", shap_values_lgbm)
    np.save(output_dir / "shap_expected_value_lgbm.npy", explainer_lgbm.expected_value)

    rows = []
    for cls_i, cls in enumerate(CLASS_NAMES):
        means = np.abs(shap_values_lgbm[cls_i]).mean(axis=0)
        for j, feat in enumerate(feature_names_39):
            rows.append({"feature": feat, "class": cls, "mean_abs_shap": float(means[j])})
    pd.DataFrame(rows).to_csv(output_dir / "shap_summary_lightgbm.csv", index=False)

    # --- ElasticNet (canonical — grid-searched params) ---
    print("  Training ElasticNet (canonical)...")
    en_p = p39["ElasticNet"]
    scaler = StandardScaler()
    X_sc_tr = scaler.fit_transform(X39_tr)
    X_sc_te = scaler.transform(X39_te)

    en = LogisticRegression(
        C=en_p["C"],
        penalty="elasticnet",
        solver="saga",
        l1_ratio=en_p["l1_ratio"],
        max_iter=5000,

        random_state=42,
    )
    en.fit(X_sc_tr, ytr)

    rows = []
    for cls_i, cls in enumerate(CLASS_NAMES):
        for j, feat in enumerate(feature_names_39):
            rows.append({
                "feature": feat,
                "class": cls,
                "coefficient": float(en.coef_[cls_i, j]),
            })
    pd.DataFrame(rows).to_csv(output_dir / "feature_coefficients.csv", index=False)
    nonzero_can = sum(1 for r in rows if abs(r["coefficient"]) > 1e-10)
    print(f"    ElasticNet canonical: {nonzero_can} nonzero coefficients (across all classes)")

    # --- ElasticNet (less regularized — for interpretability) ---
    print("  Training ElasticNet (interpretable, C=0.1, l1=0.9)...")
    en_interp = LogisticRegression(
        C=0.1,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.9,
        max_iter=5000,

        random_state=42,
    )
    en_interp.fit(X_sc_tr, ytr)

    rows = []
    for cls_i, cls in enumerate(CLASS_NAMES):
        for j, feat in enumerate(feature_names_39):
            rows.append({
                "feature": feat,
                "class": cls,
                "coefficient": float(en_interp.coef_[cls_i, j]),
            })
    pd.DataFrame(rows).to_csv(output_dir / "feature_coefficients_interp.csv", index=False)
    nonzero_interp = sum(1 for r in rows if abs(r["coefficient"]) > 1e-10)
    print(f"    ElasticNet interp: {nonzero_interp} nonzero coefficients (across all classes)")

    print(f"  Done — all files saved to {output_dir}/")


# ======================================================================
# Build labels
# ======================================================================
print("\n[2] Building labels...")

# Balanced tercile
ytr_bal = make_balanced_labels(y_train, rng=np.random.RandomState(42))
yva_bal = make_balanced_labels(y_val, rng=np.random.RandomState(42))
yte_bal = make_balanced_labels(y_test, rng=np.random.RandomState(42))
print(f"  Balanced tercile test: {dict(zip(CLASS_NAMES, [int((yte_bal==i).sum()) for i in range(3)]))}")

# Personal SD
ytr_psd = make_personal_sd_labels(y_train, pid_train, pid_train, y_train)
yva_psd = make_personal_sd_labels(y_val, pid_val, pid_train, y_train)
yte_psd = make_personal_sd_labels(y_test, pid_test, pid_train, y_train)
print(f"  Personal SD test: {dict(zip(CLASS_NAMES, [int((yte_psd==i).sum()) for i in range(3)]))}")

# ======================================================================
# Extract feature importance
# ======================================================================

# Load grid-searched params
with open("models/classifier_balanced/best_params.yaml") as f:
    bal_params = yaml.safe_load(f)

with open("models/classifier_personal_sd_all/best_params.yaml") as f:
    psd_params = yaml.safe_load(f)

extract_feature_importance(
    "balanced_tercile", ytr_bal, yva_bal, yte_bal,
    bal_params, "models/classifier_balanced"
)

extract_feature_importance(
    "personal_sd", ytr_psd, yva_psd, yte_psd,
    psd_params, "models/classifier_personal_sd_all"
)

print("\n" + "="*60)
print("All feature importance extracted for both alternative labels.")
print("="*60)
