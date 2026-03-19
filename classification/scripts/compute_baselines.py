"""Baseline evaluations for all 3 classification labels.

Baselines:
  B0: No Change           — predict all stable (class 1)
  B1: Population Mean     — predict training majority class
  B2: LVCF                — repeat previous period's class label
  B3: Person-Specific Mean — predict each person's modal training class
  B4: Regression to Mean  — predict direction CES-D moves toward person mean

Metrics: AUC (OvR macro), BalAcc, F1-macro, Sens-W, PPV-W, confusion matrix

Outputs:
  classification/models/baselines/baseline_results.csv
  classification/reports/baseline_results.md
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# Paths (run from repo root)
# ============================================================
ROOT      = Path(".")
DATA_DIR  = ROOT / "data" / "processed"
LABEL_DIR = ROOT / "classification" / "labels"
OUT_DIR   = ROOT / "classification" / "models" / "baselines"
REPORT    = ROOT / "classification" / "reports" / "baseline_results.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_TYPES  = ["sev_crossing", "personal_sd", "balanced_tercile"]
CLASS_NAMES  = {0: "improving", 1: "stable", 2: "worsening"}
SEV_MINOR, SEV_MOD = 16, 24        # sev_crossing thresholds

# ============================================================
# Load processed arrays
# ============================================================
print("[1] Loading data...")
X_train       = np.load(DATA_DIR / "X_train.npy")
X_val         = np.load(DATA_DIR / "X_val.npy")
X_test        = np.load(DATA_DIR / "X_test.npy")
y_delta_train = np.load(DATA_DIR / "y_train.npy")   # continuous CES-D delta
y_delta_val   = np.load(DATA_DIR / "y_val.npy")
y_delta_test  = np.load(DATA_DIR / "y_test.npy")
pid_train     = np.load(DATA_DIR / "pid_train.npy")
pid_val       = np.load(DATA_DIR / "pid_val.npy")
pid_test      = np.load(DATA_DIR / "pid_test.npy")

prior_train = X_train[:, 0]   # column 0 = prior_cesd
prior_val   = X_val[:, 0]
prior_test  = X_test[:, 0]

train_df = pd.read_csv(DATA_DIR / "train_scaled.csv")
val_df   = pd.read_csv(DATA_DIR / "val_scaled.csv")
test_df  = pd.read_csv(DATA_DIR / "test_scaled.csv")

print(f"  Train={len(y_delta_train)}  Val={len(y_delta_val)}  Test={len(y_delta_test)}")

# ============================================================
# Person-mean CES-D (computed from training prior_cesd only)
# ============================================================
def _pid_key(p):
    return int(p) if hasattr(p, "item") else p

person_mean_cesd: dict[int, float] = {}
for pid in np.unique(pid_train):
    person_mean_cesd[_pid_key(pid)] = float(prior_train[pid_train == pid].mean())

pop_mean = float(np.mean(list(person_mean_cesd.values())))

def get_pmcesd(pids: np.ndarray) -> np.ndarray:
    return np.array([person_mean_cesd.get(_pid_key(p), pop_mean) for p in pids])

pmcesd_test = get_pmcesd(pid_test)

# Person-SD of CES-D deltas (training), needed for personal_sd B4
pop_delta_sd = float(np.std(y_delta_train))
SD_FLOOR = 3.0

person_delta_sd: dict[int, float] = {}
for pid in np.unique(pid_train):
    y_p = y_delta_train[pid_train == pid]
    sd  = max(float(np.std(y_p)) if len(y_p) > 1 else pop_delta_sd, SD_FLOOR)
    person_delta_sd[_pid_key(pid)] = sd

# ============================================================
# Load classification labels
# ============================================================
print("\n[2] Loading classification labels...")
labels: dict = {}
for lt in LABEL_TYPES:
    raw_text = (LABEL_DIR / lt / "label_info.yaml").read_text()
    try:
        info = yaml.safe_load(raw_text)
    except yaml.constructor.ConstructorError:
        info = yaml.unsafe_load(raw_text)
    labels[lt] = {
        "train": np.load(LABEL_DIR / lt / "y_train.npy"),
        "val":   np.load(LABEL_DIR / lt / "y_val.npy"),
        "test":  np.load(LABEL_DIR / lt / "y_test.npy"),
        "info":  info,
    }
    y_te = labels[lt]["test"]
    print(f"  {lt}: test imp={(y_te==0).sum()} stb={(y_te==1).sum()} wrs={(y_te==2).sum()}")

# ============================================================
# B2: LVCF — build lag-1 label for each test observation
# ============================================================
print("\n[3] Building LVCF lag labels (B2)...")

# Attach original split index so we can reorder back after sorting
train_df = train_df.copy(); train_df["_idx"] = range(len(train_df))
val_df   = val_df.copy();   val_df["_idx"]   = range(len(val_df))
test_df  = test_df.copy();  test_df["_idx"]   = range(len(test_df))

# Attach labels from all three label types
for lt in LABEL_TYPES:
    train_df[f"y_{lt}"] = labels[lt]["train"]
    val_df[f"y_{lt}"]   = labels[lt]["val"]
    test_df[f"y_{lt}"]  = labels[lt]["test"]

combined = (
    pd.concat([train_df, val_df, test_df], ignore_index=True)
    .sort_values(["pid", "period_number"])
    .reset_index(drop=True)
)

# Shift labels by 1 within each person (chronological order)
for lt in LABEL_TYPES:
    combined[f"lag_y_{lt}"] = combined.groupby("pid")[f"y_{lt}"].shift(1)

# Extract test rows in original array order; NaN means first period → default stable
test_part = combined[combined["split"] == "test"].sort_values("_idx")

lvcf_test: dict[str, np.ndarray] = {}
for lt in LABEL_TYPES:
    lvcf_test[lt] = test_part[f"lag_y_{lt}"].fillna(1).astype(int).values
    n_missing = test_part[f"lag_y_{lt}"].isna().sum()
    print(f"  {lt}: {n_missing} first-period observations defaulted to stable")

# ============================================================
# B3: Person-specific modal class (from training labels)
# ============================================================
print("\n[4] Computing person-specific modal class (B3)...")

person_modal: dict[str, dict] = {}
for lt in LABEL_TYPES:
    person_modal[lt] = {}
    y_tr = labels[lt]["train"]
    for pid in np.unique(pid_train):
        pid_k = _pid_key(pid)
        y_p   = y_tr[pid_train == pid]
        vals, counts = np.unique(y_p, return_counts=True)
        person_modal[lt][pid_k] = int(vals[np.argmax(counts)])

def predict_modal(pids: np.ndarray, lt: str) -> np.ndarray:
    return np.array([person_modal[lt].get(_pid_key(p), 1) for p in pids])

# ============================================================
# B4: Regression-to-mean predictions
# ============================================================

def severity(cesd: np.ndarray) -> np.ndarray:
    return np.where(cesd < SEV_MINOR, 0, np.where(cesd < SEV_MOD, 1, 2))

def rtm_sev_crossing(prior: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """Predict direction toward each person's mean severity band."""
    pm = get_pmcesd(pids)
    sev_now  = severity(prior)
    sev_mean = severity(pm)
    return np.where(sev_mean < sev_now, 0,           # mean band is lower → improving
                    np.where(sev_mean > sev_now, 2,   # mean band is higher → worsening
                             1))                       # same band → stable

def rtm_personal_sd(prior: np.ndarray, pids: np.ndarray, k: float = 1.0) -> np.ndarray:
    """
    Predict regession direction using person delta-SD as threshold.
    Predicted delta = person_mean_cesd − prior_cesd.
    If |predicted_delta| > k * person_delta_sd → classify as that direction.
    """
    pm   = get_pmcesd(pids)
    preds = np.ones(len(prior), dtype=int)           # default stable
    for i, (pr, pid) in enumerate(zip(prior, pids)):
        pid_k      = _pid_key(pid)
        sd         = person_delta_sd.get(pid_k, pop_delta_sd)
        pred_delta = pm[i] - pr                      # RTM predicted delta
        if pred_delta < -k * sd:
            preds[i] = 0  # predicted large improvement
        elif pred_delta > k * sd:
            preds[i] = 2  # predicted large worsening
    return preds

def rtm_balanced_tercile(prior: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """
    Predicted delta = person_mean_cesd − prior_cesd.
    Assign to tercile class using training delta distribution boundaries.
    Tercile 0 = most negative (improving), 2 = most positive (worsening).
    """
    pm          = get_pmcesd(pids)
    pred_delta  = pm - prior                         # expected regression delta
    lo = np.percentile(y_delta_train, 100 / 3)
    hi = np.percentile(y_delta_train, 200 / 3)
    return np.where(pred_delta <= lo, 0,
                    np.where(pred_delta >= hi, 2, 1))

# ============================================================
# Metrics helper
# ============================================================

def hard_proba(y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    p = np.zeros((len(y_pred), n_classes))
    for i, c in enumerate(y_pred):
        p[i, int(c)] = 1.0
    return p

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_proba: np.ndarray) -> dict:
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
    return {
        "AUC":    round(auc, 3),
        "BalAcc": round(bacc, 3),
        "F1macro": round(f1m, 3),
        "SensW":  round(sens_w, 3),
        "PPVW":   round(ppv_w, 3),
        "confusion_matrix": cm,
    }

# ============================================================
# Evaluate all baselines × all labels
# ============================================================
print("\n[5] Evaluating baselines...")

BASELINES = {
    "B0": "No Change (predict all stable)",
    "B1": "Population Mean (majority class from training)",
    "B2": "LVCF (last value carried forward)",
    "B3": "Person-Specific Mean (modal training class per person)",
    "B4": "Regression to Mean (direction toward person mean CES-D)",
}

all_results: list[dict] = []

for lt in LABEL_TYPES:
    y_tr = labels[lt]["train"]
    y_te = labels[lt]["test"]
    info = labels[lt]["info"]
    k    = float(info.get("k", 1.0))
    n    = len(y_te)

    print(f"\n  === {lt} (n_test={n}) ===")

    # B0 — No Change
    y_pred  = np.ones(n, dtype=int)
    y_proba = hard_proba(y_pred)
    m = evaluate(y_te, y_pred, y_proba)
    m.update({"label": lt, "baseline": "B0", "description": BASELINES["B0"]})
    all_results.append(m)
    print(f"  B0 No Change:          AUC={m['AUC']:.3f}  BalAcc={m['BalAcc']:.3f}"
          f"  F1={m['F1macro']:.3f}  SensW={m['SensW']:.3f}  PPVW={m['PPVW']:.3f}")

    # B1 — Population Mean
    vals, counts = np.unique(y_tr, return_counts=True)
    majority     = int(vals[np.argmax(counts)])
    y_pred       = np.full(n, majority, dtype=int)
    y_proba      = np.zeros((n, 3))
    for c, cnt in zip(vals, counts):
        y_proba[:, int(c)] = cnt / len(y_tr)
    m = evaluate(y_te, y_pred, y_proba)
    m.update({"label": lt, "baseline": "B1", "description": BASELINES["B1"],
              "B1_majority_class": CLASS_NAMES[majority]})
    all_results.append(m)
    print(f"  B1 Pop Mean:           AUC={m['AUC']:.3f}  BalAcc={m['BalAcc']:.3f}"
          f"  F1={m['F1macro']:.3f}  SensW={m['SensW']:.3f}  PPVW={m['PPVW']:.3f}"
          f"  (predicts '{CLASS_NAMES[majority]}')")

    # B2 — LVCF
    y_pred  = lvcf_test[lt]
    y_proba = hard_proba(y_pred)
    m = evaluate(y_te, y_pred, y_proba)
    m.update({"label": lt, "baseline": "B2", "description": BASELINES["B2"]})
    all_results.append(m)
    print(f"  B2 LVCF:               AUC={m['AUC']:.3f}  BalAcc={m['BalAcc']:.3f}"
          f"  F1={m['F1macro']:.3f}  SensW={m['SensW']:.3f}  PPVW={m['PPVW']:.3f}")

    # B3 — Person-Specific Mean
    y_pred  = predict_modal(pid_test, lt)
    y_proba = hard_proba(y_pred)
    m = evaluate(y_te, y_pred, y_proba)
    m.update({"label": lt, "baseline": "B3", "description": BASELINES["B3"]})
    all_results.append(m)
    print(f"  B3 Person Mean:        AUC={m['AUC']:.3f}  BalAcc={m['BalAcc']:.3f}"
          f"  F1={m['F1macro']:.3f}  SensW={m['SensW']:.3f}  PPVW={m['PPVW']:.3f}")

    # B4 — Regression to Mean
    if lt == "sev_crossing":
        y_pred = rtm_sev_crossing(prior_test, pid_test)
    elif lt == "personal_sd":
        y_pred = rtm_personal_sd(prior_test, pid_test, k=k)
    else:  # balanced_tercile
        y_pred = rtm_balanced_tercile(prior_test, pid_test)
    y_proba = hard_proba(y_pred)
    m = evaluate(y_te, y_pred, y_proba)
    m.update({"label": lt, "baseline": "B4", "description": BASELINES["B4"]})
    all_results.append(m)
    print(f"  B4 Regress to Mean:    AUC={m['AUC']:.3f}  BalAcc={m['BalAcc']:.3f}"
          f"  F1={m['F1macro']:.3f}  SensW={m['SensW']:.3f}  PPVW={m['PPVW']:.3f}")

# ============================================================
# Save CSV
# ============================================================
print("\n[6] Saving CSV...")
clean_rows = [{k: v for k, v in r.items() if k != "confusion_matrix"}
              for r in all_results]
pd.DataFrame(clean_rows).to_csv(OUT_DIR / "baseline_results.csv", index=False)
print(f"  Saved: {OUT_DIR / 'baseline_results.csv'}")

# ============================================================
# Markdown report
# ============================================================
print(f"[7] Writing report to {REPORT}...")

lines: list[str] = []
lines += [
    "# Baseline Results — All Label Types\n",
    "",
    "Classes: **0 = improving**, **1 = stable**, **2 = worsening**",
    "",
    "## Baseline Definitions",
    "",
    "| ID | Name | Rule |",
    "|---|---|---|",
    "| B0 | No Change | Predict all stable (class 1) — lower bound |",
    "| B1 | Population Mean | Predict majority class from training set |",
    "| B2 | LVCF | Repeat previous period's class label (last value carried forward) |",
    "| B3 | Person-Specific Mean | Predict each person's modal class in training |",
    "| B4 | Regression to Mean | Predict direction CES-D moves toward person's training mean |",
    "",
    "**AUC** = one-vs-rest macro. "
    "**SensW** = recall for worsening (class 2). "
    "**PPVW** = precision for worsening.",
    "",
]

for lt in LABEL_TYPES:
    lt_results = [r for r in all_results if r["label"] == lt]
    y_te = labels[lt]["test"]
    y_tr = labels[lt]["train"]

    lines += [
        f"---",
        f"",
        f"## {lt}",
        f"",
        f"Train distribution: "
        f"imp={int((y_tr==0).sum())}  stb={int((y_tr==1).sum())}  wrs={int((y_tr==2).sum())}  "
        f"(total {len(y_tr)})",
        f"",
        f"Test distribution: "
        f"imp={int((y_te==0).sum())}  stb={int((y_te==1).sum())}  wrs={int((y_te==2).sum())}  "
        f"(total {len(y_te)})",
        f"",
        f"| Baseline | AUC | BalAcc | F1-macro | Sens-W | PPV-W |",
        f"|---|---|---|---|---|---|",
    ]

    for r in lt_results:
        lines.append(
            f"| **{r['baseline']}** {r['description']} "
            f"| {r['AUC']:.3f} | {r['BalAcc']:.3f} | {r['F1macro']:.3f} "
            f"| {r['SensW']:.3f} | {r['PPVW']:.3f} |"
        )

    lines.append("")

    # Confusion matrices
    lines += [f"### Confusion Matrices ({lt})", ""]
    for r in lt_results:
        cm = r["confusion_matrix"]
        lines += [
            f"**{r['baseline']}** — {r['description']}",
            "",
            f"```",
            f"              pred_imp  pred_stb  pred_wrs",
            f"  true_imp      {cm[0,0]:6d}    {cm[0,1]:6d}    {cm[0,2]:6d}",
            f"  true_stb      {cm[1,0]:6d}    {cm[1,1]:6d}    {cm[1,2]:6d}",
            f"  true_wrs      {cm[2,0]:6d}    {cm[2,1]:6d}    {cm[2,2]:6d}",
            f"```",
            "",
        ]

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines))
print(f"  Report saved: {REPORT}")

print("\n" + "=" * 70)
print("BASELINE EVALUATION COMPLETE")
print("=" * 70)
