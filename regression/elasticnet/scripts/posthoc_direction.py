"""Post-hoc direction analysis of ElasticNet regression predictions.

Uses pre-computed classification labels from classification/labels/ to:
  1. Stratify regression error (MAE, RMSE, Bias) by direction class
  2. Derive predicted direction from regression predictions
  3. Report classification metrics (BalAcc, AUC, Sens-W, PPV-W, confusion matrix)
  4. Generate per-person confusion matrices and trajectory plots

This script answers: "Can the regression model predict the *direction* of change,
not just the magnitude?"  -- enabling direct comparison with dedicated classifiers.

Label types (from classification/labels/):
  sev_crossing      -- Clinical severity boundary crossing (CES-D thresholds 16, 24)
  personal_sd       -- Person-specific SD-based change (k=1.0)
  balanced_tercile  -- Rank-based equal-sized terciles

Usage:
    # Default (sev_crossing labels):
    python scripts/posthoc_direction.py --condition base

    # With personal_sd labels:
    python scripts/posthoc_direction.py --condition base --label-type personal_sd

    # Skip per-person plots for speed:
    python scripts/posthoc_direction.py --condition base --skip-plots
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELASTICNET_DIR = _SCRIPT_DIR.parent
_POSTHOC_DIR = _ELASTICNET_DIR.parent / "posthoc"

LABEL_NAMES = ["improving", "stable", "worsening"]
SHORT_NAMES = ["Imp", "Stb", "Wrs"]


# ---------------------------------------------------------------------------
# Helpers: Direction derivation
# ---------------------------------------------------------------------------

def severity(cesd: np.ndarray, sev_minor: int = 16, sev_moderate: int = 24) -> np.ndarray:
    """Map CES-D score to severity category: 0=minimal, 1=moderate, 2=severe."""
    return np.where(cesd < sev_minor, 0, np.where(cesd < sev_moderate, 1, 2))


def derive_direction_sev_crossing(
    y_pred: np.ndarray, prior_cesd: np.ndarray,
) -> np.ndarray:
    """Derive predicted direction using severity boundary crossing.

    Same logic as classification/scripts/train_classifier.py make_labels()
    for sev_crossing. Compares predicted post-period severity with current severity.

    Returns: 0=improving, 1=stable, 2=worsening
    """
    sev_before = severity(prior_cesd)
    sev_after = severity(np.clip(prior_cesd + y_pred, 0, 60))
    return np.where(sev_after < sev_before, 0,
                    np.where(sev_after > sev_before, 2, 1))


def derive_direction_personal_sd(
    y_pred: np.ndarray, y_train: np.ndarray, pid_train: np.ndarray,
    pids: np.ndarray, k: float = 1.0,
) -> np.ndarray:
    """Derive predicted direction using per-person SD thresholds.

    Returns: 0=improving, 1=stable, 2=worsening
    """
    pop_sd = float(np.std(y_train))
    person_thresholds: dict = {}
    for pid in np.unique(pid_train):
        vals = y_train[pid_train == pid]
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else pop_sd
        person_thresholds[pid] = max(sd, 3.0) * k

    thresholds = np.array([person_thresholds.get(p, pop_sd * k) for p in pids])
    return np.where(y_pred > thresholds, 2,
                    np.where(y_pred < -thresholds, 0, 1))


def derive_direction_balanced_tercile(
    y_pred: np.ndarray,
) -> np.ndarray:
    """Derive predicted direction using rank-based tercile assignment.

    Same logic as classification/scripts/train_classifier.py make_labels()
    for balanced_tercile. Sorts predictions and assigns equal-sized bins.

    Returns: 0=improving, 1=stable, 2=worsening
    """
    n = len(y_pred)
    n_per = n // 3
    rng = np.random.RandomState(42)
    order = np.lexsort((rng.random(n), y_pred))
    labels = np.empty(n, dtype=int)
    labels[order[:n_per]] = 0
    labels[order[n_per:2 * n_per]] = 1
    labels[order[2 * n_per:]] = 2
    return labels


# ---------------------------------------------------------------------------
# Helpers: Metrics
# ---------------------------------------------------------------------------

def compute_stratified_error(
    y_true: np.ndarray, y_pred: np.ndarray, y_labels: np.ndarray,
) -> pd.DataFrame:
    """Stratify regression error by direction class.

    Returns DataFrame with columns: direction, N, MAE, RMSE, Bias.
    """
    rows = []
    for cls, name in enumerate(LABEL_NAMES):
        mask = y_labels == cls
        n = int(mask.sum())
        if n == 0:
            rows.append({"direction": name, "N": 0, "MAE": np.nan,
                         "RMSE": np.nan, "Bias": np.nan})
            continue
        errors = y_pred[mask] - y_true[mask]
        rows.append({
            "direction": name,
            "N": n,
            "MAE": float(np.abs(errors).mean()),
            "RMSE": float(np.sqrt(np.mean(errors ** 2))),
            "Bias": float(errors.mean()),
        })
    return pd.DataFrame(rows)


def compute_classification_metrics(
    y_true_labels: np.ndarray,
    y_pred_direction: np.ndarray,
    y_pred_continuous: np.ndarray,
) -> dict:
    """Compute BalAcc, AUC (OvR), Sens-W, PPV-W.

    AUC uses continuous regression predictions as soft scores:
      - score_improving  = -y_pred  (more negative pred -> more improving)
      - score_stable     = -|y_pred|  (closer to zero -> more stable)
      - score_worsening  = y_pred  (more positive pred -> more worsening)
    """
    classes = [0, 1, 2]

    # Balanced Accuracy
    bal_acc = float(balanced_accuracy_score(y_true_labels, y_pred_direction))

    # Confusion matrix for Sens-W and PPV-W
    cm = confusion_matrix(y_true_labels, y_pred_direction, labels=classes)
    sens_w = float(cm[2, 2] / cm[2, :].sum()) if cm[2, :].sum() > 0 else 0.0
    ppv_w = float(cm[2, 2] / cm[:, 2].sum()) if cm[:, 2].sum() > 0 else 0.0

    # AUC (OvR macro) using continuous predictions as soft scores
    scores = np.column_stack([
        -y_pred_continuous,        # improving: more negative = more likely
        -np.abs(y_pred_continuous), # stable: closer to zero = more likely
        y_pred_continuous,          # worsening: more positive = more likely
    ])
    y_true_bin = label_binarize(y_true_labels, classes=classes)

    try:
        auc = float(roc_auc_score(
            y_true_bin, scores, multi_class="ovr", average="macro",
        ))
    except ValueError:
        auc = float("nan")

    return {
        "balanced_accuracy": round(bal_acc, 4),
        "auc_ovr_macro": round(auc, 4),
        "sensitivity_worsening": round(sens_w, 4),
        "ppv_worsening": round(ppv_w, 4),
    }


def compute_per_person_direction(
    y_true_dir: np.ndarray, y_pred_dir: np.ndarray, pids: np.ndarray,
) -> pd.DataFrame:
    """Compute per-person direction accuracy for a split."""
    rows = []
    for pid in sorted(np.unique(pids)):
        mask = pids == pid
        yt = y_true_dir[mask]
        yp = y_pred_dir[mask]
        n_obs = int(mask.sum())
        n_correct = int((yt == yp).sum())
        acc = n_correct / n_obs if n_obs > 0 else 0.0
        row = {
            "person_id": pid,
            "n_obs": n_obs,
            "direction_accuracy": round(acc, 4),
            "n_correct": n_correct,
        }
        for lv, ln in enumerate(LABEL_NAMES):
            row[f"n_true_{ln}"] = int((yt == lv).sum())
            row[f"n_pred_{ln}"] = int((yp == lv).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers: Plots
# ---------------------------------------------------------------------------

def _make_cm_annotation(cm: np.ndarray) -> np.ndarray:
    """Build annotation strings with count and row percentage."""
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        row_sum = cm[i, :].sum()
        for j in range(cm.shape[1]):
            if row_sum == 0:
                annot[i, j] = "0\n(-)"
            else:
                pct = 100.0 * cm[i, j] / row_sum
                annot[i, j] = f"{cm[i, j]}\n({pct:.0f}%)"
    return annot


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: Path,
) -> None:
    """Plot aggregate 3x3 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    annot = _make_cm_annotation(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt="", ax=ax, cbar=False, cmap="Blues",
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_person_cms(
    y_true_dir: np.ndarray, y_pred_dir: np.ndarray, pids: np.ndarray,
    cm_dir: Path, model_name: str, label_type: str,
) -> int:
    """Generate per-person confusion matrix PNGs."""
    cm_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for pid in sorted(np.unique(pids)):
        mask = pids == pid
        cm_p = confusion_matrix(y_true_dir[mask], y_pred_dir[mask], labels=[0, 1, 2])
        annot_p = _make_cm_annotation(cm_p)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_p, annot=annot_p, fmt="", ax=ax, cbar=False, cmap="Blues",
                    xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name}: PID {pid}\nn={mask.sum()}, labels={label_type}",
                      fontsize=10)
        plt.tight_layout()
        fig.savefig(cm_dir / f"pid_{pid}_cm.png", dpi=150)
        plt.close(fig)
        count += 1
    return count


def plot_per_person_trajectories(
    y_true: np.ndarray, y_pred: np.ndarray, pids: np.ndarray,
    period_numbers: np.ndarray, traj_dir: Path, model_name: str,
    shared_ylim: tuple,
) -> int:
    """Generate per-person trajectory PNGs."""
    traj_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for pid in sorted(np.unique(pids)):
        mask = pids == pid
        periods = period_numbers[mask]
        sort_idx = np.argsort(periods)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(periods[sort_idx], y_true[mask][sort_idx],
                "o-", color="steelblue", label="Actual", ms=5, lw=1.2)
        ax.plot(periods[sort_idx], y_pred[mask][sort_idx],
                "s--", color="coral", label="Predicted", ms=5, lw=1.2)
        ax.axhline(0, ls="-", color="black", alpha=0.2, lw=0.5)
        ax.set_title(f"{model_name}: PID {pid} (n={mask.sum()})", fontsize=10)
        ax.set_ylim(shared_ylim)
        ax.set_xlabel("Period")
        ax.set_ylabel("CESD Delta")
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        fig.savefig(traj_dir / f"pid_{pid}_trajectory.png", dpi=150)
        plt.close(fig)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-hoc direction analysis of regression predictions",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Processed data directory (default: ../../data/processed)",
    )
    parser.add_argument(
        "--labels-dir", default=None,
        help="Classification labels directory (default: ../../classification/labels)",
    )
    parser.add_argument(
        "--condition", required=True,
        help="Feature condition name (e.g., base, base_lag)",
    )
    parser.add_argument(
        "--models-dir", default=None,
        help="Base models directory (default: ../models)",
    )
    parser.add_argument(
        "--label-type", default="sev_crossing",
        choices=["sev_crossing", "personal_sd", "balanced_tercile"],
        help="Label type for direction analysis (default: sev_crossing)",
    )
    parser.add_argument(
        "--model-name", default="ElasticNet",
        help="Model name for titles (default: ElasticNet)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: models/{condition}/posthoc/{label_type})",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip per-person plots (faster)",
    )
    args = parser.parse_args()

    # Resolve defaults relative to the elasticnet directory
    repo_root = _ELASTICNET_DIR.parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data" / "processed"
    labels_dir = Path(args.labels_dir) if args.labels_dir else repo_root / "classification" / "labels"
    models_dir = Path(args.models_dir) if args.models_dir else _ELASTICNET_DIR / "models"
    output_dir = (Path(args.output_dir) if args.output_dir
                  else _POSTHOC_DIR / "elasticnet" / args.condition / args.label_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition = args.condition
    label_type = args.label_type
    model_name = f"{args.model_name} ({condition})"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"POST-HOC DIRECTION ANALYSIS: {model_name}")
    print(f"Label type: {label_type}")
    print(f"Run timestamp: {timestamp}")
    print("=" * 70)

    # ==================================================================
    # Step 1: Load data
    # ==================================================================
    print("\n[Step 1] Loading data...")

    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    y_test = np.load(data_dir / "y_test.npy")
    pid_train = np.load(data_dir / "pid_train.npy")
    pid_val = np.load(data_dir / "pid_val.npy")
    pid_test = np.load(data_dir / "pid_test.npy")

    condition_dir = models_dir / condition
    y_pred_train = np.load(condition_dir / "y_pred_train.npy")
    y_pred_val = np.load(condition_dir / "y_pred_val.npy")

    has_test = (condition_dir / "y_pred_test.npy").exists()
    y_pred_test = np.load(condition_dir / "y_pred_test.npy") if has_test else None

    # Load pre-computed classification labels
    labels_path = labels_dir / label_type
    y_label_train = np.load(labels_path / "y_train.npy")
    y_label_val = np.load(labels_path / "y_val.npy")
    y_label_test = np.load(labels_path / "y_test.npy")

    # Load period numbers for trajectory plots
    val_csv = pd.read_csv(data_dir / "val_scaled.csv")
    period_numbers_val = val_csv["period_number"].values
    test_csv = pd.read_csv(data_dir / "test_scaled.csv")
    period_numbers_test = test_csv["period_number"].values

    print(f"  Train: {len(y_train)} samples")
    print(f"  Val:   {len(y_val)} samples")
    print(f"  Test:  {len(y_test)} samples {'(predictions available)' if has_test else '(no predictions)'}")
    print(f"  Labels: {label_type} from {labels_path}")

    for split_name, labels in [("Train", y_label_train), ("Val", y_label_val),
                                ("Test", y_label_test)]:
        dist = " | ".join(
            f"{LABEL_NAMES[i]}={int((labels == i).sum())} ({(labels == i).mean() * 100:.0f}%)"
            for i in range(3)
        )
        print(f"  {split_name} labels: {dist}")

    # ==================================================================
    # Step 2: Stratified regression error
    # ==================================================================
    print("\n[Step 2] Stratified regression error by direction class...")

    splits_to_eval = [("val", y_val, y_pred_val, y_label_val)]
    if has_test:
        splits_to_eval.append(("test", y_test, y_pred_test, y_label_test))

    for split_name, yt, yp, yl in splits_to_eval:
        strat_df = compute_stratified_error(yt, yp, yl)
        strat_df.to_csv(output_dir / f"stratified_error_{split_name}.csv", index=False)

        print(f"\n  {split_name.upper()}:")
        print(f"  {'Direction':15s}  {'N':>5s}  {'MAE':>8s}  {'RMSE':>8s}  {'Bias':>8s}")
        print(f"  {'-' * 15}  {'-' * 5}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
        for _, row in strat_df.iterrows():
            print(f"  {row['direction']:15s}  {row['N']:5.0f}  "
                  f"{row['MAE']:8.3f}  {row['RMSE']:8.3f}  {row['Bias']:+8.3f}")

    # ==================================================================
    # Step 3: Derive predicted direction from regression predictions
    # ==================================================================
    print(f"\n[Step 3] Deriving predicted direction ({label_type})...")

    if label_type == "sev_crossing":
        X_train_base = np.load(data_dir / "X_train.npy")
        X_val_base = np.load(data_dir / "X_val.npy")
        X_test_base = np.load(data_dir / "X_test.npy")
        prior_train = X_train_base[:, 0]
        prior_val = X_val_base[:, 0]
        prior_test = X_test_base[:, 0]

        y_pred_dir_train = derive_direction_sev_crossing(y_pred_train, prior_train)
        y_pred_dir_val = derive_direction_sev_crossing(y_pred_val, prior_val)
        if has_test:
            y_pred_dir_test = derive_direction_sev_crossing(y_pred_test, prior_test)

    elif label_type == "personal_sd":
        y_pred_dir_train = derive_direction_personal_sd(
            y_pred_train, y_train, pid_train, pid_train)
        y_pred_dir_val = derive_direction_personal_sd(
            y_pred_val, y_train, pid_train, pid_val)
        if has_test:
            y_pred_dir_test = derive_direction_personal_sd(
                y_pred_test, y_train, pid_train, pid_test)

    elif label_type == "balanced_tercile":
        y_pred_dir_train = derive_direction_balanced_tercile(y_pred_train)
        y_pred_dir_val = derive_direction_balanced_tercile(y_pred_val)
        if has_test:
            y_pred_dir_test = derive_direction_balanced_tercile(y_pred_test)

    for name, arr in [("pred_dir_train", y_pred_dir_train),
                       ("pred_dir_val", y_pred_dir_val)]:
        dist = " | ".join(f"{LABEL_NAMES[i]}={int((arr == i).sum())}" for i in range(3))
        print(f"  {name}: {dist}")
    if has_test:
        dist = " | ".join(f"{LABEL_NAMES[i]}={int((y_pred_dir_test == i).sum())}" for i in range(3))
        print(f"  pred_dir_test: {dist}")

    # Save direction arrays for downstream use (e.g., build_report.py fig5)
    np.save(output_dir / "y_labels_val.npy", y_label_val)
    np.save(output_dir / "y_pred_direction_val.npy", y_pred_dir_val)
    if has_test:
        np.save(output_dir / "y_labels_test.npy", y_label_test)
        np.save(output_dir / "y_pred_direction_test.npy", y_pred_dir_test)
    print(f"  Saved direction arrays to {output_dir}")

    # ==================================================================
    # Step 4: Classification metrics
    # ==================================================================
    print("\n[Step 4] Classification metrics...")

    cls_rows = []
    splits_cls = [("val", y_label_val, y_pred_dir_val, y_pred_val)]
    if has_test:
        splits_cls.append(("test", y_label_test, y_pred_dir_test, y_pred_test))

    print(f"  {'Split':>6s}  {'BalAcc':>8s}  {'AUC':>8s}  {'Sens-W':>8s}  {'PPV-W':>8s}")
    print(f"  {'-' * 6}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    for split_name, yl, ypd, ypc in splits_cls:
        metrics = compute_classification_metrics(yl, ypd, ypc)
        cls_rows.append({"split": split_name, **metrics})
        print(f"  {split_name:>6s}  {metrics['balanced_accuracy']:8.4f}  "
              f"{metrics['auc_ovr_macro']:8.4f}  "
              f"{metrics['sensitivity_worsening']:8.4f}  "
              f"{metrics['ppv_worsening']:8.4f}")

    cls_df = pd.DataFrame(cls_rows)
    cls_df.to_csv(output_dir / "classification_metrics.csv", index=False)
    print(f"  Saved: classification_metrics.csv")

    # ==================================================================
    # Step 5: Confusion matrices (aggregate)
    # ==================================================================
    print("\n[Step 5] Aggregate confusion matrices...")

    plot_confusion_matrix(
        y_label_val, y_pred_dir_val,
        f"{model_name}  -- Val ({label_type})",
        output_dir / "confusion_matrix_val.png",
    )
    print(f"  Saved: confusion_matrix_val.png")

    if has_test:
        plot_confusion_matrix(
            y_label_test, y_pred_dir_test,
            f"{model_name}  -- Test ({label_type})",
            output_dir / "confusion_matrix_test.png",
        )
        print(f"  Saved: confusion_matrix_test.png")

    # ==================================================================
    # Step 6: Per-person direction accuracy
    # ==================================================================
    print("\n[Step 6] Per-person direction accuracy...")

    per_person_val = compute_per_person_direction(y_label_val, y_pred_dir_val, pid_val)
    per_person_val.to_csv(output_dir / "direction_per_person_val.csv", index=False)
    median_acc_val = per_person_val["direction_accuracy"].median()
    print(f"  Val:  median={median_acc_val:.4f}  "
          f"range=[{per_person_val['direction_accuracy'].min():.4f}, "
          f"{per_person_val['direction_accuracy'].max():.4f}]")

    per_person_test = None
    if has_test:
        per_person_test = compute_per_person_direction(y_label_test, y_pred_dir_test, pid_test)
        per_person_test.to_csv(output_dir / "direction_per_person_test.csv", index=False)
        median_acc_test = per_person_test["direction_accuracy"].median()
        print(f"  Test: median={median_acc_test:.4f}  "
              f"range=[{per_person_test['direction_accuracy'].min():.4f}, "
              f"{per_person_test['direction_accuracy'].max():.4f}]")

    # ==================================================================
    # Step 7: Per-person plots (optional)
    # ==================================================================
    if not args.skip_plots:
        per_person_dir = output_dir / "plots" / "per_person"

        # Shared y-axis for trajectories
        all_y = [y_val, y_pred_val]
        if has_test:
            all_y += [y_test, y_pred_test]
        all_vals = np.concatenate(all_y)
        y_pad = (all_vals.max() - all_vals.min()) * 0.05
        shared_ylim = (all_vals.min() - y_pad, all_vals.max() + y_pad)

        print("\n[Step 7a] Per-person confusion matrices (val)...")
        n = plot_per_person_cms(y_label_val, y_pred_dir_val, pid_val,
                                per_person_dir / "val" / "confusion_matrices",
                                model_name, label_type)
        print(f"  Saved {n} files.")

        print("[Step 7b] Per-person trajectories (val)...")
        n = plot_per_person_trajectories(y_val, y_pred_val, pid_val,
                                          period_numbers_val, per_person_dir / "val" / "trajectories",
                                          model_name, shared_ylim)
        print(f"  Saved {n} files.")

        if has_test:
            print("[Step 7c] Per-person confusion matrices (test)...")
            n = plot_per_person_cms(y_label_test, y_pred_dir_test, pid_test,
                                    per_person_dir / "test" / "confusion_matrices",
                                    model_name, label_type)
            print(f"  Saved {n} files.")

            print("[Step 7d] Per-person trajectories (test)...")
            n = plot_per_person_trajectories(y_test, y_pred_test, pid_test,
                                              period_numbers_test, per_person_dir / "test" / "trajectories",
                                              model_name, shared_ylim)
            print(f"  Saved {n} files.")
    else:
        print("\n[Step 7] Skipping per-person plots (--skip-plots)")

    # ==================================================================
    # Step 8: Run summary
    # ==================================================================
    print("\n[Step 8] Saving run summary...")

    summary_lines = [
        "=" * 70,
        f"POST-HOC DIRECTION ANALYSIS: {model_name}",
        f"Label type: {label_type}",
        f"Run: {timestamp}",
        "=" * 70,
        "",
        "CONFIG",
        f"  Label type:  {label_type}",
        f"  Condition:   {condition}",
        f"  Data dir:    {data_dir}",
        f"  Labels dir:  {labels_path}",
        "",
    ]

    # Stratified error
    for split_name, yt, yp, yl in splits_to_eval:
        strat_df = compute_stratified_error(yt, yp, yl)
        summary_lines.append(f"STRATIFIED REGRESSION ERROR ({split_name.upper()})")
        summary_lines.append(f"  {'Direction':15s}  {'N':>5s}  {'MAE':>8s}  {'RMSE':>8s}  {'Bias':>8s}")
        for _, row in strat_df.iterrows():
            summary_lines.append(
                f"  {row['direction']:15s}  {row['N']:5.0f}  "
                f"{row['MAE']:8.3f}  {row['RMSE']:8.3f}  {row['Bias']:+8.3f}")
        summary_lines.append("")

    # Classification metrics
    summary_lines.append("CLASSIFICATION METRICS")
    summary_lines.append(f"  {'Split':>6s}  {'BalAcc':>8s}  {'AUC':>8s}  {'Sens-W':>8s}  {'PPV-W':>8s}")
    for _, row in cls_df.iterrows():
        summary_lines.append(
            f"  {row['split']:>6s}  {row['balanced_accuracy']:8.4f}  "
            f"{row['auc_ovr_macro']:8.4f}  "
            f"{row['sensitivity_worsening']:8.4f}  "
            f"{row['ppv_worsening']:8.4f}")
    summary_lines.append(f"  (BalAcc chance = 0.333, AUC chance = 0.500)")
    summary_lines.append("")

    # Direction accuracy
    summary_lines.append("DIRECTION ACCURACY (median per-person)")
    summary_lines.append(f"  Val:  {median_acc_val:.4f}")
    if per_person_test is not None:
        summary_lines.append(f"  Test: {per_person_test['direction_accuracy'].median():.4f}")
    summary_lines.append("=" * 70)

    summary_text = "\n".join(summary_lines)
    summary_path = output_dir / f"run_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"  Saved: {summary_path}")

    print("\n" + summary_text)
