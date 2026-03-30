#!/usr/bin/env python3
"""Post-hoc direction analysis for MixedLM regression predictions.

Uses classification labels from ../classification/labels/ to evaluate
how well regression predictions capture the direction of symptom change.

Reports:
- Stratified regression error by direction class (improving/stable/worsening)
- Classification metrics: BalAcc, AUC (OvR), Sens-W, PPV-W, confusion matrix
- Comparison across ablation conditions and label types
- Plots: confusion matrices, pred-vs-actual, residuals, trajectories

Usage:
    # Run posthoc for all models in regression/mixedlm/models/:
    python regression/mixedlm/scripts/posthoc_mixedlm.py

    # Single model directory:
    python regression/mixedlm/scripts/posthoc_mixedlm.py --model-dir regression/mixedlm/models/base

    # Use all three label types:
    python regression/mixedlm/scripts/posthoc_mixedlm.py --label-types sev_crossing personal_sd balanced_tercile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Allow imports from scripts directory
sys.path.insert(0, str(Path(__file__).parent))

from metrics import (
    compute_aggregate_metrics,
    compute_baselines,
    compute_train_baselines,
    baselines_to_dict,
    compute_per_person_metrics,
)

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
LABELS_DIR = REPO_ROOT / "classification" / "labels"
MODELS_BASE = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


# ---------------------------------------------------------------------------
# Stratified regression error by direction class
# ---------------------------------------------------------------------------

def stratified_regression_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_labels: np.ndarray,
    class_names: list[str] = ("improving", "stable", "worsening"),
) -> pd.DataFrame:
    """Compute MAE, RMSE, and bias stratified by direction class."""
    rows = []
    for cls_idx, name in enumerate(class_names):
        mask = y_labels == cls_idx
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        rows.append({
            "Direction": name,
            "N": int(mask.sum()),
            "MAE": float(np.mean(np.abs(yt - yp))),
            "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "Bias": float(np.mean(yp - yt)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Posthoc classification metrics
# ---------------------------------------------------------------------------

def _pred_direction_from_regression(
    y_pred: np.ndarray,
    n_classes: int = 3,
) -> np.ndarray:
    """Convert continuous regression predictions to 3-class direction labels.

    0 = improving (y_pred < 0), 1 = stable (y_pred == 0), 2 = worsening (y_pred > 0).

    Uses simple sign-based thresholding to be consistent with the classification
    label definitions where improving = decrease, worsening = increase.
    """
    # For sev_crossing labels: 0=improving, 1=stable, 2=worsening
    # We map: negative delta -> improving, ~zero -> stable, positive -> worsening
    pred_labels = np.ones(len(y_pred), dtype=int)  # default stable
    pred_labels[y_pred < 0] = 0   # improving
    pred_labels[y_pred > 0] = 2   # worsening
    return pred_labels


def posthoc_classification_metrics(
    y_true_labels: np.ndarray,
    y_pred_continuous: np.ndarray,
    class_names: list[str] = ("improving", "stable", "worsening"),
) -> dict:
    """Compute classification metrics treating regression direction as a classifier.

    Returns dict with BalAcc, AUC (OvR), Sens-W, PPV-W, confusion matrix, etc.
    """
    y_pred_labels = _pred_direction_from_regression(y_pred_continuous)
    n_classes = len(class_names)

    bal_acc = float(balanced_accuracy_score(y_true_labels, y_pred_labels))

    # AUC: use absolute predicted value as a "confidence" score
    # Create soft scores for OvR AUC
    try:
        # Build pseudo-probabilities from continuous predictions
        proba = np.zeros((len(y_pred_continuous), n_classes))
        # Improving: more negative = more confident
        proba[:, 0] = np.clip(-y_pred_continuous, 0, None)
        # Stable: closer to zero = more confident
        proba[:, 1] = np.clip(1 - np.abs(y_pred_continuous) / (np.std(y_pred_continuous) + 1e-8), 0, None)
        # Worsening: more positive = more confident
        proba[:, 2] = np.clip(y_pred_continuous, 0, None)
        # Normalize rows
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        proba = proba / row_sums

        auc_ovr = float(roc_auc_score(
            y_true_labels, proba, multi_class="ovr", average="macro",
        ))
    except Exception:
        auc_ovr = float("nan")

    # Per-class precision, recall, F1
    p, r, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, labels=list(range(n_classes)), zero_division=0,
    )

    # Worsening class = index 2
    sens_w = float(r[2])  # recall for worsening
    ppv_w = float(p[2])   # precision for worsening

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=list(range(n_classes)))

    return {
        "BalAcc": bal_acc,
        "AUC_OvR": auc_ovr,
        "Sens_W": sens_w,
        "PPV_W": ppv_w,
        "F1_macro": float(f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)),
        "confusion_matrix": cm,
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Direction",
        xlabel="Predicted Direction",
    )

    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        row_total = cm[i].sum()
        for j in range(len(class_names)):
            pct = cm[i, j] / row_total * 100 if row_total > 0 else 0
            ax.text(
                j, i, f"{cm[i, j]}\n({pct:.0f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    person_ids: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Scatter of predicted vs actual with identity line."""
    fig, ax = plt.subplots(figsize=(8, 8))

    unique_pids = np.unique(person_ids)
    cmap = plt.cm.tab20(np.linspace(0, 1, min(len(unique_pids), 20)))
    pid_to_color = {pid: cmap[i % len(cmap)] for i, pid in enumerate(unique_pids)}
    colors = [pid_to_color[pid] for pid in person_ids]

    ax.scatter(y_true, y_pred, c=colors, alpha=0.5, edgecolor="black", linewidth=0.3, s=30)

    all_vals = np.concatenate([y_true, y_pred])
    vmin, vmax = all_vals.min(), all_vals.max()
    margin = (vmax - vmin) * 0.05
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
            "r--", linewidth=1.5, label="y = x")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.mean(np.abs(y_true - y_pred))

    ax.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3f}\nn = {len(y_true)}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Residual vs predicted diagnostic plot."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(y_pred, residuals, alpha=0.4, edgecolor="black", linewidth=0.3, s=25)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5)

    sort_idx = np.argsort(y_pred)
    window = max(len(y_pred) // 20, 5)
    smoothed = pd.Series(residuals[sort_idx]).rolling(window, center=True, min_periods=1).mean()
    ax.plot(y_pred[sort_idx], smoothed, color="orange", linewidth=2, label="Smoothed trend")

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Residual (Actual - Predicted)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_person_trajectories(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    person_ids: np.ndarray,
    output_path: Path,
    max_persons: int = 20,
    ncols: int = 4,
) -> None:
    """Plot actual vs predicted trajectories per person."""
    df = pd.DataFrame({
        "y_true": y_true, "y_pred": y_pred, "pid": person_ids,
    })
    df["idx"] = range(len(df))

    # Compute direction accuracy per person
    def _dir_acc(g):
        yt = np.where(g["y_true"] > 0, 1, np.where(g["y_true"] < 0, -1, 0))
        yp = np.where(g["y_pred"] > 0, 1, np.where(g["y_pred"] < 0, -1, 0))
        return np.mean(yt == yp)

    pid_acc = df.groupby("pid").apply(_dir_acc, include_groups=False).sort_values(ascending=False)
    unique_pids = pid_acc.index.tolist()

    if len(unique_pids) > max_persons:
        n_each = max_persons // 3
        selected = unique_pids[:n_each] + unique_pids[len(unique_pids)//2 - n_each//2 : len(unique_pids)//2 + n_each//2] + unique_pids[-n_each:]
    else:
        selected = unique_pids

    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, pid in enumerate(selected):
        pdata = df[df["pid"] == pid].sort_values("idx")
        ax = axes[idx]
        ax.plot(range(len(pdata)), pdata["y_true"].values, "o-", color="#4c72b0",
                markersize=4, linewidth=1.5, label="Actual")
        ax.plot(range(len(pdata)), pdata["y_pred"].values, "s--", color="#c44e52",
                markersize=4, linewidth=1.5, label="Predicted")
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
        dacc = pid_acc.loc[pid]
        ax.set_title(f"P{pid} (dir_acc={dacc:.0%})", fontsize=9)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(len(selected), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Actual vs Predicted Trajectories", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main posthoc analysis
# ---------------------------------------------------------------------------

def run_posthoc_for_model(
    model_dir: Path,
    label_type: str,
    output_dir: Path,
) -> pd.DataFrame | None:
    """Run full posthoc analysis for one model directory and one label type.

    Returns summary DataFrame or None if predictions not found.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_dir.name

    # Load classification labels
    label_dir = LABELS_DIR / label_type
    if not label_dir.exists():
        print(f"  Warning: label directory {label_dir} not found, skipping")
        return None

    all_results = []

    for split in ["train", "val", "test"]:
        pred_path = model_dir / f"y_pred_{split}.npy"
        true_path = DATA_DIR / f"y_{split}.npy"
        label_path = label_dir / f"y_{split}.npy"
        pid_path = DATA_DIR / f"pid_{split}.npy"

        if not pred_path.exists():
            print(f"  Warning: {pred_path} not found, skipping {split}")
            continue

        y_pred = np.load(pred_path)
        y_true = np.load(true_path)
        y_labels = np.load(label_path)
        pid = np.load(pid_path)

        # Ensure consistent lengths
        min_len = min(len(y_pred), len(y_true), len(y_labels), len(pid))
        y_pred, y_true, y_labels, pid = y_pred[:min_len], y_true[:min_len], y_labels[:min_len], pid[:min_len]

        prefix = f"{split}_{label_type}"

        # 1. Stratified regression error
        strat_df = stratified_regression_error(y_true, y_pred, y_labels)
        strat_df.to_csv(output_dir / f"{prefix}_stratified_error.csv", index=False)

        # 2. Classification metrics
        cls_metrics = posthoc_classification_metrics(y_true_labels=y_labels, y_pred_continuous=y_pred)

        # Save confusion matrix plot
        cm = cls_metrics["confusion_matrix"]
        plot_confusion_matrix(
            cm, class_names=["improving", "stable", "worsening"],
            title=f"{model_name} | {split} | {label_type}\nBalAcc={cls_metrics['BalAcc']:.3f}",
            output_path=output_dir / f"{prefix}_confusion_matrix.png",
        )

        # Save classification metrics
        cls_row = {
            "model": model_name,
            "split": split,
            "label_type": label_type,
            "BalAcc": cls_metrics["BalAcc"],
            "AUC_OvR": cls_metrics["AUC_OvR"],
            "Sens_W": cls_metrics["Sens_W"],
            "PPV_W": cls_metrics["PPV_W"],
            "F1_macro": cls_metrics["F1_macro"],
        }
        all_results.append(cls_row)

        # 3. Plots (test split only to avoid clutter)
        if split == "test":
            plot_pred_vs_actual(
                y_true, y_pred, pid,
                title=f"{model_name} - Pred vs Actual (Test)",
                output_path=output_dir / f"test_pred_vs_actual.png",
            )
            plot_residual_vs_predicted(
                y_true, y_pred,
                title=f"{model_name} - Residual vs Predicted (Test)",
                output_path=output_dir / f"test_residual_vs_predicted.png",
            )
            plot_person_trajectories(
                y_true, y_pred, pid,
                output_path=output_dir / f"test_person_trajectories.png",
            )

        # Print summary
        print(f"  {split}/{label_type}: BalAcc={cls_metrics['BalAcc']:.3f}  "
              f"Sens-W={cls_metrics['Sens_W']:.3f}  PPV-W={cls_metrics['PPV_W']:.3f}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / f"posthoc_classification_{label_type}.csv", index=False)
        return results_df

    return None


def write_summary_report(
    all_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a markdown summary report combining all models and label types."""
    lines = [
        "# MixedLM Regression: Post-Hoc Direction Analysis",
        "",
        "## Classification Metrics (Regression Predictions as Direction Classifier)",
        "",
    ]

    for label_type in all_results["label_type"].unique():
        lt_df = all_results[all_results["label_type"] == label_type]
        test_df = lt_df[lt_df["split"] == "test"]

        lines.append(f"### Label type: `{label_type}`")
        lines.append("")
        lines.append("| Model | BalAcc | AUC (OvR) | Sens-W | PPV-W | F1 macro |")
        lines.append("|-------|--------|-----------|--------|-------|----------|")

        for _, row in test_df.iterrows():
            lines.append(
                f"| {row['model']} | {row['BalAcc']:.3f} | "
                f"{row['AUC_OvR']:.3f} | {row['Sens_W']:.3f} | "
                f"{row['PPV_W']:.3f} | {row['F1_macro']:.3f} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"\nSaved summary report to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-hoc direction analysis for MixedLM regression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Single model directory (default: run all in regression/mixedlm/models/)",
    )
    parser.add_argument(
        "--label-types", nargs="+", default=["sev_crossing"],
        help="Label types to use (default: sev_crossing)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: regression/mixedlm/reports/posthoc/)",
    )

    args = parser.parse_args()

    output_base = Path(args.output_dir) if args.output_dir else REPORTS_DIR / "posthoc"
    output_base.mkdir(parents=True, exist_ok=True)

    # Determine model directories to analyze
    if args.model_dir:
        model_dirs = [Path(args.model_dir)]
    else:
        model_dirs = sorted([
            d for d in MODELS_BASE.iterdir()
            if d.is_dir() and (d / "y_pred_test.npy").exists()
        ])

    if not model_dirs:
        print("No model directories with predictions found.")
        return 1

    print(f"Found {len(model_dirs)} model(s) to analyze")
    print(f"Label types: {args.label_types}")

    all_results = []

    for model_dir in model_dirs:
        print(f"\n{'='*60}")
        print(f"Model: {model_dir.name}")
        print(f"{'='*60}")

        for label_type in args.label_types:
            sub_output = output_base / model_dir.name
            results = run_posthoc_for_model(model_dir, label_type, sub_output)
            if results is not None:
                all_results.append(results)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(output_base / "all_posthoc_results.csv", index=False)
        write_summary_report(combined, output_base / "posthoc_summary.md")

        # Print final comparison
        test_results = combined[combined["split"] == "test"]
        if not test_results.empty:
            print(f"\n{'='*80}")
            print("POSTHOC DIRECTION ANALYSIS SUMMARY (Test Set)")
            print(f"{'='*80}")
            print(test_results.to_string(index=False))

    print(f"\nAll outputs saved to {output_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
