"""Generate slide-ready figures and tables from ElasticNet condition results.

Reads the cross-condition comparison summary and per-condition outputs
produced by run_all_conditions.py. Generates figures and tables that
answer three research questions:

  RQ1: What feature conditions did we run?
  RQ2: What is the best condition?
  RQ3: For whom does the model work well vs. poorly?

Usage:
    python scripts/build_report.py
    python scripts/build_report.py --output-dir ../reports
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELASTICNET_DIR = _SCRIPT_DIR.parent
_POSTHOC_DIR = _ELASTICNET_DIR.parent / "posthoc"

# Highlight required conditions in plots
REQUIRED_CONDITIONS = {"prior_cesd", "base", "base_lag", "base_lag_pmcesd"}


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate slide-ready figures and tables from ElasticNet results",
    )
    parser.add_argument(
        "--models-dir", default=None,
        help="Base models directory (default: ../models)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Processed data directory (default: ../../data/processed)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for figures and tables (default: ../reports)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else (_ELASTICNET_DIR / "models")
    data_dir = Path(args.data_dir) if args.data_dir else (_ELASTICNET_DIR.parent.parent / "data" / "processed")
    output_dir = Path(args.output_dir) if args.output_dir else (_ELASTICNET_DIR / "reports")

    models_dir = models_dir.resolve()
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILDING ELASTICNET REPORT")
    print(f"Models dir: {models_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    # Load summary
    summary_path = models_dir / "comparison_summary.csv"
    if not summary_path.exists():
        print(f"Summary not found at {summary_path}. Run run_all_conditions.py first.")
        sys.exit(1)
    summary_df = pd.read_csv(summary_path)
    print(f"Loaded summary with {len(summary_df)} conditions.\n")

    best_condition = summary_df.loc[summary_df["mae_val"].idxmin(), "condition"]

    # ==================================================================
    # RQ1: What feature conditions did we run?
    # ==================================================================
    print("--- RQ1: What feature conditions did we run? ---")

    # Table 1: Feature condition inventory
    cols = ["condition", "group", "description", "n_features", "n_nonzero"]
    available = [c for c in cols if c in summary_df.columns]
    table1 = summary_df[available].copy()
    table1 = table1.sort_values("n_features").reset_index(drop=True)
    table1.to_csv(output_dir / "table1_feature_conditions.csv", index=False)
    print(f"  Saved table1_feature_conditions.csv")

    # ==================================================================
    # RQ2: What is the best condition?
    # ==================================================================
    print("\n--- RQ2: What is the best condition? ---")

    # Table 2: Cross-condition comparison
    cols = [
        "condition", "group", "n_features",
        "mae_train", "mae_val", "mae_test",
        "rmse_val", "rmse_test",
        "within_r2_val", "within_r2_test",
        "between_r2_val", "between_r2_test",
        "bal_acc_val", "bal_acc_test",
        "auc_ovr_val", "auc_ovr_test",
        "sens_w_val", "sens_w_test",
        "ppv_w_val", "ppv_w_test",
    ]
    available = [c for c in cols if c in summary_df.columns]
    table2 = summary_df[available].copy()
    table2.to_csv(output_dir / "table2_comparison.csv", index=False)
    print(f"  Saved table2_comparison.csv")

    # Figure 1: Grouped bar chart of MAE
    mae_cols = ["mae_train", "mae_val", "mae_test"]
    df_sorted = summary_df.sort_values("mae_val")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df_sorted))
    width = 0.25
    colors = {"mae_train": "#4C72B0", "mae_val": "#DD8452", "mae_test": "#55A868"}
    labels = {"mae_train": "Train*", "mae_val": "Val", "mae_test": "Test"}

    for i, col in enumerate(mae_cols):
        offset = (i - len(mae_cols) / 2 + 0.5) * width
        ax.bar(x + offset, df_sorted[col], width, label=labels[col],
               color=colors[col], edgecolor="white", linewidth=0.5)

    # Mark required conditions
    cond_list = df_sorted["condition"].tolist()
    for idx, cond in enumerate(cond_list):
        if cond in REQUIRED_CONDITIONS:
            ax.annotate("*", (idx, 0), fontsize=14, ha="center",
                        va="top", color="red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["condition"], rotation=45, ha="right")
    ax.set_ylabel("MAE (CESD delta points)")
    ax.set_title("ElasticNet MAE by Feature Condition")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_mae_comparison.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig1_mae_comparison.png")

    # Figure 2: Direction classification metrics (BalAcc + AUC)
    bacc_cols = ["bal_acc_val", "bal_acc_test"]
    auc_cols = ["auc_ovr_val", "auc_ovr_test"]
    df_sorted = summary_df.sort_values("bal_acc_val", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Balanced Accuracy
    ax = axes[0]
    x = np.arange(len(df_sorted))
    width = 0.35
    bar_colors = {"bal_acc_val": "#DD8452", "bal_acc_test": "#55A868"}
    bar_labels = {"bal_acc_val": "Val", "bal_acc_test": "Test"}
    for i, col in enumerate(bacc_cols):
        offset = (i - len(bacc_cols) / 2 + 0.5) * width
        ax.bar(x + offset, df_sorted[col], width, label=bar_labels[col],
               color=bar_colors[col], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["condition"], rotation=45, ha="right")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Direction: Balanced Accuracy")
    ax.legend()
    ax.set_ylim(0, 1)

    # AUC OvR
    ax = axes[1]
    auc_colors = {"auc_ovr_val": "#DD8452", "auc_ovr_test": "#55A868"}
    auc_labels = {"auc_ovr_val": "Val", "auc_ovr_test": "Test"}
    for i, col in enumerate(auc_cols):
        offset = (i - len(auc_cols) / 2 + 0.5) * width
        ax.bar(x + offset, df_sorted[col], width, label=auc_labels[col],
               color=auc_colors[col], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["condition"], rotation=45, ha="right")
    ax.set_ylabel("AUC (OvR macro)")
    ax.set_title("Direction: AUC OvR Macro")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_direction_metrics.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig2_direction_metrics.png")

    # Figure 3: Within-person R² comparison
    wr2_cols = ["within_r2_val", "within_r2_test"]
    df_sorted = summary_df.sort_values("within_r2_val", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df_sorted))
    width = 0.35
    wr2_colors = {"within_r2_val": "#DD8452", "within_r2_test": "#55A868"}
    wr2_labels = {"within_r2_val": "Val", "within_r2_test": "Test"}

    for i, col in enumerate(wr2_cols):
        offset = (i - len(wr2_cols) / 2 + 0.5) * width
        ax.bar(x + offset, df_sorted[col], width, label=wr2_labels[col],
               color=wr2_colors[col], edgecolor="white", linewidth=0.5)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["condition"], rotation=45, ha="right")
    ax.set_ylabel("Within-Person R²")
    ax.set_title("ElasticNet Within-Person R² by Feature Condition")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "fig3_within_r2_comparison.png", dpi=200)
    plt.close(fig)
    print(f"  Saved fig3_within_r2_comparison.png")

    # Table 3: Best condition's feature coefficients
    coef_path = models_dir / best_condition / "feature_coefficients.csv"
    if not coef_path.exists():
        print(f"  Skipping best coefficients (file not found: {coef_path})")
    else:
        coef_df = pd.read_csv(coef_path)
        coef_df = coef_df[coef_df["abs_coefficient"] > 1e-10].copy()
        coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
        coef_df.to_csv(output_dir / "table3_best_coefficients.csv", index=False)
        print(f"  Saved table3_best_coefficients.csv (condition={best_condition})")

    # Figure 5: Aggregate confusion matrix for best condition
    for label_type in ["sev_crossing"]:
        posthoc_dir = _POSTHOC_DIR / "elasticnet" / best_condition / label_type

        for split in ["test", "val"]:
            dir_true_path = posthoc_dir / f"y_labels_{split}.npy"
            dir_pred_path = posthoc_dir / f"y_pred_direction_{split}.npy"
            if not dir_true_path.exists() or not dir_pred_path.exists():
                continue

            y_true_dir = np.load(dir_true_path)
            y_pred_dir = np.load(dir_pred_path)

            cm = confusion_matrix(y_true_dir, y_pred_dir, labels=[0, 1, 2])
            class_names = ["Improving", "Stable", "Worsening"]

            annot = np.empty_like(cm, dtype=object)
            for i in range(3):
                row_sum = cm[i, :].sum()
                for j in range(3):
                    if row_sum == 0:
                        annot[i, j] = "0"
                    else:
                        pct = 100.0 * cm[i, j] / row_sum
                        annot[i, j] = f"{cm[i, j]}\n({pct:.0f}%)"

            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(
                cm, annot=annot, fmt="", ax=ax, cbar=True,
                cmap="Blues", xticklabels=class_names, yticklabels=class_names,
            )
            ax.set_xlabel("Predicted Direction")
            ax.set_ylabel("True Direction")
            n_total = cm.sum()
            acc = np.trace(cm) / n_total if n_total > 0 else 0
            ax.set_title(
                f"ElasticNet ({best_condition})  -- {split.title()} Set\n"
                f"Overall accuracy: {acc:.1%} (n={n_total})",
            )
            plt.tight_layout()
            fig.savefig(
                output_dir / f"fig5_confusion_matrix_{split}.png", dpi=200)
            plt.close(fig)
            print(f"  Saved fig5_confusion_matrix_{split}.png")

    # ==================================================================
    # RQ3: For whom does it work well vs. poorly?
    # ==================================================================
    print("\n--- RQ3: For whom does it work? ---")

    # Table 4: Performer tier characteristics
    for split in ["val", "test"]:
        analysis_path = (
            models_dir / best_condition
            / f"performer_analysis_{split}.csv"
        )
        if analysis_path.exists():
            analysis_df = pd.read_csv(analysis_path)
            analysis_df.to_csv(
                output_dir / f"table4_performer_analysis_{split}.csv",
                index=False,
            )
            print(f"  Saved table4_performer_analysis_{split}.csv")

        tier_stats_path = (
            models_dir / best_condition
            / f"performer_tier_stats_{split}.csv"
        )
        if tier_stats_path.exists():
            tier_stats = pd.read_csv(tier_stats_path)
            tier_stats.to_csv(
                output_dir / f"table4_tier_stats_{split}.csv", index=False)
            print(f"  Saved table4_tier_stats_{split}.csv")

    # Figure 4: Per-person MAE distribution by tier (box plot)
    tier_order = ["low", "medium", "high"]
    tier_colors = {"high": "#55A868", "medium": "#4C72B0", "low": "#C44E52"}

    for split in ["val", "test"]:
        tier_path = (
            models_dir / best_condition
            / f"performer_tiers_{split}.csv"
        )
        if not tier_path.exists():
            continue

        tier_df = pd.read_csv(tier_path)

        fig, ax = plt.subplots(figsize=(8, 5))
        data_by_tier = []
        tier_labels = []
        for tier in tier_order:
            d = tier_df[tier_df["tier"] == tier]["MAE"].dropna().values
            if len(d) > 0:
                data_by_tier.append(d)
                tier_labels.append(f"{tier}\n(n={len(d)})")

        bp = ax.boxplot(data_by_tier, patch_artist=True, widths=0.6)
        for patch, tier in zip(bp["boxes"], tier_order):
            patch.set_facecolor(tier_colors.get(tier, "lightblue"))
            patch.set_alpha(0.7)

        ax.set_xticklabels(tier_labels)
        ax.set_ylabel("Per-Person MAE")
        ax.set_title(
            f"ElasticNet ({best_condition})  -- {split.title()} Set\n"
            f"Per-Person MAE by Performance Tier"
        )
        plt.tight_layout()
        fig.savefig(output_dir / f"fig4_mae_by_tier_{split}.png", dpi=200)
        plt.close(fig)
        print(f"  Saved fig4_mae_by_tier_{split}.png")

    # Figure 6: Cherry-picked trajectory plots (high vs low performers)
    for split in ["val", "test"]:
        tier_path = (
            models_dir / best_condition
            / f"performer_tiers_{split}.csv"
        )
        # Trajectories come from posthoc per-person plots
        traj_dir = (
            _POSTHOC_DIR / "elasticnet" / best_condition
            / "sev_crossing" / "plots" / "per_person"
            / split / "trajectories"
        )
        if not tier_path.exists() or not traj_dir.exists():
            continue

        tier_df = pd.read_csv(tier_path)
        high = tier_df[tier_df["tier"] == "high"].nsmallest(3, "MAE")
        low = tier_df[tier_df["tier"] == "low"].nlargest(3, "MAE")

        try:
            from PIL import Image
        except ImportError:
            print("  Skipping trajectory montage (PIL not available)")
            break

        selected = pd.concat([high, low])
        if len(selected) == 0:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 8))

        for idx, (_, row) in enumerate(high.iterrows()):
            pid = int(row["person_id"])
            img_path = traj_dir / f"pid_{pid}_trajectory.png"
            ax = axes[0, idx] if idx < 3 else axes[0, 2]
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            ax.set_title(
                f"HIGH: PID {pid} (MAE={row['MAE']:.2f})", fontsize=9)
            ax.axis("off")

        for idx, (_, row) in enumerate(low.iterrows()):
            pid = int(row["person_id"])
            img_path = traj_dir / f"pid_{pid}_trajectory.png"
            ax = axes[1, idx] if idx < 3 else axes[1, 2]
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            ax.set_title(
                f"LOW: PID {pid} (MAE={row['MAE']:.2f})", fontsize=9)
            ax.axis("off")

        for r in range(2):
            for c in range(3):
                count = len(high) if r == 0 else len(low)
                if c >= count:
                    axes[r, c].set_visible(False)

        fig.suptitle(
            f"Example Trajectories: High vs Low Performers ({split.title()})",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(
            output_dir / f"fig6_example_trajectories_{split}.png", dpi=200)
        plt.close(fig)
        print(f"  Saved fig6_example_trajectories_{split}.png")

    print("\n" + "=" * 70)
    print("REPORT COMPLETE")
    print(f"All outputs in: {output_dir}")
    print("=" * 70)
