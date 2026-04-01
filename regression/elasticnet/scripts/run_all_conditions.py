"""Run ElasticNet regression across all feature-set conditions.

Overview
--------
Top-level orchestrator for the ElasticNet feature-condition comparison.
Loops over 11 conditions (4 required ablation + 7 extra feature-set
variants), running the full pipeline for each, then collects results into
a single cross-condition summary CSV.

Conditions
----------
Required (parity with classification):
    prior_cesd        -- 1 feature (prior CES-D only)
    base              -- 21 base features
    base_lag          -- 21 base + 17 behavioral lag = 38 columns
    base_lag_pmcesd   -- 38 + person_mean_cesd = 39 columns

Extra (original screenome variants):
    dev               -- 21 base + 8 within-person deviation
    pheno             -- 21 base + 5 phenotype
    pid               -- 21 base + ~96 PID one-hot
    dev_pheno         -- base + dev + pheno
    dev_pid           -- base + dev + PID OHE
    pheno_pid         -- base + pheno + PID OHE
    dev_pheno_pid     -- base + dev + pheno + PID OHE

Per-Condition Pipeline
----------------------
Step 1  -- train_elasticnet.py   (grid search -> dev model -> final model)
Step 2  -- posthoc_direction.py  (direction classification using labels)
Step 3  -- performer tier analysis (in-process, not subprocess)

Cross-Condition Summary
-----------------------
After all conditions finish, build_summary() produces:
    models/comparison_summary.csv

Usage
-----
    python scripts/run_all_conditions.py                  # all 11 conditions
    python scripts/run_all_conditions.py --dry-run        # show commands only
    python scripts/run_all_conditions.py --only base      # single condition
    python scripts/run_all_conditions.py --summary-only   # rebuild summary
    python scripts/run_all_conditions.py --skip-plots     # no per-person plots
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELASTICNET_DIR = _SCRIPT_DIR.parent
_POSTHOC_DIR = _ELASTICNET_DIR.parent / "posthoc"


# ======================================================================
# Condition definitions
# ======================================================================

CONDITIONS = {
    # Required 4 (from regression/README.md)
    "prior_cesd": {
        "description": "prior_cesd only (1 feature)",
        "group": "required",
    },
    "base": {
        "description": "All 21 base features",
        "group": "required",
    },
    "base_lag": {
        "description": "Base + 17 behavioral lag (38 cols)",
        "group": "required",
    },
    "base_lag_pmcesd": {
        "description": "Base + lag + person_mean_cesd (39 cols)",
        "group": "required",
    },
    # Extra 7 (original screenome variants)
    "dev": {
        "description": "Base + 8 within-person deviation features",
        "group": "extra",
    },
    "pheno": {
        "description": "Base + 5 phenotype features",
        "group": "extra",
    },
    "pid": {
        "description": "Base + ~96 PID one-hot features",
        "group": "extra",
    },
    "dev_pheno": {
        "description": "Base + dev + pheno",
        "group": "extra",
    },
    "dev_pid": {
        "description": "Base + dev + PID OHE",
        "group": "extra",
    },
    "pheno_pid": {
        "description": "Base + pheno + PID OHE",
        "group": "extra",
    },
    "dev_pheno_pid": {
        "description": "Base + dev + pheno + PID OHE",
        "group": "extra",
    },
}


def run_cmd(cmd, dry_run=False):
    """Run a subprocess command, returning True on success."""
    cmd_str = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"  [DRY RUN] {cmd_str}")
        return True

    print(f"  Running: {cmd_str[:140]}...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: Command failed with return code {result.returncode}")
        return False
    return True


# ======================================================================
# Performer tier analysis
# ======================================================================

def run_performer_analysis(condition_name, output_dir, data_dir):
    """Run performer tier analysis for a single condition.

    Loads y_true, y_pred, pids for val (and test if available), computes
    per-person MAE, classifies into high/medium/low tiers by MAE
    percentile (25th/75th), and saves CSVs.
    """
    print("\n--- Performer tier analysis ---")

    train_df = pd.read_csv(data_dir / "train_scaled.csv")

    # Aggregate to person level: use non-delta numeric columns
    _metadata = {"pid", "period_number", "row_name", "split",
                 "target_cesd_delta", "gender_mode"}
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    level_cols = [c for c in numeric_cols
                  if c not in _metadata and "_delta" not in c
                  and not c.startswith("gender_mode_")]
    agg_dict = {c: "mean" for c in level_cols}
    for gc in train_df.columns:
        if gc.startswith("gender_mode_") and gc in numeric_cols:
            agg_dict[gc] = "first"
    person_features = train_df.groupby("pid").agg(agg_dict).reset_index()
    person_features = person_features.rename(columns={"pid": "person_id"})

    # Merge phenotype assignments if available
    pheno_path = data_dir / "phenotype_assignments.csv"
    if pheno_path.exists():
        pheno_df = pd.read_csv(pheno_path)
        pheno_df = pheno_df.rename(columns={"pid": "person_id"})
        person_features = person_features.merge(
            pheno_df, on="person_id", how="left"
        )

    # Add n_train_obs per person
    n_train = train_df.groupby("pid").size().reset_index(name="n_train_obs")
    n_train = n_train.rename(columns={"pid": "person_id"})
    person_features = person_features.merge(n_train, on="person_id", how="left")

    for split in ["val", "test"]:
        y_pred_path = output_dir / f"y_pred_{split}.npy"
        if not y_pred_path.exists():
            continue

        y_true = np.load(data_dir / f"y_{split}.npy")
        y_pred = np.load(y_pred_path)
        pids = np.load(data_dir / f"pid_{split}.npy")

        # -- Per-person metrics --
        pred_df = pd.DataFrame({
            "y_true": y_true, "y_pred": y_pred, "person_id": pids,
        })

        def _stats(g):
            errors = g["y_true"] - g["y_pred"]
            return pd.Series({
                "MAE": float(np.abs(errors).mean()),
                "RMSE": float(np.sqrt(np.mean(errors.values ** 2))),
                "n_samples": len(g),
            })

        per_person_df = pred_df.groupby("person_id").apply(
            _stats, include_groups=False
        ).reset_index()

        # -- Classify into tiers by MAE percentile --
        mae_vals = per_person_df["MAE"].values
        lo_thresh = float(np.percentile(mae_vals, 25))
        hi_thresh = float(np.percentile(mae_vals, 75))

        pid_arr = per_person_df["person_id"].values
        high_mask = mae_vals <= lo_thresh   # high performers = low MAE
        low_mask = mae_vals >= hi_thresh    # low performers = high MAE

        tier_df = per_person_df.copy()
        tier_df["tier"] = "medium"
        tier_df.loc[high_mask, "tier"] = "high"
        tier_df.loc[low_mask, "tier"] = "low"

        # -- Tier statistics --
        tier_stats_rows = []
        for tier_name in ["high", "medium", "low"]:
            subset = tier_df[tier_df["tier"] == tier_name]
            if len(subset) == 0:
                continue
            tier_stats_rows.append({
                "tier": tier_name,
                "n_persons": len(subset),
                "mae_mean": float(subset["MAE"].mean()),
                "mae_std": float(subset["MAE"].std()),
                "mae_median": float(subset["MAE"].median()),
                "rmse_mean": float(subset["RMSE"].mean()),
            })
        tier_stats = pd.DataFrame(tier_stats_rows)

        # -- Extreme tier comparison --
        high_mae_ids = set(pid_arr[low_mask])   # low performers = high MAE
        low_mae_ids = set(pid_arr[high_mask])   # high performers = low MAE

        analysis_rows = []
        for col in ["MAE", "RMSE"]:
            hi = tier_df[tier_df["person_id"].isin(high_mae_ids)]
            lo = tier_df[tier_df["person_id"].isin(low_mae_ids)]
            analysis_rows.append({
                "feature": col,
                "high_mae_mean": float(hi[col].mean()),
                "high_mae_std": float(hi[col].std()),
                "low_mae_mean": float(lo[col].mean()),
                "low_mae_std": float(lo[col].std()),
            })

        hi_feat = person_features[person_features["person_id"].isin(high_mae_ids)]
        lo_feat = person_features[person_features["person_id"].isin(low_mae_ids)]
        feat_numeric_cols = person_features.select_dtypes(include=[np.number]).columns
        for col in feat_numeric_cols:
            if col == "person_id":
                continue
            analysis_rows.append({
                "feature": col,
                "high_mae_mean": float(hi_feat[col].mean()),
                "high_mae_std": float(hi_feat[col].std()),
                "low_mae_mean": float(lo_feat[col].mean()),
                "low_mae_std": float(lo_feat[col].std()),
            })
        analysis = pd.DataFrame(analysis_rows)

        # -- Save --
        tier_df.to_csv(
            output_dir / f"performer_tiers_{split}.csv", index=False
        )
        tier_stats.to_csv(
            output_dir / f"performer_tier_stats_{split}.csv", index=False
        )
        analysis.to_csv(
            output_dir / f"performer_analysis_{split}.csv", index=False
        )

        n_hi = int(high_mask.sum())
        n_med = int((~high_mask & ~low_mask).sum())
        n_lo = int(low_mask.sum())
        print(f"  [{split}] High={n_hi}, Med={n_med}, Low={n_lo}")


# ======================================================================
# Cross-condition summary builder
# ======================================================================

def build_summary(models_dir, data_dir):
    """Build cross-condition comparison CSV from existing outputs.

    Reads best_params.yaml and final_params.yaml for metrics, plus
    posthoc classification metrics and feature coefficients.
    """
    rows = []
    for cond_name, cond_cfg in CONDITIONS.items():
        cond_dir = models_dir / cond_name
        if not cond_dir.exists():
            print(f"  Skipping {cond_name} (no output dir)")
            continue

        row = {
            "condition": cond_name,
            "description": cond_cfg["description"],
            "group": cond_cfg["group"],
        }

        # ---- Metrics from best_params.yaml (dev model) ----
        params_path = cond_dir / "best_params.yaml"
        if params_path.exists():
            with open(params_path) as f:
                params = yaml.safe_load(f)
            row["best_alpha"] = params.get("alpha")
            row["best_l1_ratio"] = params.get("l1_ratio")
            row["n_features"] = params.get("n_features")

            metrics = params.get("metrics", {})
            for split in ["train", "val"]:
                sm = metrics.get(split, {})
                row[f"mae_{split}"] = sm.get("mae")
                row[f"rmse_{split}"] = sm.get("rmse")
                row[f"within_r2_{split}"] = sm.get("within_r2")
                row[f"between_r2_{split}"] = sm.get("between_r2")

        # ---- Test metrics from final_params.yaml ----
        final_path = cond_dir / "final_params.yaml"
        if final_path.exists():
            with open(final_path) as f:
                fparams = yaml.safe_load(f)
            fmetrics = fparams.get("metrics", {})
            sm = fmetrics.get("test", {})
            row["mae_test"] = sm.get("mae")
            row["rmse_test"] = sm.get("rmse")
            row["within_r2_test"] = sm.get("within_r2")
            row["between_r2_test"] = sm.get("between_r2")

        # ---- Posthoc classification metrics (all label types) ----
        for label_type in ["sev_crossing", "personal_sd", "balanced_tercile"]:
            cls_path = (_POSTHOC_DIR / "elasticnet" / cond_name / label_type
                        / "classification_metrics.csv")
            if cls_path.exists():
                cls_df = pd.read_csv(cls_path)
                prefix = f"{label_type}_" if label_type != "sev_crossing" else ""
                for _, cls_row in cls_df.iterrows():
                    split = cls_row.get("split", "")
                    row[f"{prefix}bal_acc_{split}"] = cls_row.get("balanced_accuracy")
                    row[f"{prefix}auc_ovr_{split}"] = cls_row.get("auc_ovr_macro")
                    row[f"{prefix}sens_w_{split}"] = cls_row.get("sensitivity_worsening")
                    row[f"{prefix}ppv_w_{split}"] = cls_row.get("ppv_worsening")

        # ---- N features / n_nonzero from coefficients ----
        coef_path = cond_dir / "feature_coefficients.csv"
        if coef_path.exists():
            coef_df = pd.read_csv(coef_path)
            row["n_features"] = len(coef_df)
            row["n_nonzero"] = int(
                (coef_df["abs_coefficient"] > 1e-10).sum()
            )

        rows.append(row)

    if not rows:
        print("  No condition outputs found!")
        return None

    summary_df = pd.DataFrame(rows)
    if "mae_val" in summary_df.columns:
        summary_df = summary_df.sort_values("mae_val")

    summary_path = models_dir / "comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")
    return summary_df


def print_best_summary(summary_df):
    """Print formatted summary of the best condition."""
    best = summary_df.iloc[0]  # already sorted by mae_val

    print("\n" + "=" * 70)
    print("BEST CONDITION SUMMARY")
    print("=" * 70)
    print(f"  Condition:     {best['condition']}  ({best['description']})")
    print(f"  Hyperparams:   alpha={best.get('best_alpha', '?')}, "
          f"l1_ratio={best.get('best_l1_ratio', '?')}")
    print(f"  Features:      {int(best.get('n_features', 0))} total, "
          f"{int(best.get('n_nonzero', 0))} non-zero")

    # Regression metrics table
    print(f"\n  {'Metric':<25s}  {'Train*':>8s}  {'Val':>8s}  {'Test':>8s}")
    print(f"  {'-' * 25}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for metric, label in [
        ("mae", "MAE (CESD pts)"),
        ("rmse", "RMSE (CESD pts)"),
        ("within_r2", "Within-person R²"),
        ("between_r2", "Between-person R²"),
    ]:
        tr = best.get(f"{metric}_train", float("nan"))
        vl = best.get(f"{metric}_val", float("nan"))
        te = best.get(f"{metric}_test", float("nan"))
        tr = tr if tr is not None else float("nan")
        vl = vl if vl is not None else float("nan")
        te = te if te is not None else float("nan")
        print(f"  {label:<25s}  {tr:8.3f}  {vl:8.3f}  {te:8.3f}")

    print(f"\n  * Train metrics are in-sample (diagnostic only).")

    # Direction classification metrics
    for metric, label in [
        ("bal_acc", "Balanced Accuracy"),
        ("auc_ovr", "AUC (OvR macro)"),
        ("sens_w", "Sensitivity (Worsening)"),
        ("ppv_w", "PPV (Worsening)"),
    ]:
        vl = best.get(f"{metric}_val", float("nan"))
        te = best.get(f"{metric}_test", float("nan"))
        vl = vl if vl is not None else float("nan")
        te = te if te is not None else float("nan")
        print(f"  {label:<25s}  {'':>8s}  {vl:8.3f}  {te:8.3f}")

    # Runner-up
    if len(summary_df) > 1:
        runner = summary_df.iloc[1]
        gap = runner["mae_val"] - best["mae_val"]
        print(f"\n  Runner-up:     {runner['condition']}  "
              f"(val MAE {runner['mae_val']:.3f}, +{gap:.3f})")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all ElasticNet feature-set conditions",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Processed data directory (default: ../../data/processed)",
    )
    parser.add_argument(
        "--models-dir", default=None,
        help="Base models directory (default: ../models)",
    )
    parser.add_argument(
        "--labels-dir", default=None,
        help="Classification labels directory (default: ../../classification/labels)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--only", default=None,
        help="Run only the specified condition (e.g., --only base)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip training, just build summary from existing outputs",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip posthoc per-person plots (faster for iteration)",
    )
    parser.add_argument(
        "--label-types", nargs="+",
        default=["sev_crossing", "personal_sd", "balanced_tercile"],
        help="Label types for posthoc direction analysis (default: sev_crossing personal_sd balanced_tercile)",
    )
    args = parser.parse_args()

    # Resolve defaults relative to this script's location
    data_dir = Path(args.data_dir) if args.data_dir else (_ELASTICNET_DIR.parent.parent / "data" / "processed")
    models_dir = Path(args.models_dir) if args.models_dir else (_ELASTICNET_DIR / "models")
    labels_dir = Path(args.labels_dir) if args.labels_dir else (_ELASTICNET_DIR.parent.parent / "classification" / "labels")

    data_dir = data_dir.resolve()
    models_dir = models_dir.resolve()
    labels_dir = labels_dir.resolve()

    print(f"Data dir:   {data_dir}")
    print(f"Models dir: {models_dir}")
    print(f"Labels dir: {labels_dir}")

    # Filter conditions
    conditions_to_run = CONDITIONS
    if args.only:
        if args.only not in CONDITIONS:
            print(f"Unknown condition: {args.only}")
            print(f"Available: {', '.join(CONDITIONS.keys())}")
            sys.exit(1)
        conditions_to_run = {args.only: CONDITIONS[args.only]}

    # ==================================================================
    # Run pipeline per condition
    # ==================================================================
    if not args.summary_only:
        for cond_name, cond_cfg in conditions_to_run.items():
            output_dir = models_dir / cond_name

            print("\n" + "=" * 70)
            print(f"CONDITION: {cond_name}")
            print(f"  {cond_cfg['description']}  [{cond_cfg['group']}]")
            print(f"  Output: {output_dir}")
            print("=" * 70)

            # ----------------------------------------------------------
            # Step 1: Train
            # ----------------------------------------------------------
            print("\n--- Step 1: Training ---")
            train_cmd = [
                sys.executable,
                str(_SCRIPT_DIR / "train_elasticnet.py"),
                "--data-dir", str(data_dir),
                "--condition", cond_name,
                "--output-dir", str(output_dir),
                "--run-test",
            ]
            if args.skip_plots:
                train_cmd.append("--skip-plots")

            if not run_cmd(train_cmd, args.dry_run):
                print(f"  Skipping remaining steps for {cond_name}")
                continue

            # ----------------------------------------------------------
            # Step 2: Post-hoc direction classification
            # ----------------------------------------------------------
            for label_type in args.label_types:
                print(f"\n--- Step 2: Posthoc direction ({label_type}) ---")
                posthoc_output = _POSTHOC_DIR / "elasticnet" / cond_name / label_type
                posthoc_cmd = [
                    sys.executable,
                    str(_SCRIPT_DIR / "posthoc_direction.py"),
                    "--data-dir", str(data_dir),
                    "--labels-dir", str(labels_dir),
                    "--condition", cond_name,
                    "--models-dir", str(models_dir),
                    "--label-type", label_type,
                    "--model-name", f"ElasticNet ({cond_name})",
                    "--output-dir", str(posthoc_output),
                ]
                if args.skip_plots:
                    posthoc_cmd.append("--skip-plots")

                run_cmd(posthoc_cmd, args.dry_run)

            # ----------------------------------------------------------
            # Step 3: Performer tier analysis (in-process)
            # ----------------------------------------------------------
            if not args.dry_run:
                run_performer_analysis(cond_name, output_dir, data_dir)

    # ==================================================================
    # Build cross-condition summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("BUILDING CROSS-CONDITION SUMMARY")
    print("=" * 70)

    summary_df = build_summary(models_dir, data_dir)

    if summary_df is not None:
        print(f"\n{summary_df.to_string(index=False)}")
        print_best_summary(summary_df)

    print("\n" + "=" * 70)
    print("ALL CONDITIONS COMPLETE")
    print("=" * 70)
