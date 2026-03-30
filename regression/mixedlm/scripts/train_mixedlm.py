#!/usr/bin/env python3
"""Train Mixed-Effects LMM with feature ablation conditions.

Trains MixedLMModel across four required feature ablation conditions
(matching the classification task) plus optional random slope variants.

Feature ablation conditions:
    1. prior_cesd only                      (1 feature)
    2. base                                 (21 features)
    3. base + within-person deviation (dev)  (29 features)
    4. base + dev + person_mean_cesd        (30 features)

Usage:
    # Run all four ablation conditions (random intercept only):
    python regression/mixedlm/scripts/train_mixedlm.py

    # Also train random slope variants:
    python regression/mixedlm/scripts/train_mixedlm.py --with-slopes

    # Train a single custom condition:
    python regression/mixedlm/scripts/train_mixedlm.py --condition base

Output (per condition):
    regression/mixedlm/models/<condition>/
        model.pkl
        y_pred_train.npy, y_pred_val.npy, y_pred_test.npy
        random_effects.csv
        convergence_info.json
        aggregate_comparison.csv       (per split)
        direction_classification.csv   (per split)
        training_results.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from this scripts directory
sys.path.insert(0, str(Path(__file__).parent))

from model import MixedLMModel
from metrics import (
    build_comparison_table,
    compute_baselines,
    compute_direction_classification,
    compute_train_baselines,
    baselines_to_dict,
)

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"


# ---------------------------------------------------------------------------
# Feature ablation conditions
# ---------------------------------------------------------------------------

def load_feature_names() -> list[str]:
    """Load the 21 base feature names from features.txt."""
    txt = (DATA_DIR / "features.txt").read_text()
    names = []
    for line in txt.splitlines():
        line = line.strip().lstrip("- ")
        if line and not line.startswith("#") and not line.startswith("Total") and ":" not in line:
            names.append(line)
    return names


def _get_condition_spec(condition: str, base_features: list[str]) -> dict:
    """Return feature specification for an ablation condition.

    Conditions match classification ablation for comparability:
        prior_cesd      — 1 feature
        base            — 21 base features
        base_dev        — 21 base + 8 within-person deviation features (X_dev)
        base_dev_pmcesd — 21 base + 8 dev + person_mean_cesd

    Returns dict with:
        features: list of feature column names to use (for subsetting)
        use_dev: whether to append X_dev features
        add_pmcesd: whether to add person_mean_cesd column
    """
    if condition == "prior_cesd":
        return {"features": ["prior_cesd"], "use_dev": False, "add_pmcesd": False}
    elif condition == "base":
        return {"features": base_features, "use_dev": False, "add_pmcesd": False}
    elif condition == "base_dev":
        return {"features": base_features, "use_dev": True, "add_pmcesd": False}
    elif condition == "base_dev_pmcesd":
        return {"features": base_features, "use_dev": True, "add_pmcesd": True}
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split_data(
    split: str,
    spec: dict,
    base_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load X, y, pid for a single split according to feature spec.

    Returns (X_df, y, pid).
    """
    X = np.load(DATA_DIR / f"X_{split}.npy")
    y = np.load(DATA_DIR / f"y_{split}.npy")
    pid = np.load(DATA_DIR / f"pid_{split}.npy")

    # Build DataFrame with feature names
    X_df = pd.DataFrame(X, columns=base_features)

    if spec["use_dev"]:
        dev_path = DATA_DIR / f"X_dev_{split}.npy"
        if dev_path.exists():
            X_dev = np.load(dev_path)
            dev_names = [f"dev_{i}" for i in range(X_dev.shape[1])]
            X_dev_df = pd.DataFrame(X_dev, columns=dev_names)
            X_df = pd.concat([X_df, X_dev_df], axis=1)
        else:
            print(f"  Warning: {dev_path} not found, skipping dev features for {split}")

    if spec["add_pmcesd"]:
        csv_path = DATA_DIR / f"{split}_scaled.csv"
        if csv_path.exists():
            df_csv = pd.read_csv(csv_path)
            if "person_mean_cesd" in df_csv.columns:
                X_df["person_mean_cesd"] = df_csv["person_mean_cesd"].values[:len(X_df)]
            else:
                # Compute from training pid means of prior_cesd
                print(f"  Warning: person_mean_cesd not in CSV, computing from prior_cesd")
                pid_means = X_df.groupby(pid)["prior_cesd"].transform("mean")
                X_df["person_mean_cesd"] = pid_means.values
        else:
            print(f"  Warning: {csv_path} not found, computing person_mean_cesd from prior_cesd")
            tmp_df = pd.DataFrame({"pid": pid, "prior_cesd": X_df["prior_cesd"].values})
            X_df["person_mean_cesd"] = tmp_df.groupby("pid")["prior_cesd"].transform("mean").values

    # Select only the features specified in the condition
    if spec["features"] != base_features:
        # For prior_cesd-only condition: select just that column
        cols_to_use = [c for c in spec["features"] if c in X_df.columns]
        X_df = X_df[cols_to_use]
    # For base + dev + pmcesd, we already have all columns

    return X_df, y, pid


# ---------------------------------------------------------------------------
# Training a single condition
# ---------------------------------------------------------------------------

def train_condition(
    condition: str,
    random_effects: list[str],
    base_features: list[str],
    output_dir: Path,
    label: str | None = None,
    pooled: bool = False,
) -> dict:
    """Train one ablation condition and save all outputs.

    Args:
        pooled: If True, replace person IDs with unique row indices so each
                observation is its own group (no random effects structure).

    Returns convergence info dict.
    """
    label = label or condition
    spec = _get_condition_spec(condition, base_features)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print(f"  Random effects: {random_effects or ['intercept only']}")
    if pooled:
        print(f"  Mode: POOLED (no person grouping)")
    print(f"{'='*60}")

    # Load data
    splits = {}
    idx = 0
    for s in ["train", "val", "test"]:
        X_df, y, pid = load_split_data(s, spec, base_features)
        if pooled:
            pid = np.arange(idx, idx + len(y))
            idx += len(y)
        splits[s] = {"X": X_df, "y": y, "pid": pid}

    feature_names = list(splits["train"]["X"].columns)
    print(f"  Features ({len(feature_names)}): {feature_names[:5]}{'...' if len(feature_names) > 5 else ''}")
    print(f"  Train: {len(splits['train']['y'])} | Val: {len(splits['val']['y'])} | Test: {len(splits['test']['y'])}")

    # Filter random effects to only include features present in this condition
    valid_re = [r for r in random_effects if r in feature_names]
    if valid_re != random_effects and random_effects:
        print(f"  Note: random effects filtered to {valid_re} (available features)")

    # Fit model
    model = MixedLMModel(random_effects=valid_re, reml=True)
    fitted_model, convergence_info = model.fit_with_fallback(
        X=splits["train"]["X"],
        y=splits["train"]["y"],
        groups=splits["train"]["pid"],
    )
    print(f"  Converged: {convergence_info['converged']}")
    print(f"  Attempts: {len(convergence_info['attempts'])}")

    if convergence_info.get("simplified"):
        print(f"  Note: simplified to random intercept only")

    # Save convergence info
    with open(output_dir / "convergence_info.json", "w") as f:
        json.dump(convergence_info, f, indent=2, default=str)

    # Generate predictions for all splits
    predictions = {}
    for s in ["train", "val", "test"]:
        result = fitted_model.predict(splits[s]["X"], groups=splits[s]["pid"])
        predictions[s] = result.predictions
        np.save(output_dir / f"y_pred_{s}.npy", result.predictions)

    # Save model
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(fitted_model, f)

    # Save random effects
    re_df = fitted_model.get_random_effects()
    re_df.to_csv(output_dir / "random_effects.csv", index=False)

    # Save model summary
    summary_path = output_dir / "model_summary.txt"
    summary_path.write_text(fitted_model.summary())

    # Compute baselines and evaluation for each split
    train_bl = compute_train_baselines(splits["train"]["y"], splits["train"]["pid"])
    val_bl = compute_baselines(
        splits["train"]["y"], splits["val"]["y"],
        splits["train"]["pid"], splits["val"]["pid"],
    )
    test_bl = compute_baselines(
        splits["train"]["y"], splits["test"]["y"],
        splits["train"]["pid"], splits["test"]["pid"],
    )

    bl_map = {"train": train_bl, "val": val_bl, "test": test_bl}

    for s in ["train", "val", "test"]:
        # Aggregate comparison table
        comp_df = build_comparison_table(
            splits[s]["y"], predictions[s], splits[s]["pid"],
            bl_map[s], model_name=f"MixedLM ({label})",
        )
        comp_df.to_csv(output_dir / f"{s}_aggregate_comparison.csv", index=False)

        # Direction classification
        cls_df = compute_direction_classification(
            splits[s]["y"], predictions[s], baselines_to_dict(bl_map[s]),
        )
        cls_df.to_csv(output_dir / f"{s}_direction_classification.csv", index=False)

        # Print summary
        m = comp_df.iloc[0]
        print(f"  {s.upper()}: MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"R2={m['R2']:.4f}  W-R2-med={m['Within-R2-median']:.4f}")

    # Save training results JSON
    model_info = fitted_model.get_convergence_info()
    model_info["condition"] = condition
    model_info["label"] = label
    model_info["n_features"] = len(feature_names)
    model_info["feature_names"] = feature_names
    model_info["random_effects"] = list(fitted_model.random_effects)

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(model_info, f, indent=2, default=str)

    return model_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Mixed-Effects LMM with feature ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        choices=["prior_cesd", "base", "base_dev", "base_dev_pmcesd"],
        help="Run a single ablation condition (default: run all four)",
    )
    parser.add_argument(
        "--with-slopes", action="store_true",
        help="Also train random slope for prior_cesd (in addition to intercept-only)",
    )
    parser.add_argument(
        "--random-slopes", nargs="+", default=[],
        help="Custom random slope columns (e.g., --random-slopes prior_cesd)",
    )
    parser.add_argument(
        "--no-pid", action="store_true",
        help="Pooled model: replace person IDs with unique row indices (no random effects structure)",
    )
    parser.add_argument(
        "--full-sweep", action="store_true",
        help="Run all 9 model variants from the screenome_mh_pred repo (on base features)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: regression/mixedlm/models/)",
    )

    args = parser.parse_args()

    output_base = Path(args.output_dir) if args.output_dir else MODELS_DIR
    base_features = load_feature_names()
    print(f"Loaded {len(base_features)} base features")

    all_results = {}

    if args.full_sweep:
        # Run all 9 model variants matching the screenome_mh_pred repo.
        # All use the "base" (21-feature) condition unless noted.
        sweep_configs = [
            # (label,            condition,  random_effects,                                                    pooled)
            ("1_pooled",         "base",     [],                                                                True),
            ("2_intercept",      "base",     [],                                                                False),
            ("3_prior_slope",    "base",     ["prior_cesd"],                                                    False),
            ("4_prior_switches", "base",     ["prior_cesd", "mean_daily_switches"],                             False),
            ("5_prior_social",   "base",     ["prior_cesd", "mean_daily_social_ratio"],                         False),
            ("6_prior_soc_ext",  "base",     ["prior_cesd", "mean_daily_social_screens", "mean_daily_social_ratio"], False),
            ("7_sw_screens",     "base",     ["mean_daily_switches", "mean_daily_screens"],                     False),
            ("8_prior_sw_scr",   "base",     ["prior_cesd", "mean_daily_switches", "mean_daily_screens"],       False),
            ("9_dev_features",   "base_dev", ["prior_cesd"],                                                    False),
        ]
        for label, condition, re_list, pooled in sweep_configs:
            out_dir = output_base / label
            info = train_condition(
                condition=condition,
                random_effects=re_list,
                base_features=base_features,
                output_dir=out_dir,
                label=label,
                pooled=pooled,
            )
            all_results[label] = info
    else:
        # Standard mode: feature ablation conditions x random effect configs
        if args.condition:
            conditions = [args.condition]
        else:
            conditions = ["prior_cesd", "base", "base_dev", "base_dev_pmcesd"]

        re_configs = [("intercept", [])]
        if args.with_slopes:
            re_configs.append(("slope_prior_cesd", ["prior_cesd"]))
        if args.random_slopes:
            slope_label = "slope_" + "_".join(args.random_slopes)
            re_configs.append((slope_label, args.random_slopes))

        for condition in conditions:
            for re_label, re_list in re_configs:
                if re_label == "intercept":
                    full_label = condition
                    out_dir = output_base / condition
                else:
                    full_label = f"{condition}_{re_label}"
                    out_dir = output_base / full_label

                info = train_condition(
                    condition=condition,
                    random_effects=re_list,
                    base_features=base_features,
                    output_dir=out_dir,
                    label=full_label,
                    pooled=args.no_pid,
                )
                all_results[full_label] = info

    # Save combined results summary
    with open(output_base / "all_training_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print final summary table
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY (Test Set)")
    print(f"{'='*80}")
    print(f"{'Condition':<30} {'MAE':<10} {'RMSE':<10} {'R2':<10} {'W-R2-med':<10} {'Conv'}")
    print("-" * 80)

    for label, info in all_results.items():
        test_csv = output_base / label / "test_aggregate_comparison.csv"
        if test_csv.exists():
            df = pd.read_csv(test_csv)
            row = df.iloc[0]
            print(f"{label:<30} {row['MAE']:<10.4f} {row['RMSE']:<10.4f} "
                  f"{row['R2']:<10.4f} {row['Within-R2-median']:<10.4f} "
                  f"{info.get('converged', '?')}")

    print(f"\nResults saved to: {output_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
