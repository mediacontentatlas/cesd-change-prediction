"""Train ElasticNet regression with grid search hyperparameter tuning.

ML Pipeline
-----------
  1. Grid search: fit on Train, evaluate on Val -> select best (alpha, l1_ratio)
  2. Dev model:   fit on Train with best params -> predict Train + Val
  3. Final model:  refit on Train+Val with SAME params -> predict Test
     (Step 3 only runs when --run-test is passed.)

Hyperparameters are locked after step 1 and never re-tuned.
Train+Val combination in step 3 maximises training data for the final
held-out test evaluation. Val metrics from step 2 remain the valid
development-phase holdout numbers.

Feature Ablation Conditions
---------------------------
Required (to match classification):
  prior_cesd        -- prior_cesd only (1 feature)
  base              -- all 21 base features
  base_lag          -- base + lag-1 features + lag_cesd_delta
  base_lag_pmcesd   -- base_lag + person_mean_cesd

Additional (from feature engineered variables):
  dev               -- base + within-person deviation features
  pheno             -- base + phenotype features
  pid               -- base + PID one-hot encoding
  dev_pheno         -- base + dev + pheno
  dev_pid           -- base + dev + PID OHE
  pheno_pid         -- base + pheno + PID OHE
  dev_pheno_pid     -- base + dev + pheno + PID OHE

Usage:
    # Dev only (grid search + val evaluation):
    python scripts/train_elasticnet.py --condition base

    # Full pipeline including test evaluation:
    python scripts/train_elasticnet.py --condition base --run-test

    # With lag features + person_mean_cesd:
    python scripts/train_elasticnet.py --condition base_lag_pmcesd --run-test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_ELASTICNET_DIR = _SCRIPT_DIR.parent

ALL_CONDITIONS = [
    "prior_cesd", "base", "base_lag", "base_lag_pmcesd",
    "dev", "pheno", "pid",
    "dev_pheno", "dev_pid", "pheno_pid", "dev_pheno_pid",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_feature_names(features_txt: Path) -> list[str]:
    """Parse feature names from features.txt."""
    names = []
    with open(features_txt) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("- ") and ":" not in stripped:
                names.append(stripped[2:].strip())
    return names


def build_lag_features(
    X_base: np.ndarray,
    feature_names: list[str],
    data_dir: Path,
    split: str,
) -> tuple[np.ndarray, list[str]]:
    """Build lag-1 behavioral features from scaled CSVs for a single split.

    Lags the 17 time-varying behavioral features only. Static demographics
    (age, gender) and clinical lags (prior_cesd, cesd_delta) are excluded
    per ablation (DATA_README.md §8.7). Matches the classification pipeline's
    39-feature model (bootstrap_ci.py, deployment_scenarios.py, etc.).

    Returns (X_with_lag, updated_feature_names).
    """
    # Static/clinical lags excluded per ablation -- see DATA_README.md §8.7
    _DROP_LAGS = {
        "lag_age", "lag_gender_mode_1", "lag_gender_mode_2",
        "lag_prior_cesd", "lag_cesd_delta",
    }

    all_df = pd.concat([
        pd.read_csv(data_dir / "train_scaled.csv"),
        pd.read_csv(data_dir / "val_scaled.csv"),
        pd.read_csv(data_dir / "test_scaled.csv"),
    ]).sort_values(["pid", "period_number"]).reset_index(drop=True)

    lag_cols_all = [f"lag_{c}" for c in feature_names] + ["lag_cesd_delta"]

    for col in feature_names:
        if col in all_df.columns:
            all_df[f"lag_{col}"] = all_df.groupby("pid")[col].shift(1)
        else:
            all_df[f"lag_{col}"] = 0.0

    all_df["lag_cesd_delta"] = all_df.groupby("pid")["target_cesd_delta"].shift(1)
    all_df[lag_cols_all] = all_df[lag_cols_all].fillna(0)

    # Drop static + clinical lags
    lag_cols = [c for c in lag_cols_all if c not in _DROP_LAGS]

    df_split = all_df[all_df["split"] == split].copy()
    lag_arr = df_split[lag_cols].values

    X_out = np.hstack([X_base, lag_arr])
    return X_out, feature_names + lag_cols


def build_feature_matrix(
    condition: str,
    X_base: np.ndarray,
    pid: np.ndarray,
    feature_names: list[str],
    data_dir: Path,
    split: str,
    pid_encoder: OneHotEncoder | None = None,
    pmcesd_lookup: dict | None = None,
) -> tuple[np.ndarray, list[str], OneHotEncoder | None]:
    """Build the feature matrix for a given condition and split.

    Returns (X, feature_names, pid_encoder).
    """
    enc = pid_encoder

    if condition == "prior_cesd":
        return X_base[:, 0:1], ["prior_cesd"], enc

    if condition == "base":
        return X_base.copy(), list(feature_names), enc

    if condition in ("base_lag", "base_lag_pmcesd"):
        X_out, names_out = build_lag_features(X_base, feature_names, data_dir, split)
        if condition == "base_lag_pmcesd":
            pop_mean = pmcesd_lookup["pop_mean"]
            person_means = pmcesd_lookup["person_means"]
            pmcesd = np.array([person_means.get(p, pop_mean) for p in pid])
            X_out = np.hstack([X_out, pmcesd.reshape(-1, 1)])
            names_out = names_out + ["person_mean_cesd"]
        return X_out, names_out, enc

    # --- Source repo extras ---
    X_out = X_base.copy()
    names_out = list(feature_names)

    # Determine which extras to load
    use_dev = condition in ("dev", "dev_pheno", "dev_pid", "dev_pheno_pid")
    use_pheno = condition in ("pheno", "dev_pheno", "pheno_pid", "dev_pheno_pid")
    use_pid = condition in ("pid", "dev_pid", "pheno_pid", "dev_pheno_pid")

    if use_dev:
        X_dev = np.load(data_dir / f"X_dev_{split}.npy")
        dev_names = [f"dev_{i}" for i in range(X_dev.shape[1])]
        X_out = np.hstack([X_out, X_dev])
        names_out = names_out + dev_names

    if use_pheno:
        X_pheno = np.load(data_dir / f"X_all_phenotype_{split}.npy")
        pheno_names = [f"pheno_{i}" for i in range(X_pheno.shape[1])]
        X_out = np.hstack([X_out, X_pheno])
        names_out = names_out + pheno_names

    if use_pid:
        if enc is None:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            enc.fit(pid.reshape(-1, 1))
        pid_ohe = enc.transform(pid.reshape(-1, 1))
        pid_names = [f"pid_{int(c)}" for c in enc.categories_[0]]
        X_out = np.hstack([pid_ohe, X_out])
        names_out = pid_names + names_out

    return X_out, names_out, enc


def compute_within_person_r2(
    y_true: np.ndarray, y_pred: np.ndarray, pids: np.ndarray,
) -> float:
    """Variance explained WITHIN individuals.

    For each person: compute SS_res and SS_tot (centered on person mean).
    Pool across persons: 1 - sum(SS_res_i) / sum(SS_tot_i).
    """
    ss_res_total, ss_tot_total = 0.0, 0.0
    for pid in np.unique(pids):
        mask = pids == pid
        yt = y_true[mask]
        yp = y_pred[mask]
        person_mean = yt.mean()
        ss_res_total += np.sum((yt - yp) ** 2)
        ss_tot_total += np.sum((yt - person_mean) ** 2)
    if ss_tot_total == 0:
        return 0.0
    return 1.0 - ss_res_total / ss_tot_total


def compute_between_person_r2(
    y_true: np.ndarray, y_pred: np.ndarray, pids: np.ndarray,
) -> float:
    """Variance explained BETWEEN individuals (person-mean level)."""
    person_means_true = []
    person_means_pred = []
    for pid in np.unique(pids):
        mask = pids == pid
        person_means_true.append(y_true[mask].mean())
        person_means_pred.append(y_pred[mask].mean())
    pm_true = np.array(person_means_true)
    pm_pred = np.array(person_means_pred)
    grand_mean = pm_true.mean()
    ss_res = np.sum((pm_true - pm_pred) ** 2)
    ss_tot = np.sum((pm_true - grand_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def plot_validation_curves(
    results_df: pd.DataFrame,
    best_alpha: float,
    best_l1_ratio: float,
    best_val_mae: float,
    output_dir: Path,
) -> None:
    """Plot validation curves from grid search results.

    Generates three plots:
      - Heatmap of Val MAE across alpha x l1_ratio
      - Line plot: Val MAE vs alpha (one line per l1_ratio)
      - Line plot: Val MAE vs l1_ratio (subset of alphas)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = results_df

    # --- Plot 1: Heatmap ---
    pivot = df.pivot_table(values="val_mae", index="l1_ratio", columns="alpha")
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{a:.4g}" for a in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("Alpha (regularization strength)")
    ax.set_ylabel("L1 Ratio (0=Ridge, 1=Lasso)")
    ax.set_title("Validation MAE: Alpha x L1 Ratio Grid Search")
    plt.colorbar(im, ax=ax, label="Val MAE")

    # Mark best cell
    best_row_idx = list(pivot.index).index(best_l1_ratio)
    best_col_idx = list(pivot.columns).index(best_alpha)
    ax.add_patch(plt.Rectangle(
        (best_col_idx - 0.5, best_row_idx - 0.5), 1, 1,
        fill=False, edgecolor="blue", linewidth=3,
    ))
    plt.tight_layout()
    fig.savefig(output_dir / "validation_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Val MAE vs alpha ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for l1r in sorted(df["l1_ratio"].unique()):
        subset = df[df["l1_ratio"] == l1r].sort_values("alpha")
        ax.plot(subset["alpha"], subset["val_mae"], marker="o",
                label=f"l1_ratio={l1r:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (log scale)")
    ax.set_ylabel("Validation MAE")
    ax.set_title("Validation MAE vs Alpha")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.axhline(y=best_val_mae, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / "validation_curves_alpha.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Val MAE vs l1_ratio ---
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_alphas = sorted(df["alpha"].unique())
    if len(unique_alphas) > 8:
        step = max(1, len(unique_alphas) // 6)
        plot_alphas = unique_alphas[::step]
        if best_alpha not in plot_alphas:
            plot_alphas.append(best_alpha)
            plot_alphas.sort()
    else:
        plot_alphas = unique_alphas

    for alpha in plot_alphas:
        subset = df[df["alpha"] == alpha].sort_values("l1_ratio")
        lw = 2.5 if alpha == best_alpha else 1.0
        style = "-" if alpha == best_alpha else "--"
        ax.plot(subset["l1_ratio"], subset["val_mae"],
                marker="o", linestyle=style, linewidth=lw,
                label=f"a={alpha:.4g}")
    ax.set_xlabel("L1 Ratio")
    ax.set_ylabel("Validation MAE")
    ax.set_title("Validation MAE vs L1 Ratio")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(output_dir / "validation_curves_l1ratio.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ElasticNet regression with grid search",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Processed data directory (default: ../../data/processed relative to elasticnet/)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Model config YAML (default: ../configs/elasticnet.yaml relative to scripts/)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: ../models/{condition})",
    )
    parser.add_argument(
        "--condition", default="base", choices=ALL_CONDITIONS,
        help="Feature ablation condition (default: base)",
    )
    parser.add_argument(
        "--run-test", action="store_true",
        help="Run final test evaluation: refit on Train+Val with locked params, predict Test",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip validation curve plots",
    )
    args = parser.parse_args()

    # Resolve defaults relative to the elasticnet directory
    data_dir = Path(args.data_dir) if args.data_dir else _ELASTICNET_DIR.parent.parent / "data" / "processed"
    config_path = Path(args.config) if args.config else _ELASTICNET_DIR / "configs" / "elasticnet.yaml"
    output_dir = Path(args.output_dir) if args.output_dir else _ELASTICNET_DIR / "models" / args.condition

    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    params = config["params"]
    alphas = params["alphas"]
    l1_ratios = params["l1_ratio"]
    max_iter = params.get("max_iter", 10000)
    condition = args.condition

    print("=" * 70)
    print("ELASTICNET REGRESSION  -- GRID SEARCH TRAINING")
    print("=" * 70)
    print(f"Config:     {config_path}")
    print(f"Condition:  {condition}")
    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Grid:       {len(alphas)} alphas x {len(l1_ratios)} l1_ratios "
          f"= {len(alphas) * len(l1_ratios)} combos")

    # ======================================================================
    # Step 1: Load data
    # ======================================================================
    print("\n[Step 1] Loading data...")

    X_train_base = np.load(data_dir / "X_train.npy")
    X_val_base = np.load(data_dir / "X_val.npy")
    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    pid_train = np.load(data_dir / "pid_train.npy")
    pid_val = np.load(data_dir / "pid_val.npy")

    base_feature_names = parse_feature_names(data_dir / "features.txt")

    X_test_base = y_test = pid_test = None
    if args.run_test:
        X_test_base = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test.npy")
        pid_test = np.load(data_dir / "pid_test.npy")

    print(f"  Train: X={X_train_base.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val_base.shape}, y={y_val.shape}")
    if X_test_base is not None:
        print(f"  Test:  X={X_test_base.shape}, y={y_test.shape}")
    print(f"  Test evaluation: {'YES (--run-test)' if args.run_test else 'NO'}")

    # ======================================================================
    # Step 2: Build feature matrix per condition
    # ======================================================================
    print(f"\n[Step 2] Building features for condition: {condition}...")

    # Pre-compute person_mean_cesd if needed
    pmcesd_lookup = None
    if condition == "base_lag_pmcesd":
        prior_train = X_train_base[:, 0]
        pop_mean = float(prior_train.mean())
        person_means: dict = {}
        for pid in np.unique(pid_train):
            person_means[pid] = float(prior_train[pid_train == pid].mean())
        pmcesd_lookup = {"pop_mean": pop_mean, "person_means": person_means}
        print(f"  person_mean_cesd: pop_mean={pop_mean:.1f}, "
              f"person range [{min(person_means.values()):.1f}, "
              f"{max(person_means.values()):.1f}]")

    # Build for train
    X_tr, feature_names, pid_enc = build_feature_matrix(
        condition, X_train_base, pid_train, base_feature_names,
        data_dir, "train", pid_encoder=None, pmcesd_lookup=pmcesd_lookup,
    )

    # Build for val (reuse pid_encoder if applicable)
    X_va, _, pid_enc = build_feature_matrix(
        condition, X_val_base, pid_val, base_feature_names,
        data_dir, "val", pid_encoder=pid_enc, pmcesd_lookup=pmcesd_lookup,
    )

    # Build for test if needed
    X_te = None
    if args.run_test:
        X_te, _, pid_enc = build_feature_matrix(
            condition, X_test_base, pid_test, base_feature_names,
            data_dir, "test", pid_encoder=pid_enc, pmcesd_lookup=pmcesd_lookup,
        )

    if len(feature_names) != X_tr.shape[1]:
        print(f"  NOTE: feature_names has {len(feature_names)} entries but X has "
              f"{X_tr.shape[1]} columns  -- using generic names")
        feature_names = [f"feature_{i}" for i in range(X_tr.shape[1])]

    print(f"  Final feature matrix: {X_tr.shape[1]} features")
    print(f"  Train: X={X_tr.shape}")
    print(f"  Val:   X={X_va.shape}")
    if X_te is not None:
        print(f"  Test:  X={X_te.shape}")

    # ======================================================================
    # Step 3: Distribution check
    # ======================================================================
    print("\n[Step 3] Checking distributions...")
    print(f"\n{'':2s}Target (cesd_delta):")
    print(f"  {'':15s}  {'Train':>10s}  {'Val':>10s}  {'Diff':>10s}")
    for label, fn in [("Mean", np.mean), ("Std", np.std), ("Min", np.min),
                       ("Max", np.max), ("Median", np.median)]:
        tr_val = fn(y_train)
        vl_val = fn(y_val)
        print(f"  {label:15s}  {tr_val:10.3f}  {vl_val:10.3f}  {vl_val - tr_val:10.3f}")

    print(f"\n  Features (mean comparison):")
    print(f"  {'Feature':35s}  {'Train':>10s}  {'Val':>10s}  {'Diff':>8s}  {'Flag':>5s}")
    print(f"  {'-' * 35}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 5}")

    flagged = 0
    for i, name in enumerate(feature_names):
        if i >= X_tr.shape[1]:
            break
        tr_mean = np.mean(X_tr[:, i])
        vl_mean = np.mean(X_va[:, i])
        tr_std = np.std(X_tr[:, i])
        diff = abs(vl_mean - tr_mean)
        flag = " !" if (tr_std > 0 and diff / tr_std > 1.0) else ""
        if flag:
            flagged += 1
        print(f"  {name:35s}  {tr_mean:10.3f}  {vl_mean:10.3f}  {diff:8.3f}  {flag:>5s}")

    pids_tr_set = set(pid_train)
    pids_vl_set = set(pid_val)
    print(f"\n  Participants: Train={len(pids_tr_set)}, Val={len(pids_vl_set)}, "
          f"Overlap={len(pids_tr_set & pids_vl_set)}")
    if flagged:
        print(f"  NOTE: {flagged} feature(s) flagged with >1 std difference.")
    else:
        print("  OK: No features flagged for large distribution shift.")

    # ======================================================================
    # Step 4: Grid search
    # ======================================================================
    print(f"\n[Step 4] Running grid search ({len(alphas) * len(l1_ratios)} combos)...")

    results = []
    total = len(alphas) * len(l1_ratios)

    for i, alpha in enumerate(alphas):
        for j, l1_ratio in enumerate(l1_ratios):
            combo_num = i * len(l1_ratios) + j + 1

            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
            model.fit(X_tr, y_train)

            yp_tr = model.predict(X_tr)
            yp_va = model.predict(X_va)

            train_mae = float(mean_absolute_error(y_train, yp_tr))
            train_rmse = float(np.sqrt(mean_squared_error(y_train, yp_tr)))
            val_mae = float(mean_absolute_error(y_val, yp_va))
            val_rmse = float(np.sqrt(mean_squared_error(y_val, yp_va)))
            n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-10))

            results.append({
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "n_nonzero_coefs": n_nonzero,
                "n_iter": model.n_iter_,
                "intercept": float(model.intercept_),
            })

            if combo_num % 10 == 0 or combo_num == total:
                print(f"  Grid search: {combo_num}/{total} combos evaluated", end="\r")

    print()  # newline after progress

    results_df = pd.DataFrame(results)
    best_idx = results_df["val_mae"].idxmin()
    best_row = results_df.iloc[best_idx]
    best_alpha = float(best_row["alpha"])
    best_l1_ratio = float(best_row["l1_ratio"])
    best_val_mae = float(best_row["val_mae"])

    print(f"  Best alpha:    {best_alpha:.6g}")
    print(f"  Best l1_ratio: {best_l1_ratio:.4f}")
    print(f"  Best Val MAE:  {best_val_mae:.4f}")

    results_df.to_csv(output_dir / "grid_search_results.csv", index=False)
    print(f"  Saved grid results to {output_dir / 'grid_search_results.csv'}")

    # ======================================================================
    # Step 5: Validation curves
    # ======================================================================
    if not args.skip_plots:
        print("\n[Step 5] Generating validation curves...")
        plot_validation_curves(results_df, best_alpha, best_l1_ratio, best_val_mae, plots_dir)
        print(f"  Saved plots to {plots_dir}")
    else:
        print("\n[Step 5] Skipping validation curves (--skip-plots)")

    gap = best_row["val_mae"] - best_row["train_mae"]
    gap_pct = (100 * gap / best_row["train_mae"]) if best_row["train_mae"] > 0 else 0
    print(f"  Train MAE: {best_row['train_mae']:.4f}  (in-sample, diagnostic)")
    print(f"  Val MAE:   {best_row['val_mae']:.4f}  (out-of-sample)")
    print(f"  Overfitting gap (Train->Val): {gap:.4f} ({gap_pct:.1f}%)")
    if gap_pct > 20:
        print("  WARNING: Large overfitting gap  -- model may be memorizing noise.")
    else:
        print("  OK: Overfitting gap is small.")

    # ======================================================================
    # Step 6: Train dev model with best hyperparameters
    # ======================================================================
    print("\n[Step 6] Training dev model with best hyperparameters...")
    final_model = ElasticNet(
        alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=max_iter,
    )
    final_model.fit(X_tr, y_train)
    print(f"  Fitted on {X_tr.shape[0]} samples ({X_tr.shape[1]} features)")

    # ======================================================================
    # Step 7: Generate predictions (Train + Val)
    # ======================================================================
    print("\n[Step 7] Generating predictions...")
    y_pred_train = final_model.predict(X_tr)
    y_pred_val = final_model.predict(X_va)

    np.save(output_dir / "y_pred_train.npy", y_pred_train)
    np.save(output_dir / "y_pred_val.npy", y_pred_val)
    print(f"  Saved predictions to {output_dir}")

    # ======================================================================
    # Step 8: Compute metrics (MAE, RMSE, within-person R², between-person R²)
    # ======================================================================
    print("\n[Step 8] Computing metrics...")

    train_mae = float(mean_absolute_error(y_train, y_pred_train))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    train_wp_r2 = float(compute_within_person_r2(y_train, y_pred_train, pid_train))
    train_bp_r2 = float(compute_between_person_r2(y_train, y_pred_train, pid_train))

    val_mae = float(mean_absolute_error(y_val, y_pred_val))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
    val_wp_r2 = float(compute_within_person_r2(y_val, y_pred_val, pid_val))
    val_bp_r2 = float(compute_between_person_r2(y_val, y_pred_val, pid_val))

    header = f"  {'Split':10s}  {'MAE':>8s}  {'RMSE':>8s}  {'W-R²':>8s}  {'B-R²':>8s}  {'Note':>15s}"
    sep = f"  {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 15}"
    print(header)
    print(sep)
    print(f"  {'Train':10s}  {train_mae:8.4f}  {train_rmse:8.4f}  {train_wp_r2:8.4f}  "
          f"{train_bp_r2:8.4f}  {'in-sample':>15s}")
    print(f"  {'Val':10s}  {val_mae:8.4f}  {val_rmse:8.4f}  {val_wp_r2:8.4f}  "
          f"{val_bp_r2:8.4f}  {'out-of-sample':>15s}")

    gap = val_mae - train_mae
    print(f"\n  Overfitting gap (Train->Val): {gap:.4f} MAE ({100 * gap / train_mae:.1f}%)")

    # ======================================================================
    # Step 9: Save model artifact + feature coefficients
    # ======================================================================
    print("\n[Step 9] Saving model artifact and feature coefficients...")
    joblib.dump(final_model, output_dir / "model.joblib")

    best_params = {
        "alpha": best_alpha,
        "l1_ratio": best_l1_ratio,
        "max_iter": max_iter,
        "best_val_mae": best_val_mae,
        "condition": condition,
        "n_features": int(X_tr.shape[1]),
        "trained_on": "train",
        "n_train_samples": int(X_tr.shape[0]),
        "metrics": {
            "train": {"mae": train_mae, "rmse": train_rmse,
                      "within_r2": train_wp_r2, "between_r2": train_bp_r2},
            "val": {"mae": val_mae, "rmse": val_rmse,
                    "within_r2": val_wp_r2, "between_r2": val_bp_r2},
        },
    }
    with open(output_dir / "best_params.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    coefs = final_model.coef_
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    })
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(output_dir / "feature_coefficients.csv", index=False)

    print(f"  Saved model to {output_dir / 'model.joblib'}")
    print(f"  Saved best params to {output_dir / 'best_params.yaml'}")
    print(f"  Saved coefficients to {output_dir / 'feature_coefficients.csv'}")

    # Save PID encoder if used
    if pid_enc is not None:
        joblib.dump(pid_enc, output_dir / "pid_encoder.joblib")
        print(f"  Saved PID encoder to {output_dir / 'pid_encoder.joblib'}")

    # ======================================================================
    # Step 10: Final test evaluation (Train+Val refit)  -- only with --run-test
    # ======================================================================
    if args.run_test:
        print("\n[Step 10] Final test evaluation (Train+Val refit)...")
        print("  Hyperparameters LOCKED from grid search (no re-tuning):")
        print(f"    alpha={best_alpha:.6g}, l1_ratio={best_l1_ratio:.4f}")

        # 10a. Concatenate Train + Val
        X_trainval = np.vstack([X_tr, X_va])
        y_trainval = np.concatenate([y_train, y_val])
        pid_trainval = np.concatenate([pid_train, pid_val])
        print(f"  Train+Val: X={X_trainval.shape}, y={y_trainval.shape}")

        # 10b. Refit with same hyperparameters
        final_test_model = ElasticNet(
            alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=max_iter,
        )
        final_test_model.fit(X_trainval, y_trainval)

        # 10c. Predict Test
        y_pred_test = final_test_model.predict(X_te)
        np.save(output_dir / "y_pred_test.npy", y_pred_test)

        # 10d. Save final model artifact
        joblib.dump(final_test_model, output_dir / "final_model.joblib")

        # 10e. Test metrics
        test_mae = float(mean_absolute_error(y_test, y_pred_test))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        test_wp_r2 = float(compute_within_person_r2(y_test, y_pred_test, pid_test))
        test_bp_r2 = float(compute_between_person_r2(y_test, y_pred_test, pid_test))

        final_params = {
            "alpha": best_alpha,
            "l1_ratio": best_l1_ratio,
            "max_iter": max_iter,
            "best_val_mae": best_val_mae,
            "condition": condition,
            "n_features": int(X_trainval.shape[1]),
            "trained_on": "train+val",
            "n_trainval_samples": int(X_trainval.shape[0]),
            "n_test_samples": int(X_te.shape[0]),
            "metrics": {
                "train": {"mae": train_mae, "rmse": train_rmse,
                          "within_r2": train_wp_r2, "between_r2": train_bp_r2},
                "val": {"mae": val_mae, "rmse": val_rmse,
                        "within_r2": val_wp_r2, "between_r2": val_bp_r2},
                "test": {"mae": test_mae, "rmse": test_rmse,
                         "within_r2": test_wp_r2, "between_r2": test_bp_r2},
            },
        }
        with open(output_dir / "final_params.yaml", "w") as f:
            yaml.dump(final_params, f, default_flow_style=False)

        # 10f. Final test metrics summary
        print(f"\n  {'Split':10s}  {'MAE':>8s}  {'RMSE':>8s}  {'W-R²':>8s}  "
              f"{'B-R²':>8s}  {'Model':>15s}  {'Note':>15s}")
        print(f"  {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 8}  "
              f"{'-' * 8}  {'-' * 15}  {'-' * 15}")
        print(f"  {'Train':10s}  {train_mae:8.4f}  {train_rmse:8.4f}  {train_wp_r2:8.4f}  "
              f"{train_bp_r2:8.4f}  {'dev (Train)':>15s}  {'diagnostic':>15s}")
        print(f"  {'Val':10s}  {val_mae:8.4f}  {val_rmse:8.4f}  {val_wp_r2:8.4f}  "
              f"{val_bp_r2:8.4f}  {'dev (Train)':>15s}  {'out-of-sample':>15s}")
        print(f"  {'Test':10s}  {test_mae:8.4f}  {test_rmse:8.4f}  {test_wp_r2:8.4f}  "
              f"{test_bp_r2:8.4f}  {'final (Tr+Val)':>15s}  {'out-of-sample':>15s}")

        gap_tv = val_mae - train_mae
        gap_vt = test_mae - val_mae
        print(f"\n  Overfitting gap (Train->Val): {gap_tv:.4f} MAE")
        print(f"  Generalization gap (Val->Test): {gap_vt:.4f} MAE")

        # 10g. Save final test feature coefficients
        coefs_final = final_test_model.coef_
        coef_final_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs_final,
            "abs_coefficient": np.abs(coefs_final),
        })
        coef_final_df = coef_final_df.sort_values("abs_coefficient", ascending=False)
        coef_final_df.to_csv(output_dir / "final_feature_coefficients.csv", index=False)
        print(f"  Saved final model to {output_dir / 'final_model.joblib'}")
        print(f"  Saved test predictions to {output_dir / 'y_pred_test.npy'}")
        print(f"  Saved final coefficients to {output_dir / 'final_feature_coefficients.csv'}")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Condition: {condition}")
    print(f"Best hyperparameters: alpha={best_alpha:.6g}, l1_ratio={best_l1_ratio:.2f}")

    n_selected = int(np.sum(np.abs(final_model.coef_) > 1e-5))
    print(f"Selected features ({n_selected}/{len(feature_names)}):")
    for fname, coef in sorted(
        zip(feature_names, final_model.coef_),
        key=lambda x: abs(x[1]), reverse=True,
    ):
        if abs(coef) > 1e-5:
            print(f"  {fname:35s}  {coef:+.4f}")

    if args.run_test:
        print(f"\nTest MAE: {test_mae:.4f} (final model trained on Train+Val)")
    print(f"\nAll outputs saved to: {output_dir}")
    print("=" * 70)
