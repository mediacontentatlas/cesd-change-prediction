"""Evaluation metrics, baselines, and comparison utilities for ElasticNet regression.

Provides:
- Aggregate metrics (MAE, RMSE, R², within-person R² median, between-person R²)
- Per-person metrics
- Five regression baselines (B0-B4)
- Baseline comparison reporting

Ported from regression/mixedlm/scripts/metrics.py for cross-model consistency.
Within-person R² uses the median of per-person R² values (not pooled), matching
the MixedLM implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Aggregate & per-person metrics
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    mae: float
    rmse: float
    r2: float
    within_person_r2_median: float
    between_person_r2: float


def compute_aggregate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    person_ids: np.ndarray,
) -> AggregateMetrics:
    """Compute MAE, RMSE, overall R², within-person R² (median), between-person R²."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Overall R²
    ss_res_all = np.sum((y_true - y_pred) ** 2)
    ss_tot_all = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res_all / ss_tot_all) if ss_tot_all > 0 else float("nan")

    # Within-person R²: use MEDIAN (not mean) because a few persons with
    # very few observations can produce extreme negative R² values that
    # blow up the mean. Median is robust to these outliers.
    within_r2s = []
    for pid in np.unique(person_ids):
        mask = person_ids == pid
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) < 2:
            continue
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        if ss_tot > 0:
            within_r2s.append(1 - ss_res / ss_tot)
    within_r2_median = float(np.median(within_r2s)) if within_r2s else float("nan")

    # Between-person R²: R² on person-level means
    person_mean_true = []
    person_mean_pred = []
    for pid in np.unique(person_ids):
        mask = person_ids == pid
        person_mean_true.append(np.mean(y_true[mask]))
        person_mean_pred.append(np.mean(y_pred[mask]))
    pmt = np.array(person_mean_true)
    pmp = np.array(person_mean_pred)
    ss_res_b = np.sum((pmt - pmp) ** 2)
    ss_tot_b = np.sum((pmt - np.mean(pmt)) ** 2)
    between_r2 = float(1 - ss_res_b / ss_tot_b) if ss_tot_b > 0 else float("nan")

    return AggregateMetrics(
        mae=mae, rmse=rmse, r2=r2,
        within_person_r2_median=within_r2_median,
        between_person_r2=between_r2,
    )


def compute_per_person_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    person_ids: np.ndarray,
) -> pd.DataFrame:
    """Per-person MAE, RMSE, and n_obs."""
    rows = []
    for pid in np.unique(person_ids):
        mask = person_ids == pid
        yt, yp = y_true[mask], y_pred[mask]
        rows.append({
            "person_id": pid,
            "n_obs": int(mask.sum()),
            "mae": float(np.mean(np.abs(yt - yp))),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        })
    return pd.DataFrame(rows).sort_values("person_id")


# ---------------------------------------------------------------------------
# Baselines (B0-B4)
# ---------------------------------------------------------------------------

@dataclass
class BaselinePredictions:
    b0_no_change: np.ndarray
    b1_population_mean: np.ndarray
    b2_last_value_carried_forward: np.ndarray
    b3_person_mean: np.ndarray
    b4_regression_to_mean: np.ndarray


def compute_baselines(
    y_train: np.ndarray,
    y_target: np.ndarray,
    pid_train: np.ndarray,
    pid_target: np.ndarray,
) -> BaselinePredictions:
    """Compute five baselines for a target split (val or test).

    Baselines are fit on training data and applied to the target split.
    """
    pop_mean = float(np.mean(y_train))

    # B0: predict 0 (no change)
    b0 = np.zeros(len(y_target), dtype=float)

    # B1: population mean from train
    b1 = np.full(len(y_target), pop_mean, dtype=float)

    # B2: last training value per person
    last_vals = {}
    for pid in np.unique(pid_train):
        mask = pid_train == pid
        last_vals[pid] = float(y_train[mask][-1])
    b2 = np.array([last_vals.get(p, pop_mean) for p in pid_target], dtype=float)

    # B3: person-specific mean from training
    person_means = {}
    for pid in np.unique(pid_train):
        mask = pid_train == pid
        person_means[pid] = float(np.mean(y_train[mask]))
    b3 = np.array([person_means.get(p, pop_mean) for p in pid_target], dtype=float)

    # B4: regression to mean (shrinkage = 0.5)
    shrinkage = 0.5
    b4 = np.array([
        pop_mean + (person_means.get(p, pop_mean) - pop_mean) * (1 - shrinkage)
        for p in pid_target
    ], dtype=float)

    return BaselinePredictions(
        b0_no_change=b0,
        b1_population_mean=b1,
        b2_last_value_carried_forward=b2,
        b3_person_mean=b3,
        b4_regression_to_mean=b4,
    )


def compute_train_baselines(
    y_train: np.ndarray,
    pid_train: np.ndarray,
) -> BaselinePredictions:
    """Compute baselines for the training set itself."""
    pop_mean = float(np.mean(y_train))

    b0 = np.zeros(len(y_train), dtype=float)
    b1 = np.full(len(y_train), pop_mean, dtype=float)

    # B2 within training: carry forward previous observation per person
    b2 = np.empty(len(y_train), dtype=float)
    for pid in np.unique(pid_train):
        inds = np.sort(np.where(pid_train == pid)[0])
        for pos, idx in enumerate(inds):
            b2[idx] = y_train[inds[pos - 1]] if pos > 0 else pop_mean

    person_means = {}
    for pid in np.unique(pid_train):
        mask = pid_train == pid
        person_means[pid] = float(np.mean(y_train[mask]))

    b3 = np.array([person_means.get(p, pop_mean) for p in pid_train], dtype=float)

    shrinkage = 0.5
    b4 = np.array([
        pop_mean + (person_means.get(p, pop_mean) - pop_mean) * (1 - shrinkage)
        for p in pid_train
    ], dtype=float)

    return BaselinePredictions(
        b0_no_change=b0,
        b1_population_mean=b1,
        b2_last_value_carried_forward=b2,
        b3_person_mean=b3,
        b4_regression_to_mean=b4,
    )


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def build_comparison_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    person_ids: np.ndarray,
    baselines: BaselinePredictions,
    model_name: str = "ElasticNet",
) -> pd.DataFrame:
    """Build aggregate comparison table: model vs all baselines."""
    entries = [
        (model_name, y_pred),
        ("B0: No Change", baselines.b0_no_change),
        ("B1: Population Mean", baselines.b1_population_mean),
        ("B2: Last Value Carried Forward", baselines.b2_last_value_carried_forward),
        ("B3: Person-Specific Mean", baselines.b3_person_mean),
        ("B4: Regression to Mean", baselines.b4_regression_to_mean),
    ]

    rows = []
    for name, pred in entries:
        m = compute_aggregate_metrics(y_true, pred, person_ids)
        pp = compute_per_person_metrics(y_true, pred, person_ids)
        rows.append({
            "Method": name,
            "MAE": m.mae,
            "RMSE": m.rmse,
            "R2": m.r2,
            "Within-R2-median": m.within_person_r2_median,
            "Between-R2": m.between_person_r2,
            "MedianPersonMAE": float(pp["mae"].median()),
            "Q1_MAE": float(pp["mae"].quantile(0.25)),
            "Q3_MAE": float(pp["mae"].quantile(0.75)),
        })
    return pd.DataFrame(rows)


def baselines_to_dict(bl: BaselinePredictions) -> dict[str, np.ndarray]:
    """Convert BaselinePredictions to a name->array dict."""
    return {
        "B0: No Change": bl.b0_no_change,
        "B1: Population Mean": bl.b1_population_mean,
        "B2: Last Value Carried Forward": bl.b2_last_value_carried_forward,
        "B3: Person-Specific Mean": bl.b3_person_mean,
        "B4: Regression to Mean": bl.b4_regression_to_mean,
    }
