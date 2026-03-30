"""Mixed-effects linear model usingstatsmodels MixedLM with random intercepts and optional random slopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM


@dataclass
class ModelResult:
    """Container for model predictions and metadata."""

    predictions: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    coefficients: dict[str, float] | None = None


class MixedLMModel:
    """Linear mixed-effects model using statsmodels.

    Accounts for repeated measures within participants by including
    random intercepts (and optionally random slopes).
    """

    def __init__(
        self,
        random_effects: list[str] | None = None,
        reml: bool = True,
    ):
        self.random_effects = random_effects or []
        self.reml = reml
        self.is_fitted: bool = False
        self._model: Any = None
        self._result: Any = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> MixedLMModel:
        """Fit mixed-effects model.

        Args:
            X: Feature matrix (DataFrame preferred for column names)
            y: Target values
            groups: Participant IDs (required for random effects)
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = X
            self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

        data = pd.DataFrame(X_arr, columns=self._feature_names)
        data["y"] = y
        data["group"] = groups

        fixed_formula = " + ".join(self._feature_names)
        formula = f"y ~ {fixed_formula}"

        if self.random_effects:
            re_formula = " + ".join(["1"] + self.random_effects)
        else:
            re_formula = "1"

        self._model = MixedLM.from_formula(
            formula,
            groups=data["group"],
            re_formula=f"~{re_formula}",
            data=data,
        )
        self._result = self._model.fit(reml=self.reml)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        groups: np.ndarray | None = None,
    ) -> ModelResult:
        """Generate predictions with fixed + random effects for known groups."""
        if not self.is_fitted or self._result is None:
            raise RuntimeError("Model must be fit before predicting")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        data = pd.DataFrame(X_arr, columns=self._feature_names)
        fixed_pred = self._result.predict(data)
        if hasattr(fixed_pred, "values"):
            fixed_pred = fixed_pred.values

        predictions = fixed_pred.copy()

        if groups is not None:
            random_effects = self._result.random_effects
            for i, group_id in enumerate(groups):
                if group_id in random_effects:
                    re = random_effects[group_id]
                    predictions[i] += re.iloc[0]  # random intercept
                    for j, col_name in enumerate(self.random_effects):
                        if j + 1 < len(re) and col_name in self._feature_names:
                            col_idx = self._feature_names.index(col_name)
                            predictions[i] += re.iloc[j + 1] * X_arr[i, col_idx]

        coefficients = dict(self._result.fe_params)

        if groups is not None:
            n_known = sum(1 for g in groups if g in self._result.random_effects)
        else:
            n_known = 0
        n_total = len(X_arr)

        return ModelResult(
            predictions=predictions,
            coefficients=coefficients,
            metadata={
                "aic": self._result.aic,
                "bic": self._result.bic,
                "llf": self._result.llf,
                "converged": self._result.converged,
                "n_known_groups": n_known,
                "n_total": n_total,
                "pct_with_random_effects": n_known / n_total if n_total > 0 else 0,
            },
        )

    def summary(self) -> str:
        if self._result is None:
            return "Model not fitted"
        return str(self._result.summary())

    def get_random_effects(self) -> pd.DataFrame:
        """Extract per-group random effects as a DataFrame."""
        if self._result is None:
            raise RuntimeError("Model must be fit first")

        re = self._result.random_effects
        records = []
        for group_id, effects in re.items():
            record = {"group": group_id, "intercept": effects.iloc[0]}
            for i, col_name in enumerate(self.random_effects):
                if i + 1 < len(effects):
                    record[col_name] = effects.iloc[i + 1]
            records.append(record)
        return pd.DataFrame(records)

    def get_convergence_info(self) -> dict[str, Any]:
        """Get convergence diagnostics."""
        if self._result is None:
            raise RuntimeError("Model must be fit first")

        aic = self._result.aic
        bic = self._result.bic
        llf = self._result.llf

        # Compute AIC/BIC from llf if nan (common with REML)
        if np.isnan(aic) and not np.isnan(llf):
            cov_re = self._result.cov_re
            n_var = cov_re.size if hasattr(cov_re, "size") else len(cov_re.values.flatten())
            k = len(self._result.fe_params) + n_var
            aic = -2 * llf + 2 * k

        if np.isnan(bic) and not np.isnan(llf):
            cov_re = self._result.cov_re
            n_var = cov_re.size if hasattr(cov_re, "size") else len(cov_re.values.flatten())
            k = len(self._result.fe_params) + n_var
            n = self._result.nobs
            bic = -2 * llf + k * np.log(n)

        info = {
            "converged": self._result.converged,
            "aic": float(aic) if not np.isnan(aic) else None,
            "bic": float(bic) if not np.isnan(bic) else None,
            "llf": float(llf) if llf is not None else None,
            "nobs": self._result.nobs,
            "scale": float(self._result.scale),
            "random_effects_config": self.random_effects,
        }

        if hasattr(self._result, "cov_re"):
            cov_re = self._result.cov_re
            info["cov_random_effects"] = cov_re.tolist() if hasattr(cov_re, "tolist") else cov_re

        return info

    def fit_with_fallback(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        optimizers: list[str] | None = None,
    ) -> tuple[MixedLMModel, dict[str, Any]]:
        """Fit with fallback optimization strategies.

        Tries multiple optimizers. If random slopes fail, simplifies to
        random intercept only.
        """
        if optimizers is None:
            optimizers = ["lbfgs", "bfgs", "powell"]

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = X
            self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

        data = pd.DataFrame(X_arr, columns=self._feature_names)
        data["y"] = y
        data["group"] = groups

        fixed_formula = " + ".join(self._feature_names)
        formula = f"y ~ {fixed_formula}"

        convergence_info = {
            "attempts": [],
            "final_config": None,
            "converged": False,
            "simplified": False,
        }

        re_configs = [self.random_effects]
        if self.random_effects:
            re_configs.append([])  # fallback to intercept only

        for re_config in re_configs:
            re_formula = " + ".join(["1"] + re_config) if re_config else "1"

            model = MixedLM.from_formula(
                formula,
                groups=data["group"],
                re_formula=f"~{re_formula}",
                data=data,
            )

            for optimizer in optimizers:
                attempt = {
                    "optimizer": optimizer,
                    "random_effects": re_config,
                    "success": False,
                    "error": None,
                }
                try:
                    if optimizer == "lbfgs":
                        result = model.fit(reml=self.reml)
                    else:
                        result = model.fit(reml=self.reml, method=optimizer)

                    if result.converged:
                        self._model = model
                        self._result = result
                        self.is_fitted = True
                        self.random_effects = re_config

                        attempt["success"] = True
                        convergence_info["attempts"].append(attempt)
                        convergence_info["final_config"] = re_config
                        convergence_info["converged"] = True
                        convergence_info["simplified"] = re_config != self.random_effects
                        return self, convergence_info

                    attempt["error"] = "Did not converge"
                except Exception as e:
                    attempt["error"] = str(e)

                convergence_info["attempts"].append(attempt)

        if self._result is not None:
            convergence_info["final_config"] = re_configs[-1]
            return self, convergence_info

        raise RuntimeError(
            f"Model fitting failed after {len(convergence_info['attempts'])} attempts. "
            f"Last error: {convergence_info['attempts'][-1]['error']}"
        )
