"""
factorlib.models
----------------
CAPM, Fama-French 3-Factor, and Fama-French 5-Factor regression models.
Each model follows the same interface: fit() → summary() / plot_*() / compare().
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import fetch_prices, fetch_ff_factors
from .metrics import compute_metrics
from .plot import (
    plot_loadings,
    plot_residuals,
    plot_rolling_alpha,
    plot_model_comparison,
)

ModelType = Literal["CAPM", "FF3", "FF5"]

FACTOR_COLS: dict[ModelType, list[str]] = {
    "CAPM": ["Mkt-RF"],
    "FF3":  ["Mkt-RF", "SMB", "HML"],
    "FF5":  ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}


@dataclass
class RegressionResult:
    """Holds output from a single factor model regression."""
    model_type: ModelType
    ticker: str
    alpha: float
    betas: dict[str, float]
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    residuals: pd.Series
    fitted_values: pd.Series
    params: pd.Series
    pvalues: pd.Series
    tvalues: pd.Series
    conf_int: pd.DataFrame
    nobs: int
    _sm_result: object = field(repr=False, default=None)

    def __str__(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  {self.model_type} — {self.ticker}",
            f"{'='*55}",
            f"  Alpha (annualised):  {self.alpha * 252:.4f}",
            f"  R²:                  {self.r_squared:.4f}",
            f"  Adj. R²:             {self.adj_r_squared:.4f}",
            f"  F-statistic:         {self.f_statistic:.2f}  (p={self.f_pvalue:.4f})",
            f"  Observations:        {self.nobs}",
            f"\n  Factor loadings:",
        ]
        for factor, beta in self.betas.items():
            pval = self.pvalues.get(factor, float("nan"))
            sig  = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            lines.append(f"    {factor:<10} β = {beta:+.4f}  (p={pval:.4f}) {sig}")
        lines.append(f"{'='*55}\n")
        return "\n".join(lines)


class FactorModel:
    """
    Unified interface for CAPM, FF3, and FF5 factor models.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL').
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    frequency : str
        Return frequency — 'daily' or 'monthly'. Default 'daily'.

    Examples
    --------
    >>> from factorlib import FactorModel
    >>> model = FactorModel('AAPL', '2018-01-01', '2024-01-01')
    >>> model.fit('FF5')
    >>> model.summary()
    >>> model.plot_loadings()
    >>> model.compare()
    """

    def __init__(
        self,
        ticker: str,
        start: str,
        end: str,
        frequency: Literal["daily", "monthly"] = "daily",
    ) -> None:
        self.ticker    = ticker.upper()
        self.start     = start
        self.end       = end
        self.frequency = frequency

        self._results:  dict[ModelType, RegressionResult] = {}
        self._data:     pd.DataFrame | None = None
        self._loaded:   bool = False

    # ------------------------------------------------------------------ #
    #  Data                                                                #
    # ------------------------------------------------------------------ #

    def _load_data(self) -> None:
        """Fetch and align price + factor data (lazy, called on first fit)."""
        prices  = fetch_prices(self.ticker, self.start, self.end, self.frequency)
        factors = fetch_ff_factors(self.start, self.end, self.frequency)

        data = factors.join(prices, how="inner").dropna()

        # excess return = stock return − risk-free rate
        data["excess_return"] = data["stock_return"] - data["RF"]
        self._data   = data
        self._loaded = True

    # ------------------------------------------------------------------ #
    #  Fitting                                                             #
    # ------------------------------------------------------------------ #

    def fit(self, model: ModelType = "FF5") -> "FactorModel":
        """
        Fit a factor model via OLS.

        Parameters
        ----------
        model : 'CAPM' | 'FF3' | 'FF5'

        Returns
        -------
        self  (allows method chaining)
        """
        if model not in FACTOR_COLS:
            raise ValueError(f"model must be one of {list(FACTOR_COLS)}. Got '{model}'.")

        if not self._loaded:
            self._load_data()

        y = self._data["excess_return"]
        X = sm.add_constant(self._data[FACTOR_COLS[model]])
        X.columns = ["Alpha"] + FACTOR_COLS[model]

        res = sm.OLS(y, X).fit()

        self._results[model] = RegressionResult(
            model_type    = model,
            ticker        = self.ticker,
            alpha         = float(res.params["Alpha"]),
            betas         = {f: float(res.params[f]) for f in FACTOR_COLS[model]},
            r_squared     = float(res.rsquared),
            adj_r_squared = float(res.rsquared_adj),
            f_statistic   = float(res.fvalue),
            f_pvalue      = float(res.f_pvalue),
            residuals     = res.resid,
            fitted_values = res.fittedvalues,
            params        = res.params,
            pvalues       = res.pvalues,
            tvalues       = res.tvalues,
            conf_int      = res.conf_int(),
            nobs          = int(res.nobs),
            _sm_result    = res,
        )
        return self

    def fit_all(self) -> "FactorModel":
        """Fit CAPM, FF3, and FF5 in one call."""
        for m in ("CAPM", "FF3", "FF5"):
            self.fit(m)
        return self

    # ------------------------------------------------------------------ #
    #  Results access                                                      #
    # ------------------------------------------------------------------ #

    def _get_result(self, model: ModelType) -> RegressionResult:
        if model not in self._results:
            raise RuntimeError(
                f"Model '{model}' has not been fitted yet. Call model.fit('{model}') first."
            )
        return self._results[model]

    @property
    def alpha(self) -> float:
        """Daily alpha of the most recently fitted model."""
        return list(self._results.values())[-1].alpha

    @property
    def r_squared(self) -> float:
        return list(self._results.values())[-1].r_squared

    @property
    def residuals(self) -> pd.Series:
        return list(self._results.values())[-1].residuals

    # ------------------------------------------------------------------ #
    #  Output                                                              #
    # ------------------------------------------------------------------ #

    def summary(self, model: ModelType | None = None) -> None:
        """Print a formatted regression summary."""
        if model:
            print(self._get_result(model))
        else:
            for r in self._results.values():
                print(r)

    def metrics(self, model: ModelType | None = None) -> pd.DataFrame:
        """Return a DataFrame of performance metrics (Sharpe, IR, etc.)."""
        targets = [model] if model else list(self._results)
        rows = []
        for m in targets:
            r = self._get_result(m)
            rows.append({"model": m, **compute_metrics(r, self._data)})
        return pd.DataFrame(rows).set_index("model")

    # ------------------------------------------------------------------ #
    #  Plots                                                               #
    # ------------------------------------------------------------------ #

    def plot_loadings(self, model: ModelType = "FF5") -> None:
        """Bar chart of factor betas with confidence intervals."""
        plot_loadings(self._get_result(model))

    def plot_residuals(self, model: ModelType = "FF5") -> None:
        """Residual diagnostics: time series, histogram, Q-Q plot, ACF."""
        plot_residuals(self._get_result(model))

    def plot_rolling_alpha(self, model: ModelType = "FF5", window: int = 126) -> None:
        """Rolling alpha (annualised) over a given window."""
        plot_rolling_alpha(self._data, self._get_result(model), window)

    def compare(self) -> None:
        """Side-by-side comparison of all fitted models."""
        if len(self._results) < 2:
            warnings.warn("Fit at least 2 models before calling compare(). Use fit_all().")
            return
        plot_model_comparison(self._results)

    def __repr__(self) -> str:
        fitted = list(self._results) or ["none fitted"]
        return f"FactorModel(ticker='{self.ticker}', fitted={fitted})"
