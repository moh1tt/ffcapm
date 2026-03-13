"""
factorlib.plot
--------------
Visualisation functions for factor model results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


COLORS = {
    "CAPM": "#4C9BE8",
    "FF3":  "#7C6CE0",
    "FF5":  "#E07C4C",
}

_STYLE = {
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


# ────────────────────────────────────────────────────────────────────────
#  Factor loadings bar chart
# ────────────────────────────────────────────────────────────────────────

def plot_loadings(result) -> None:
    """
    Horizontal bar chart of factor betas with 95% confidence intervals.
    """
    _apply_style()
    betas   = result.betas
    ci      = result.conf_int.drop("Alpha")
    factors = list(betas)
    values  = [betas[f] for f in factors]
    lo      = [betas[f] - ci.loc[f, 0] for f in factors]
    hi      = [ci.loc[f, 1] - betas[f] for f in factors]

    color = COLORS.get(result.model_type, "#555")

    fig, ax = plt.subplots(figsize=(8, max(3, len(factors) * 0.9)))
    bars = ax.barh(factors, values, xerr=[lo, hi],
                   color=color, alpha=0.8, capsize=4, height=0.5,
                   error_kw={"elinewidth": 1.2, "ecolor": "#333"})
    ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")

    for bar, val in zip(bars, values):
        ax.text(val + 0.01 * np.sign(val), bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=9, color="#333")

    ax.set_xlabel("Beta (factor loading)", fontsize=10)
    ax.set_title(
        f"{result.ticker} — {result.model_type} factor loadings\n"
        f"α = {result.alpha*252:.4f} (ann.)  ·  R² = {result.r_squared:.3f}",
        fontsize=11, pad=12,
    )
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────────
#  Residual diagnostics
# ────────────────────────────────────────────────────────────────────────

def plot_residuals(result) -> None:
    """
    2×2 residual diagnostic panel:
      [time series]  [histogram + normal overlay]
      [Q-Q plot]     [ACF]
    """
    _apply_style()
    resid = result.residuals

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        f"{result.ticker} — {result.model_type} residual diagnostics",
        fontsize=12, y=1.01,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Time series
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(resid.index, resid.values, linewidth=0.6, color="#4C9BE8", alpha=0.8)
    ax1.axhline(0, color="#333", linewidth=0.8, linestyle="--")
    ax1.set_title("Residuals over time", fontsize=10)
    ax1.set_ylabel("Residual", fontsize=9)

    # 2. Histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(resid, bins=50, color="#7C6CE0", alpha=0.7, density=True, edgecolor="white")
    xr = np.linspace(resid.min(), resid.max(), 200)
    mu, sigma = resid.mean(), resid.std()
    ax2.plot(xr, stats.norm.pdf(xr, mu, sigma), "r-", linewidth=1.5, label="Normal fit")
    ax2.set_title("Residual distribution", fontsize=10)
    ax2.legend(fontsize=8)

    # 3. Q-Q plot
    ax3 = fig.add_subplot(gs[1, 0])
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax3.scatter(osm, osr, s=4, color="#E07C4C", alpha=0.6)
    ax3.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1.2)
    ax3.set_title("Q-Q plot (normality check)", fontsize=10)
    ax3.set_xlabel("Theoretical quantiles", fontsize=9)
    ax3.set_ylabel("Sample quantiles", fontsize=9)

    # 4. ACF (manual, no statsmodels dependency for plot)
    ax4 = fig.add_subplot(gs[1, 1])
    nlags = min(40, len(resid) // 5)
    acf_vals = [resid.autocorr(lag=i) for i in range(1, nlags + 1)]
    ci_bound = 1.96 / np.sqrt(len(resid))
    ax4.bar(range(1, nlags + 1), acf_vals, color="#4C9BE8", alpha=0.7, width=0.6)
    ax4.axhline( ci_bound, color="red", linestyle="--", linewidth=0.8)
    ax4.axhline(-ci_bound, color="red", linestyle="--", linewidth=0.8)
    ax4.set_title("Autocorrelation of residuals", fontsize=10)
    ax4.set_xlabel("Lag", fontsize=9)

    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────────
#  Rolling alpha
# ────────────────────────────────────────────────────────────────────────

def plot_rolling_alpha(data: pd.DataFrame, result, window: int = 126) -> None:
    """
    Rolling annualised alpha using a simple expanding OLS on the residuals.
    """
    from factorlib.models import FACTOR_COLS
    import statsmodels.api as sm

    _apply_style()
    model_type = result.model_type
    factors    = FACTOR_COLS[model_type]

    y = data["excess_return"]
    X = sm.add_constant(data[factors])
    X.columns = ["Alpha"] + factors

    rolling_alpha = []
    dates = []

    for i in range(window, len(y)):
        y_w = y.iloc[i - window:i]
        X_w = X.iloc[i - window:i]
        try:
            alpha = sm.OLS(y_w, X_w).fit().params["Alpha"] * 252
        except Exception:
            alpha = np.nan
        rolling_alpha.append(alpha)
        dates.append(y.index[i])

    series = pd.Series(rolling_alpha, index=dates)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, color=COLORS.get(model_type, "#555"), linewidth=1.2)
    ax.axhline(0, color="#333", linewidth=0.8, linestyle="--")
    ax.fill_between(series.index, series.values, 0,
                    where=series.values > 0, alpha=0.15, color="green")
    ax.fill_between(series.index, series.values, 0,
                    where=series.values < 0, alpha=0.15, color="red")
    ax.set_title(
        f"{result.ticker} — {model_type} rolling alpha (annualised, {window}-day window)",
        fontsize=11,
    )
    ax.set_ylabel("Alpha (annualised)", fontsize=10)
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────────
#  Model comparison
# ────────────────────────────────────────────────────────────────────────

def plot_model_comparison(results: dict) -> None:
    """
    Side-by-side comparison of R², adj-R², and annualised alpha across models.
    """
    _apply_style()
    models    = list(results)
    r2        = [results[m].r_squared     for m in models]
    adj_r2    = [results[m].adj_r_squared for m in models]
    alpha_ann = [results[m].alpha * 252   for m in models]
    colors    = [COLORS.get(m, "#888")    for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ticker = list(results.values())[0].ticker
    fig.suptitle(f"{ticker} — model comparison", fontsize=12, y=1.02)

    for ax, values, title, fmt in zip(
        axes,
        [r2, adj_r2, alpha_ann],
        ["R²", "Adjusted R²", "Alpha (annualised)"],
        [".3f", ".3f", ".4f"],
    ):
        bars = ax.bar(models, values, color=colors, alpha=0.8, width=0.5)
        ax.set_title(title, fontsize=10)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 0.1)

    plt.tight_layout()
    plt.show()
