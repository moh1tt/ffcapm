"""
factorlib.metrics
-----------------
Risk-adjusted performance metrics derived from factor model output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_metrics(result, data: pd.DataFrame) -> dict:
    """
    Compute a full set of performance metrics for a fitted model.

    Parameters
    ----------
    result : RegressionResult
    data   : aligned DataFrame (contains 'excess_return', 'RF')

    Returns
    -------
    dict of metric name → value
    """
    resid     = result.residuals
    excess    = data.loc[resid.index, "excess_return"]
    rf_daily  = data.loc[resid.index, "RF"]

    periods = len(excess)
    ann     = 252 if periods > 100 else 12   # daily vs monthly

    # ── annualised alpha ────────────────────────────────────────────────
    alpha_ann = result.alpha * ann

    # ── information ratio (alpha / tracking error) ─────────────────────
    te  = resid.std() * np.sqrt(ann)
    ir  = alpha_ann / te if te > 0 else np.nan

    # ── Sharpe ratio on total excess return ────────────────────────────
    mu_ann    = excess.mean() * ann
    sigma_ann = excess.std() * np.sqrt(ann)
    sharpe    = mu_ann / sigma_ann if sigma_ann > 0 else np.nan

    # ── Sortino ratio (downside deviation only) ─────────────────────────
    downside  = excess[excess < 0].std() * np.sqrt(ann)
    sortino   = mu_ann / downside if downside > 0 else np.nan

    # ── max drawdown ────────────────────────────────────────────────────
    cum_returns = (1 + excess).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown    = (cum_returns - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # ── residual normality (Jarque-Bera) ───────────────────────────────
    jb_stat, jb_pval = stats.jarque_bera(resid)

    return {
        "alpha_ann":     round(alpha_ann, 6),
        "r_squared":     round(result.r_squared, 4),
        "adj_r_squared": round(result.adj_r_squared, 4),
        "sharpe":        round(sharpe, 4),
        "sortino":       round(sortino, 4),
        "info_ratio":    round(ir, 4),
        "tracking_error":round(te, 6),
        "max_drawdown":  round(max_dd, 4),
        "jb_pvalue":     round(jb_pval, 4),
        "nobs":          result.nobs,
    }
