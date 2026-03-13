"""
ffcapm
---------
Equity return decomposition using CAPM, Fama-French 3-Factor,
and Fama-French 5-Factor models.

Quick start
-----------
>>> from ffcapm import FactorModel
>>> model = FactorModel('AAPL', '2018-01-01', '2024-01-01')
>>> model.fit_all()
>>> model.summary()
>>> model.compare()
"""

from ffcapm.models import FactorModel, RegressionResult
from ffcapm.data import fetch_prices, fetch_ff_factors
from ffcapm.metrics import compute_metrics

__version__ = "0.1.0"
__author__  = "Mohit Appari"
__all__ = [
    "FactorModel",
    "RegressionResult",
    "fetch_prices",
    "fetch_ff_factors",
    "compute_metrics",
]
