"""
Tests for factorlib.models and factorlib.metrics.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from factorlib.metrics import compute_metrics


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_result():
    """Minimal mock of RegressionResult for metric tests."""
    class MockResult:
        model_type    = "FF5"
        ticker        = "TEST"
        alpha         = 0.0002
        r_squared     = 0.85
        adj_r_squared = 0.849
        residuals     = pd.Series(np.random.normal(0, 0.01, 500))
        nobs          = 500

    return MockResult()


@pytest.fixture
def mock_data(mock_result):
    """Aligned mock data DataFrame."""
    n   = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "excess_return": rng.normal(0.0003, 0.012, n),
        "RF":            rng.uniform(0.00005, 0.00015, n),
    }, index=idx)


# ── Metric tests ─────────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_returns_dict(self, mock_result, mock_data):
        m = compute_metrics(mock_result, mock_data)
        assert isinstance(m, dict)

    def test_expected_keys(self, mock_result, mock_data):
        m = compute_metrics(mock_result, mock_data)
        expected = {
            "alpha_ann", "r_squared", "adj_r_squared",
            "sharpe", "sortino", "info_ratio",
            "tracking_error", "max_drawdown", "jb_pvalue", "nobs",
        }
        assert expected.issubset(m.keys())

    def test_alpha_annualisation(self, mock_result, mock_data):
        m = compute_metrics(mock_result, mock_data)
        assert abs(m["alpha_ann"] - mock_result.alpha * 252) < 1e-8

    def test_max_drawdown_negative(self, mock_result, mock_data):
        m = compute_metrics(mock_result, mock_data)
        assert m["max_drawdown"] <= 0

    def test_sharpe_finite(self, mock_result, mock_data):
        m = compute_metrics(mock_result, mock_data)
        assert np.isfinite(m["sharpe"])

    def test_r_squared_passthrough(self, mock_result, mock_data):
        m = compute_metrics(mock_result, mock_data)
        assert m["r_squared"] == mock_result.r_squared


# ── Model type validation ─────────────────────────────────────────────────

class TestFactorModelValidation:

    def test_invalid_model_type(self):
        """fit() should raise ValueError for unknown model strings."""
        from factorlib import FactorModel
        model = FactorModel("AAPL", "2020-01-01", "2021-01-01")
        with pytest.raises(ValueError, match="model must be one of"):
            model.fit("FF7")

    def test_summary_before_fit(self):
        """_get_result() should raise RuntimeError if model not fitted."""
        from factorlib import FactorModel
        model = FactorModel("AAPL", "2020-01-01", "2021-01-01")
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model._get_result("FF5")

    def test_repr(self):
        from factorlib import FactorModel
        model = FactorModel("MSFT", "2020-01-01", "2021-01-01")
        assert "MSFT" in repr(model)
        assert "none fitted" in repr(model)
