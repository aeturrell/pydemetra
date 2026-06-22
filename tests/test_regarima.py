from __future__ import annotations

import numpy as np
from pydemetra._models import Likelihood
from pydemetra.regarima import (
    regarima_coef,
    regarima_loglik,
    regarima_residuals,
    regarima_summary,
    regarima_vcov,
)


def _make_result():
    return {
        "b": np.array([1.0, 2.0, 3.0]),
        "bcovariance": np.eye(3),
        "parameters": {
            "val": np.array([0.5, -0.3]),
            "cov": np.eye(2) * 0.01,
        },
        "residuals": np.array([0.1, -0.2, 0.05]),
        "likelihood": Likelihood(
            nobs=100,
            neffectiveobs=95,
            nparams=5,
            ll=-200.0,
            aic=410.0,
            aicc=411.0,
            bic=425.0,
            bicc=426.0,
            ssq=50.0,
            adjustedll=-195.0,
        ),
        "orders": {
            "order": (1, 1, 1),
            "seasonal": {"order": (0, 1, 1), "period": 12},
        },
    }


class TestRegarimaCoef:
    def test_regression(self):
        c = regarima_coef(_make_result(), "regression")
        np.testing.assert_array_equal(c, np.array([1.0, 2.0, 3.0]))

    def test_arima(self):
        c = regarima_coef(_make_result(), "arima")
        np.testing.assert_array_equal(c, np.array([0.5, -0.3]))

    def test_both(self):
        c = regarima_coef(_make_result(), "both")
        assert len(c) == 5


class TestRegarimaLoglik:
    def test_loglik(self):
        ll = regarima_loglik(_make_result())
        assert ll.ll == -200.0


class TestRegarimaVcov:
    def test_regression(self):
        v = regarima_vcov(_make_result(), "regression")
        assert v.shape == (3, 3)

    def test_arima(self):
        v = regarima_vcov(_make_result(), "arima")
        assert v.shape == (2, 2)


class TestRegarimaResiduals:
    def test_residuals(self):
        assert len(regarima_residuals(_make_result())) == 3


class TestRegarimaSummary:
    def test_summary_contains_arima(self):
        s = regarima_summary(_make_result())
        assert "ARIMA(1,1,1)(0,1,1)[12]" in s
        assert "Log-likelihood" in s
