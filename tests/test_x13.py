from __future__ import annotations

import copy

import pytest

from pydemetra.x13 import (
    _p2r_spec_regarima,
    _p2r_spec_x11,
    _p2r_spec_x13,
    _p2r_x11_rslts,
    _p2r_x13_final,
    _p2r_x13_preadjust,
    _r2p_spec_regarima,
    _r2p_spec_x11,
    _r2p_spec_x13,
    set_x11,
)
from pydemetra._proto import x13_pb2


class TestSetX11:
    def _default_x11_spec(self) -> dict:
        return {
            "mode": "UNKNOWN",
            "seasonal": True,
            "henderson": 0,
            "sfilters": [],
            "lsig": 1.5,
            "usig": 2.5,
            "nfcasts": -1,
            "nbcasts": 0,
            "sigma": "NONE",
            "vsigmas": [],
            "excludefcasts": False,
            "bias": "NONE",
        }

    def test_set_mode(self):
        spec = self._default_x11_spec()
        result = set_x11(spec, mode="Additive")
        assert result["mode"] == "ADDITIVE"

    def test_set_mode_undefined(self):
        spec = self._default_x11_spec()
        result = set_x11(spec, mode="Undefined")
        assert result["mode"] == "UNKNOWN"

    def test_invalid_mode_raises(self):
        spec = self._default_x11_spec()
        with pytest.raises(ValueError, match="Invalid mode"):
            set_x11(spec, mode="INVALID")

    def test_set_seasonal_comp(self):
        spec = self._default_x11_spec()
        result = set_x11(spec, seasonal_comp=False)
        assert result["seasonal"] is False

    def test_set_seasonal_filter(self):
        spec = self._default_x11_spec()
        result = set_x11(spec, seasonal_filter="S3X5")
        assert result["sfilters"] == ["FILTER_S3X5"]

    def test_set_seasonal_filter_list(self):
        spec = self._default_x11_spec()
        result = set_x11(spec, seasonal_filter=["S3X3", "Msr"])
        assert result["sfilters"] == ["FILTER_S3X3", "FILTER_MSR"]

    def test_set_henderson_filter(self):
        spec = self._default_x11_spec()
        result = set_x11(spec, henderson_filter=13)
        assert result["henderson"] == 13

    def test_henderson_even_raises(self):
        spec = self._default_x11_spec()
        with pytest.raises(ValueError, match="odd"):
            set_x11(spec, henderson_filter=12)

    def test_set_sigma_thresholds(self):
        spec = self._default_x11_spec()
        result = set_x11(spec, lsigma=1.7, usigma=2.7)
        assert result["lsig"] == 1.7
        assert result["usig"] == 2.7

    def test_set_casts(self):
        spec = self._default_x11_spec()
        result = set_x11(spec, fcasts=-2, bcasts=-1)
        assert result["nfcasts"] == -2
        assert result["nbcasts"] == -1

    def test_sigma_vector(self):
        spec = self._default_x11_spec()
        result = set_x11(spec, sigma_vector=[1, 2, 1, 2])
        assert result["sigma"] == "SELECT"
        assert result["vsigmas"] == [1, 2, 1, 2]

    def test_invalid_sigma_vector_raises(self):
        spec = self._default_x11_spec()
        with pytest.raises(ValueError, match="1 or 2"):
            set_x11(spec, sigma_vector=[1, 3])

    def test_does_not_mutate(self):
        spec = self._default_x11_spec()
        original = copy.deepcopy(spec)
        set_x11(spec, mode="Multiplicative")
        assert spec == original

    def test_works_on_x13_spec(self):
        x13_spec = {
            "regarima": {},
            "x11": self._default_x11_spec(),
            "benchmarking": {},
        }
        result = set_x11(x13_spec, henderson_filter=7)
        assert result["x11"]["henderson"] == 7
        assert result["regarima"] == {}


class TestX11SpecRoundTrip:
    def test_round_trip(self):
        original = {
            "mode": "ADDITIVE",
            "seasonal": True,
            "henderson": 13,
            "sfilters": ["FILTER_S3X5"],
            "lsig": 1.5,
            "usig": 2.5,
            "nfcasts": -1,
            "nbcasts": 0,
            "sigma": "NONE",
            "vsigmas": [],
            "excludefcasts": False,
            "bias": "NONE",
        }
        p = _r2p_spec_x11(original)
        result = _p2r_spec_x11(p)
        assert result["mode"] == "ADDITIVE"
        assert result["seasonal"] is True
        assert result["henderson"] == 13
        assert result["sfilters"] == ["FILTER_S3X5"]


class TestRegArimaSpecRoundTrip:
    def _minimal_regarima_spec(self) -> dict:
        from pydemetra._models import Span

        return {
            "basic": {
                "span": Span(type="ALL", d0=None, d1=None, n0=0, n1=0),
                "preprocessing": True,
                "preliminaryCheck": True,
            },
            "transform": {
                "fn": "AUTO",
                "adjust": "NONE",
                "aicdiff": -2.0,
                "outliers": True,
            },
            "outlier": {
                "outliers": [],
                "span": Span(type="ALL", d0=None, d1=None, n0=0, n1=0),
                "defva": 0.0,
                "method": "ADDONE",
                "monthlytcrate": 0.7,
                "maxiter": 0,
                "lsrun": 0,
            },
            "arima": {
                "period": 0,
                "d": 0,
                "bd": 0,
                "phi": [],
                "theta": [],
                "bphi": [],
                "btheta": [],
            },
            "automodel": {
                "enabled": True,
                "ljungbox": 0.95,
                "tsig": 1.0,
                "predcv": 0.14286,
                "ubfinal": 1.05,
                "ub1": 0.97,
                "ub2": 0.91,
                "cancel": 0.1,
                "fct": 0.0,
                "acceptdef": False,
                "mixed": True,
                "balanced": False,
            },
            "regression": {
                "mean": None,
                "check_mean": True,
                "td": {
                    "td": "TD_NONE",
                    "lp": "NONE",
                    "holidays": "",
                    "users": [],
                    "w": 0,
                    "test": "NO",
                    "auto": "AUTO_NO",
                    "autoadjust": True,
                    "tdcoefficients": [],
                    "lpcoefficient": None,
                    "ptest1": 0.01,
                    "ptest2": 0.01,
                },
                "easter": {
                    "type": "UNUSED",
                    "duration": 0,
                    "test": "NO",
                    "coefficient": None,
                },
                "outliers": None,
                "users": None,
                "interventions": None,
                "ramps": None,
            },
            "estimate": {
                "span": Span(type="ALL", d0=None, d1=None, n0=0, n1=0),
                "tol": 1e-7,
            },
        }

    def test_round_trip_preserves_automodel(self):
        spec = self._minimal_regarima_spec()
        p = _r2p_spec_regarima(spec)
        result = _p2r_spec_regarima(p)
        assert result["automodel"]["enabled"] is True
        assert result["automodel"]["ljungbox"] == pytest.approx(0.95)


class TestX13SpecRoundTrip:
    def test_x13_spec_has_expected_keys(self):
        from pydemetra._models import Span

        minimal_x11 = {
            "mode": "UNKNOWN",
            "seasonal": True,
            "henderson": 0,
            "sfilters": [],
            "lsig": 1.5,
            "usig": 2.5,
            "nfcasts": -1,
            "nbcasts": 0,
            "sigma": "NONE",
            "vsigmas": [],
            "excludefcasts": False,
            "bias": "NONE",
        }
        minimal_regarima = {
            "basic": {
                "span": Span(type="ALL", d0=None, d1=None, n0=0, n1=0),
                "preprocessing": True,
                "preliminaryCheck": True,
            },
            "transform": {"fn": "AUTO", "adjust": "NONE", "aicdiff": -2.0, "outliers": True},
            "outlier": {
                "outliers": [],
                "span": Span(type="ALL", d0=None, d1=None, n0=0, n1=0),
                "defva": 0.0, "method": "ADDONE", "monthlytcrate": 0.7,
                "maxiter": 0, "lsrun": 0,
            },
            "arima": {"period": 0, "d": 0, "bd": 0, "phi": [], "theta": [], "bphi": [], "btheta": []},
            "automodel": {
                "enabled": True, "ljungbox": 0.95, "tsig": 1.0, "predcv": 0.14286,
                "ubfinal": 1.05, "ub1": 0.97, "ub2": 0.91, "cancel": 0.1,
                "fct": 0.0, "acceptdef": False, "mixed": True, "balanced": False,
            },
            "regression": {
                "mean": None, "check_mean": True,
                "td": {
                    "td": "TD_NONE", "lp": "NONE", "holidays": "", "users": [], "w": 0,
                    "test": "NO", "auto": "AUTO_NO", "autoadjust": True,
                    "tdcoefficients": [], "lpcoefficient": None, "ptest1": 0.01, "ptest2": 0.01,
                },
                "easter": {"type": "UNUSED", "duration": 0, "test": "NO", "coefficient": None},
                "outliers": None, "users": None, "interventions": None, "ramps": None,
            },
            "estimate": {
                "span": Span(type="ALL", d0=None, d1=None, n0=0, n1=0),
                "tol": 1e-7,
            },
        }
        benchmarking = {
            "enabled": False, "target": "BENCH_TARGET_UNSPECIFIED", "lambda": 1.0,
            "rho": 1.0, "bias": "BENCH_BIAS_NONE", "forecast": False,
        }

        spec = {"regarima": minimal_regarima, "x11": minimal_x11, "benchmarking": benchmarking}
        p = _r2p_spec_x13(spec)
        result = _p2r_spec_x13(p)
        assert set(result.keys()) == {"regarima", "x11", "benchmarking"}
        assert set(result["x11"].keys()) == {
            "mode", "seasonal", "henderson", "sfilters", "lsig", "usig",
            "nfcasts", "nbcasts", "sigma", "vsigmas", "excludefcasts", "bias",
        }


class TestResultExtraction:
    def test_x11_rslts_keys(self):
        p = x13_pb2.X11Results()
        result = _p2r_x11_rslts(p)
        expected = {"d1", "d2", "d4", "d5", "d6", "d7", "d8", "d9",
                    "d10", "d11", "d12", "d13", "final_seasonal", "final_henderson"}
        assert set(result.keys()) == expected

    def test_x13_final_keys(self):
        p = x13_pb2.X13Finals()
        result = _p2r_x13_final(p)
        expected = {"d11final", "d12final", "d13final", "d16", "d18",
                    "d11a", "d12a", "d16a", "d18a", "e1", "e2", "e3", "e11"}
        assert set(result.keys()) == expected

    def test_x13_preadjust_keys(self):
        p = x13_pb2.X13Preadjustment()
        result = _p2r_x13_preadjust(p)
        expected = {"a1", "a1a", "a1b", "a6", "a7", "a8", "a9"}
        assert set(result.keys()) == expected
