from __future__ import annotations

import copy

import pytest
from pydemetra._models import Span
from pydemetra._proto import tramoseats_pb2


class TestTramoSpecRoundTrip:
    def _minimal_tramo_spec(self) -> dict:
        return {
            "basic": {
                "span": Span(type="ALL", d0=None, d1=None, n0=0, n1=0),
                "preliminaryCheck": True,
            },
            "transform": {
                "fn": "AUTO",
                "fct": 0.0,
                "adjust": "NONE",
                "outliers": True,
            },
            "outlier": {
                "enabled": True,
                "span": Span(type="ALL", d0=None, d1=None, n0=0, n1=0),
                "ao": True,
                "ls": True,
                "tc": False,
                "so": False,
                "va": 0.0,
                "tcrate": 0.7,
                "ml": False,
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
                "acceptdef": True,
                "cancel": 0.05,
                "ub1": 0.97,
                "ub2": 0.91,
                "pcr": 0.95,
                "pc": 0.12,
                "tsig": 1.0,
                "amicompare": False,
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
                    "test": "TEST_NO",
                    "auto": "AUTO_NO",
                    "ptest": 0.01,
                    "autoadjust": True,
                    "tdcoefficients": [],
                    "lpcoefficient": None,
                },
                "easter": {
                    "type": "UNUSED",
                    "duration": 0,
                    "julian": False,
                    "test": False,
                    "coefficient": None,
                },
                "outliers": None,
                "users": None,
                "interventions": None,
                "ramps": None,
            },
            "estimate": {
                "span": Span(type="ALL", d0=None, d1=None, n0=0, n1=0),
                "ml": True,
                "tol": 1e-7,
                "ubp": 0.96,
            },
        }

    def test_round_trip_preserves_basic(self):
        from pydemetra.tramoseats import _p2r_spec_tramo, _r2p_spec_tramo

        spec = self._minimal_tramo_spec()
        p = _r2p_spec_tramo(spec)
        result = _p2r_spec_tramo(p)
        assert result["basic"]["preliminaryCheck"] is True

    def test_round_trip_preserves_outlier(self):
        from pydemetra.tramoseats import _p2r_spec_tramo, _r2p_spec_tramo

        spec = self._minimal_tramo_spec()
        p = _r2p_spec_tramo(spec)
        result = _p2r_spec_tramo(p)
        assert result["outlier"]["enabled"] is True
        assert result["outlier"]["ao"] is True
        assert result["outlier"]["ls"] is True
        assert result["outlier"]["tc"] is False
        assert result["outlier"]["tcrate"] == pytest.approx(0.7)

    def test_round_trip_preserves_automodel(self):
        from pydemetra.tramoseats import _p2r_spec_tramo, _r2p_spec_tramo

        spec = self._minimal_tramo_spec()
        p = _r2p_spec_tramo(spec)
        result = _p2r_spec_tramo(p)
        assert result["automodel"]["enabled"] is True
        assert result["automodel"]["cancel"] == pytest.approx(0.05)
        assert result["automodel"]["tsig"] == pytest.approx(1.0)

    def test_round_trip_preserves_transform(self):
        from pydemetra.tramoseats import _p2r_spec_tramo, _r2p_spec_tramo

        spec = self._minimal_tramo_spec()
        p = _r2p_spec_tramo(spec)
        result = _p2r_spec_tramo(p)
        assert result["transform"]["fn"] == "AUTO"
        assert result["transform"]["outliers"] is True

    def test_round_trip_preserves_estimate(self):
        from pydemetra.tramoseats import _p2r_spec_tramo, _r2p_spec_tramo

        spec = self._minimal_tramo_spec()
        p = _r2p_spec_tramo(spec)
        result = _p2r_spec_tramo(p)
        assert result["estimate"]["ml"] is True
        assert result["estimate"]["tol"] == pytest.approx(1e-7)
        assert result["estimate"]["ubp"] == pytest.approx(0.96)


class TestSeatsSpecRoundTrip:
    def _default_seats_spec(self) -> dict:
        return {
            "xl": 0.95,
            "approximation": "APP_NONE",
            "epsphi": 2.0,
            "rmod": 0.5,
            "sbound": 0.8,
            "sboundatpi": 0.8,
            "bias": False,
            "nfcasts": -1,
            "nbcasts": 0,
            "algorithm": "ALG_BURMAN",
        }

    def test_round_trip(self):
        from pydemetra.tramoseats import _p2r_spec_seats, _r2p_spec_seats

        spec = self._default_seats_spec()
        p = _r2p_spec_seats(spec)
        result = _p2r_spec_seats(p)
        assert result["xl"] == pytest.approx(0.95)
        assert result["approximation"] == "APP_NONE"
        assert result["epsphi"] == pytest.approx(2.0)
        assert result["rmod"] == pytest.approx(0.5)
        assert result["sbound"] == pytest.approx(0.8)
        assert result["sboundatpi"] == pytest.approx(0.8)
        assert result["bias"] is False
        assert result["nfcasts"] == -1
        assert result["nbcasts"] == 0
        assert result["algorithm"] == "ALG_BURMAN"


class TestTramoSeatsSpecRoundTrip:
    def test_has_expected_keys(self):
        from pydemetra.tramoseats import _p2r_spec_tramoseats, _r2p_spec_tramoseats

        spec = {
            "tramo": TestTramoSpecRoundTrip()._minimal_tramo_spec(),
            "seats": TestSeatsSpecRoundTrip()._default_seats_spec(),
            "benchmarking": {
                "enabled": False,
                "target": "BENCH_TARGET_UNSPECIFIED",
                "lambda": 1.0,
                "rho": 1.0,
                "bias": "BENCH_BIAS_NONE",
                "forecast": False,
            },
        }
        p = _r2p_spec_tramoseats(spec)
        result = _p2r_spec_tramoseats(p)
        assert set(result.keys()) == {"tramo", "seats", "benchmarking"}


class TestSetSeats:
    def _default_seats_spec(self) -> dict:
        return {
            "xl": 0.95,
            "approximation": "APP_NONE",
            "epsphi": 2.0,
            "rmod": 0.5,
            "sbound": 0.8,
            "sboundatpi": 0.8,
            "bias": False,
            "nfcasts": -1,
            "nbcasts": 0,
            "algorithm": "ALG_BURMAN",
        }

    def test_set_approximation(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        result = set_seats(spec, approximation="Legacy")
        assert result["approximation"] == "APP_LEGACY"

    def test_set_trend_boundary(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        result = set_seats(spec, trend_boundary=0.6)
        assert result["rmod"] == 0.6

    def test_set_seas_boundary(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        result = set_seats(spec, seas_boundary=0.9)
        assert result["sbound"] == 0.9

    def test_set_seas_boundary_unique(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        result = set_seats(spec, seas_boundary_unique=0.7)
        assert result["sboundatpi"] == 0.7

    def test_set_seas_tolerance(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        result = set_seats(spec, seas_tolerance=3.0)
        assert result["epsphi"] == 3.0

    def test_set_ma_boundary(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        result = set_seats(spec, ma_boundary=0.98)
        assert result["xl"] == 0.98

    def test_set_fcasts(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        result = set_seats(spec, fcasts=-2, bcasts=-1)
        assert result["nfcasts"] == -2
        assert result["nbcasts"] == -1

    def test_set_algorithm(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        result = set_seats(spec, algorithm="KalmanSmoother")
        assert result["algorithm"] == "ALG_KALMANSMOOTHER"

    def test_set_bias(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        result = set_seats(spec, bias=True)
        assert result["bias"] is True

    def test_does_not_mutate(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        original = copy.deepcopy(spec)
        set_seats(spec, approximation="Noisy")
        assert spec == original

    def test_works_on_tramoseats_spec(self):
        from pydemetra.tramoseats import set_seats

        ts_spec = {
            "tramo": {},
            "seats": self._default_seats_spec(),
            "benchmarking": {},
        }
        result = set_seats(ts_spec, trend_boundary=0.7)
        assert result["seats"]["rmod"] == 0.7
        assert result["tramo"] == {}

    def test_invalid_approximation_raises(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        with pytest.raises(ValueError, match="Invalid approximation"):
            set_seats(spec, approximation="INVALID")

    def test_invalid_algorithm_raises(self):
        from pydemetra.tramoseats import set_seats

        spec = self._default_seats_spec()
        with pytest.raises(ValueError, match="Invalid algorithm"):
            set_seats(spec, algorithm="INVALID")


class TestSeatsResultExtraction:
    def test_seats_rslts_keys(self):
        from pydemetra.tramoseats import _p2r_seats_rslts

        p = tramoseats_pb2.SeatsResults()
        result = _p2r_seats_rslts(p)
        assert set(result.keys()) == {
            "seatsmodel",
            "canonicaldecomposition",
            "stochastics",
        }

    def test_tramoseats_rslts_keys(self):
        from pydemetra.tramoseats import _p2r_tramoseats_rslts

        p = tramoseats_pb2.TramoSeatsResults()
        result = _p2r_tramoseats_rslts(p)
        assert set(result.keys()) == {
            "preprocessing",
            "decomposition",
            "final",
            "diagnostics",
        }
