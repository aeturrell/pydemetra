from __future__ import annotations

from pydemetra.spec.benchmarking import set_benchmarking
from pydemetra.spec.regarima import add_outlier, add_ramp, remove_outlier, remove_ramp


class TestAddOutlier:
    def test_add_single(self):
        spec = add_outlier({}, type="AO", date="2020-01-01")
        assert len(spec["outliers"]) == 1
        assert spec["outliers"][0]["type"] == "AO"
        assert spec["outliers"][0]["name"] == "AO.2020-01-01"

    def test_add_with_name(self):
        spec = add_outlier({}, type="LS", date="2020-06-01", name="my_outlier")
        assert spec["outliers"][0]["name"] == "my_outlier"

    def test_does_not_mutate(self):
        original = {"outliers": [{"type": "AO", "date": "2020-01-01", "name": "x", "coef": 0}]}
        _ = add_outlier(original, type="LS", date="2020-06-01")
        assert len(original["outliers"]) == 1


class TestRemoveOutlier:
    def test_remove_by_type(self):
        spec = {
            "outliers": [
                {"type": "AO", "date": "2020-01-01", "name": "a"},
                {"type": "LS", "date": "2020-06-01", "name": "b"},
            ]
        }
        result = remove_outlier(spec, type="AO")
        assert len(result["outliers"]) == 1
        assert result["outliers"][0]["type"] == "LS"

    def test_remove_by_date(self):
        spec = {
            "outliers": [
                {"type": "AO", "date": "2020-01-01", "name": "a"},
            ]
        }
        result = remove_outlier(spec, date="2020-01-01")
        assert len(result["outliers"]) == 0

    def test_no_outliers(self):
        result = remove_outlier({}, type="AO")
        assert "outliers" not in result


class TestAddRamp:
    def test_add_ramp(self):
        spec = add_ramp({}, start="2020-01-01", end="2020-12-01")
        assert len(spec["ramps"]) == 1
        assert spec["ramps"][0]["name"] == "ramp.2020-01-01.2020-12-01"


class TestRemoveRamp:
    def test_remove_by_start(self):
        spec = {"ramps": [{"start": "2020-01-01", "end": "2020-12-01", "name": "r"}]}
        result = remove_ramp(spec, start="2020-01-01")
        assert len(result["ramps"]) == 0


class TestSetBenchmarking:
    def test_set_values(self):
        spec = set_benchmarking({}, enabled=True, target="Original", rho=1.0, lambda_=0.5)
        assert spec["benchmarking"]["enabled"] is True
        assert spec["benchmarking"]["target"] == "Original"
        assert spec["benchmarking"]["rho"] == 1.0
        assert spec["benchmarking"]["lambda"] == 0.5

    def test_does_not_mutate(self):
        original = {"benchmarking": {"enabled": False}}
        _ = set_benchmarking(original, enabled=True)
        assert original["benchmarking"]["enabled"] is False
