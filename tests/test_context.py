from __future__ import annotations

from pydemetra.context import modelling_context


class TestModellingContext:
    def test_empty(self):
        ctx = modelling_context()
        assert ctx["calendars"] == {}
        assert ctx["variables"] == {}

    def test_with_data(self):
        ctx = modelling_context(calendars={"cal1": "some_cal"}, variables={"v1": "some_var"})
        assert ctx["calendars"]["cal1"] == "some_cal"
        assert ctx["variables"]["v1"] == "some_var"
