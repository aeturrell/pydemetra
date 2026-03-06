from __future__ import annotations

from pydemetra._models import (
    ArimaModel,
    ChainedCalendar,
    EasterDay,
    FixedDay,
    FixedWeekDay,
    Likelihood,
    NationalCalendar,
    Parameter,
    SaDecomposition,
    SarimaModel,
    SingleDay,
    Span,
    SpecialDay,
    StatisticalTest,
    UcarimaModel,
    WeightedCalendar,
)


class TestStatisticalTest:
    def test_creation(self):
        t = StatisticalTest(value=1.5, pvalue=0.05, distribution="chi2")
        assert t.value == 1.5
        assert t.pvalue == 0.05


class TestParameter:
    def test_creation(self):
        p = Parameter(value=0.5, type="Estimated")
        assert p.value == 0.5


class TestSpan:
    def test_defaults(self):
        s = Span()
        assert s.type == "ALL"
        assert s.d0 is None
        assert s.d1 is None


class TestLikelihood:
    def test_creation(self):
        ll = Likelihood(
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
        )
        assert ll.nobs == 100
        assert ll.aic == 410.0


class TestSarimaModel:
    def test_defaults(self):
        m = SarimaModel(period=12)
        assert m.period == 12
        assert m.d == 0
        assert m.bd == 0

    def test_custom(self):
        m = SarimaModel(
            period=12,
            d=1,
            bd=1,
            phi=[0.5],
            theta=[-0.3],
            bphi=[0.2],
            btheta=[-0.1],
        )
        assert m.phi == [0.5]
        assert m.d == 1


class TestArimaModel:
    def test_creation(self):
        m = ArimaModel(
            ar=[1.0, -0.5],
            delta=[1.0, -1.0],
            ma=[1.0, 0.3],
            var=1.0,
        )
        assert m.var == 1.0


class TestUcarimaModel:
    def test_creation(self):
        comp = ArimaModel()
        u = UcarimaModel(model=comp, components=[comp], complements=[comp])
        assert len(u.components) == 1


class TestCalendarModels:
    def test_fixed_day(self):
        f = FixedDay(month=1, day=1)
        assert f.month == 1

    def test_fixed_week_day(self):
        f = FixedWeekDay(month=3, dayofweek=1, week=2)
        assert f.week == 2

    def test_easter_day(self):
        e = EasterDay(offset=0)
        assert e.offset == 0

    def test_special_day(self):
        s = SpecialDay(event="NEWYEAR", offset=0, weight=1.0, validity=None)
        assert s.event == "NEWYEAR"

    def test_single_day(self):
        s = SingleDay(date="2020-01-01")
        assert s.date == "2020-01-01"

    def test_national_calendar(self):
        nc = NationalCalendar(days=[FixedDay(month=1, day=1)])
        assert len(nc.days) == 1

    def test_chained_calendar(self):
        nc1 = NationalCalendar(days=[])
        nc2 = NationalCalendar(days=[])
        cc = ChainedCalendar(calendar1=nc1, calendar2=nc2, break_date="2020-01-01")
        assert cc.break_date == "2020-01-01"

    def test_weighted_calendar(self):
        wc = WeightedCalendar(calendars=["a", "b"], weights=[0.5, 0.5])
        assert sum(wc.weights) == 1.0


class TestSaDecomposition:
    def test_creation(self):
        d = SaDecomposition(mode="ADDITIVE", series=None, sa=None, t=None, s=None, i=None)
        assert d.mode == "ADDITIVE"
