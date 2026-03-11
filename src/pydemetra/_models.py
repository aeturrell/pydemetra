from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StatisticalTest:
    """Result of a statistical test."""

    value: float
    pvalue: float
    distribution: str | None = None

    def __repr__(self) -> str:
        parts = [f"Value: {self.value}", f"P-Value: {self.pvalue:.4f}"]
        if self.distribution:
            parts.append(f"[{self.distribution}]")
        return "\n".join(parts)


@dataclass
class Parameter:
    """A model parameter with its estimation type."""

    value: float
    type: str


@dataclass
class Span:
    """A time span selection."""

    type: str = "ALL"
    d0: datetime.date | None = None
    d1: datetime.date | None = None
    n0: int = 0
    n1: int = 0


@dataclass
class Likelihood:
    """Likelihood statistics from model estimation."""

    nobs: int = 0
    neffectiveobs: int = 0
    nparams: int = 0
    ll: float = 0.0
    adjustedll: float = 0.0
    aic: float = 0.0
    aicc: float = 0.0
    bic: float = 0.0
    bicc: float = 0.0
    ssq: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Likelihood(nobs={self.nobs}, ll={self.ll:.4f}, "
            f"aic={self.aic:.4f}, bic={self.bic:.4f})"
        )


@dataclass
class SarimaModel:
    """Seasonal ARIMA model (Box-Jenkins)."""

    name: str = "sarima"
    period: int = 1
    phi: list[float] = field(default_factory=list)
    d: int = 0
    theta: list[float] = field(default_factory=list)
    bphi: list[float] = field(default_factory=list)
    bd: int = 0
    btheta: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SARIMA({len(self.phi)},{self.d},{len(self.theta)})"
            f"({len(self.bphi)},{self.bd},{len(self.btheta)})[{self.period}]"
        )


@dataclass
class ArimaModel:
    """ARIMA model."""

    name: str = "arima"
    ar: list[float] = field(default_factory=lambda: [1.0])
    delta: list[float] = field(default_factory=lambda: [1.0])
    ma: list[float] = field(default_factory=lambda: [1.0])
    var: float = 1.0

    def __repr__(self) -> str:
        return f"ARIMA(ar={self.ar}, delta={self.delta}, ma={self.ma}, var={self.var})"


@dataclass
class UcarimaModel:
    """Unobserved Components ARIMA model."""

    model: ArimaModel | None = None
    components: list[ArimaModel] = field(default_factory=list)
    complements: list[ArimaModel] = field(default_factory=list)


@dataclass
class FixedDay:
    """Holiday on a fixed calendar day."""

    month: int
    day: int
    weight: float = 1.0
    validity: dict | None = None


@dataclass
class FixedWeekDay:
    """Holiday on a specific weekday occurrence in a month."""

    month: int
    week: int
    dayofweek: int
    weight: float = 1.0
    validity: dict | None = None


@dataclass
class EasterDay:
    """Holiday relative to Easter Sunday."""

    offset: int
    julian: bool = False
    weight: float = 1.0
    validity: dict | None = None


@dataclass
class SpecialDay:
    """Pre-defined holiday event."""

    event: str
    offset: int = 0
    weight: float = 1.0
    validity: dict | None = None


@dataclass
class SingleDay:
    """One-time holiday on a specific date."""

    date: str
    weight: float = 1.0


@dataclass
class NationalCalendar:
    """Calendar defined by a list of holidays."""

    days: list = field(default_factory=list)
    mean_correction: bool = True


@dataclass
class ChainedCalendar:
    """Two calendars chained at a break date."""

    calendar1: NationalCalendar | None = None
    calendar2: NationalCalendar | None = None
    break_date: str = ""


@dataclass
class WeightedCalendar:
    """Weighted combination of calendars."""

    calendars: list = field(default_factory=list)
    weights: list[float] = field(default_factory=list)


@dataclass
class SaDecomposition:
    """Seasonal adjustment decomposition result."""

    mode: str = ""
    series: pd.Series | None = None
    sa: pd.Series | None = None
    t: pd.Series | None = None
    s: pd.Series | None = None
    i: pd.Series | None = None
