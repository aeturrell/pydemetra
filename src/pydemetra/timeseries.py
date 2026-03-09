from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from pydemetra._converters import jd2r_tsdata, r2jd_tsdata, r2jd_tsdomain
from pydemetra._java import _ensure_jvm


def _ts_params(
    s: pd.Series | None,
    frequency: int | None,
    start: tuple[int, int] | None,
    length: int | None,
):
    """Extract frequency, start, and length from a Series or explicit parameters."""
    if isinstance(start, int):
        start = (start, 1)
    if s is not None and isinstance(s.index, pd.PeriodIndex):
        freq_str = s.index.freqstr
        if "M" in freq_str:
            frequency = 12
            start = (s.index[0].year, s.index[0].month)
        elif "Q" in freq_str:
            frequency = 4
            start = (s.index[0].year, s.index[0].quarter)
        else:
            frequency = 1
            start = (s.index[0].year, 1)
        length = len(s)
    return frequency, start, length


def aggregate(
    s: pd.Series,
    nfreq: int = 1,
    conversion: str = "Sum",
    complete: bool = True,
) -> pd.Series | None:
    """Aggregate a time series to a lower frequency.

    Args:
        s: Input time series with PeriodIndex.
        nfreq: New frequency (must divide the original frequency).
        conversion: ``"Sum"``, ``"Average"``, ``"First"``, ``"Last"``, ``"Min"``, ``"Max"``.
        complete: If True, incomplete boundary periods are set to NaN.

    Returns:
        Aggregated series at the new frequency.
    """
    _ensure_jvm()
    import jpype

    jd_s = r2jd_tsdata(s)
    TsUtility = jpype.JClass("jdplus.toolkit.base.r.timeseries.TsUtility")
    jd_agg = TsUtility.aggregate(jd_s, int(nfreq), conversion, complete)
    if jd_agg is None:
        return None
    return jd2r_tsdata(jd_agg)


def clean_extremities(s: pd.Series) -> pd.Series | None:
    """Remove missing values at the beginning and end of a series.

    Args:
        s: Input time series.

    Returns:
        Cleaned series without leading/trailing NaN values.
    """
    _ensure_jvm()
    import jpype

    jd_s = r2jd_tsdata(s)
    TsUtility = jpype.JClass("jdplus.toolkit.base.r.timeseries.TsUtility")
    jd_cleaned = TsUtility.cleanExtremities(jd_s)
    if jd_cleaned is None:
        return None
    return jd2r_tsdata(jd_cleaned)


def ts_interpolate(s: pd.Series, method: str = "airline") -> pd.Series | None:
    """Interpolate missing values in a time series.

    Args:
        s: Input time series with missing values.
        method: ``"airline"`` or ``"average"``.

    Returns:
        Interpolated series.
    """
    _ensure_jvm()
    import jpype

    jd_s = r2jd_tsdata(s)
    Interpolation = jpype.JClass("jdplus.toolkit.base.r.modelling.Interpolation")
    if method == "airline":
        jd_si = Interpolation.airlineInterpolation(jd_s)
    elif method == "average":
        jd_si = Interpolation.averageInterpolation(jd_s)
    else:
        raise ValueError(f"Unknown method: {method}")
    return jd2r_tsdata(jd_si)


def ts_adjust(
    s: pd.Series,
    method: str = "LeapYear",
    reverse: bool = False,
) -> pd.Series | None:
    """Multiplicative adjustment for leap year or length of periods.

    Args:
        s: Input time series.
        method: ``"LeapYear"`` or ``"LengthOfPeriod"``.
        reverse: If True, reverse the adjustment.

    Returns:
        Adjusted series.
    """
    _ensure_jvm()
    import jpype

    jd_s = r2jd_tsdata(s)
    Transformation = jpype.JClass("jdplus.toolkit.base.r.modelling.Transformation")
    jd_st = Transformation.adjust(jd_s, method, reverse)
    if jd_st is None:
        return None
    return jd2r_tsdata(jd_st)


def days_of(s: pd.Series, pos: int = 1) -> list[datetime.date]:
    """Return the starting dates for each period of a time series.

    Args:
        s: Input time series.
        pos: Position of the first considered period.

    Returns:
        List of dates.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, None, None, None)
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    TsUtility = jpype.JClass("jdplus.toolkit.base.r.timeseries.TsUtility")
    days = TsUtility.daysOf(jdom, int(pos - 1))
    return [datetime.date.fromisoformat(str(d)) for d in days]


def tsdata_of(values: list[float], dates: list[str]) -> pd.Series | None:
    """Create a time series from values and dates.

    Args:
        values: Numeric values.
        dates: Date strings in ``"YYYY-MM-DD"`` format.

    Returns:
        A pandas Series with automatically detected frequency.
    """
    _ensure_jvm()
    import jpype

    TsDataCollector = jpype.JClass("jdplus.toolkit.base.r.timeseries.TsDataCollector")
    jtsdata = TsDataCollector.of(
        np.array(values, dtype=np.float64),
        [str(d) for d in dates],
    )
    return jd2r_tsdata(jtsdata)


def compare_annual_totals(raw: pd.Series, sa: pd.Series) -> float:
    """Compare the annual totals of raw and seasonally adjusted series.

    Args:
        raw: Raw time series.
        sa: Seasonally adjusted time series.

    Returns:
        Largest annual difference as a percentage of SA average.
    """
    _ensure_jvm()
    import jpype

    jsa = r2jd_tsdata(sa)
    jraw = r2jd_tsdata(raw)
    SaUtility = jpype.JClass("jdplus.sa.base.r.SaUtility")
    return float(SaUtility.compareAnnualTotals(jraw, jsa))
