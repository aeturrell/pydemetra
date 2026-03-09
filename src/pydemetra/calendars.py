from __future__ import annotations

import numpy as np
import pandas as pd

from pydemetra._converters import jd2r_matrix, r2jd_tsdomain
from pydemetra._java import _ensure_jvm
from pydemetra._models import (
    ChainedCalendar,
    EasterDay,
    FixedDay,
    FixedWeekDay,
    NationalCalendar,
    SingleDay,
    SpecialDay,
    WeightedCalendar,
)
from pydemetra.timeseries import _ts_params


def fixed_day(month: int, day: int, weight: float = 1.0, validity: dict | None = None) -> FixedDay:
    """Create a holiday on a fixed calendar day."""
    return FixedDay(month=month, day=day, weight=weight, validity=validity)


def fixed_week_day(
    month: int, week: int, dayofweek: int, weight: float = 1.0, validity: dict | None = None
) -> FixedWeekDay:
    """Create a holiday on a specific weekday occurrence in a month."""
    return FixedWeekDay(
        month=month, week=week, dayofweek=dayofweek, weight=weight, validity=validity
    )


def easter_day(
    offset: int, julian: bool = False, weight: float = 1.0, validity: dict | None = None
) -> EasterDay:
    """Create a holiday relative to Easter Sunday."""
    return EasterDay(offset=offset, julian=julian, weight=weight, validity=validity)


def single_day(date: str, weight: float = 1.0) -> SingleDay:
    """Create a one-time holiday on a specific date."""
    return SingleDay(date=date, weight=weight)


def special_day(
    event: str, offset: int = 0, weight: float = 1.0, validity: dict | None = None
) -> SpecialDay:
    """Create a holiday from a pre-defined event."""
    return SpecialDay(event=event, offset=offset, weight=weight, validity=validity)


def national_calendar(days: list | None = None, mean_correction: bool = True) -> NationalCalendar:
    """Create a national calendar from a list of holidays."""
    return NationalCalendar(days=days or [], mean_correction=mean_correction)


def chained_calendar(
    calendar1: NationalCalendar, calendar2: NationalCalendar, break_date: str
) -> ChainedCalendar:
    """Create a calendar chaining two calendars at a break date."""
    return ChainedCalendar(calendar1=calendar1, calendar2=calendar2, break_date=break_date)


def weighted_calendar(calendars: list, weights: list[float]) -> WeightedCalendar:
    """Create a weighted combination of calendars."""
    if len(calendars) != len(weights):
        raise ValueError("Calendars and weights must have the same length")
    return WeightedCalendar(calendars=calendars, weights=weights)


def _r2p_calendar(calendar: NationalCalendar):
    """Convert a NationalCalendar to a protobuf Calendar."""
    from pydemetra._proto import toolkit_pb2

    p = toolkit_pb2.Calendar()
    for day in calendar.days:
        if isinstance(day, FixedDay):
            fd = toolkit_pb2.FixedDay(month=day.month, day=day.day, weight=day.weight)
            if day.validity:
                fd.validity.CopyFrom(_r2p_validity(day.validity))
            p.fixed_days.append(fd)
        elif isinstance(day, FixedWeekDay):
            fwd = toolkit_pb2.FixedWeekDay(
                month=day.month, position=day.week, weekday=day.dayofweek, weight=day.weight
            )
            if day.validity:
                fwd.validity.CopyFrom(_r2p_validity(day.validity))
            p.fixed_week_days.append(fwd)
        elif isinstance(day, EasterDay):
            erd = toolkit_pb2.EasterRelatedDay(
                offset=day.offset, julian=day.julian, weight=day.weight
            )
            if day.validity:
                erd.validity.CopyFrom(_r2p_validity(day.validity))
            p.easter_related_days.append(erd)
        elif isinstance(day, SpecialDay):
            from pydemetra._protobuf import enum_of

            ph = toolkit_pb2.PrespecifiedHoliday(
                event=enum_of(toolkit_pb2.CalendarEvent, day.event, "HOLIDAY"),
                offset=day.offset,
                weight=day.weight,
            )
            if day.validity:
                ph.validity.CopyFrom(_r2p_validity(day.validity))
            p.prespecified_holidays.append(ph)
        elif isinstance(day, SingleDay):
            from pydemetra._converters import r2p_date

            sd = toolkit_pb2.SingleDate(weight=day.weight)
            sd.date.CopyFrom(r2p_date(day.date))
            p.single_dates.append(sd)
    p.mean_correction = calendar.mean_correction
    return p


def _r2p_validity(v: dict):
    """Convert a validity dict to a protobuf ValidityPeriod."""
    from pydemetra._converters import r2p_date
    from pydemetra._proto import toolkit_pb2

    vp = toolkit_pb2.ValidityPeriod()
    if v.get("start"):
        vp.start.CopyFrom(r2p_date(v["start"]))
    else:
        vp.start.CopyFrom(r2p_date("0001-01-01"))
    if v.get("end"):
        vp.end.CopyFrom(r2p_date(v["end"]))
    else:
        vp.end.CopyFrom(r2p_date("9999-12-31"))
    return vp


def _p2jd_calendar(pcalendar) -> object:
    """Convert a protobuf Calendar to a Java Calendar object."""
    _ensure_jvm()
    import jpype

    buf = pcalendar.SerializeToString()
    Calendars = jpype.JClass("jdplus.toolkit.base.r.calendar.Calendars")
    return Calendars.calendarOf(buf)


def td(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    groups: list[int] | None = None,
    contrasts: bool = True,
) -> pd.DataFrame:
    """Generate trading day regressors without holidays.

    Args:
        frequency (int): Annual frequency (12, 4, etc.).
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        groups (list[int] | None): Day-of-week grouping (length 7, Mon=index 0).
        contrasts (bool): Whether to return contrasts against the 0-group.

    Returns:
        pd.DataFrame: DataFrame with trading day regressor columns.
    """
    _ensure_jvm()
    import jpype

    if groups is None:
        groups = [1, 2, 3, 4, 5, 6, 0]

    frequency, start, length = _ts_params(s, frequency, start, length)

    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    jm = Variables.td(jdom, np.array(groups, dtype=np.int32), contrasts)
    data = jd2r_matrix(jm)

    if data is not None and data.ndim == 2:
        offset = 0 if not contrasts else 1
        cols = [f"group_{i + offset}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=cols)
    return pd.DataFrame(data)


def calendar_td(
    calendar: NationalCalendar,
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    groups: list[int] | None = None,
    holiday: int = 7,
    contrasts: bool = True,
) -> pd.DataFrame:
    """Generate trading day regressors with pre-defined holidays.

    Args:
        calendar (NationalCalendar): The calendar containing holidays.
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        groups (list[int] | None): Day-of-week grouping.
        holiday (int): Day to aggregate holidays with (7=Sunday).
        contrasts (bool): Whether to return contrasts.

    Returns:
        pd.DataFrame: DataFrame with trading day regressor columns.
    """
    _ensure_jvm()
    import jpype

    if groups is None:
        groups = [1, 2, 3, 4, 5, 6, 0]

    frequency, start, length = _ts_params(s, frequency, start, length)

    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    pcal = _r2p_calendar(calendar)
    jcal = _p2jd_calendar(pcal)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    jm = Variables.htd(jcal, jdom, np.array(groups, dtype=np.int32), int(holiday), contrasts)
    data = jd2r_matrix(jm)

    if data is not None and data.ndim == 2:
        offset = 0 if not contrasts else 1
        cols = [f"group_{i + offset}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=cols)
    return pd.DataFrame(data)


def holidays(
    calendar: NationalCalendar,
    start: str,
    length: int,
    nonworking: list[int] | None = None,
    type: str = "Skip",
    single: bool = False,
) -> np.ndarray:
    """Generate daily holiday regressors from a calendar.

    Args:
        calendar (NationalCalendar): The calendar with holidays.
        start (str): Start date in ``"YYYY-MM-DD"`` format.
        length (int): Number of days.
        nonworking (list[int] | None): Non-working day indices (1=Monday, 7=Sunday).
        type (str): ``"Skip"``, ``"All"``, ``"NextWorkingDay"``, ``"PreviousWorkingDay"``.
        single (bool): If True, return a single column.

    Returns:
        np.ndarray: Matrix with holiday dummy variables.
    """
    _ensure_jvm()
    import jpype

    if nonworking is None:
        nonworking = [6, 7]

    pcal = _r2p_calendar(calendar)
    jcal = _p2jd_calendar(pcal)
    CalendarsR = jpype.JClass("jdplus.toolkit.base.r.calendar.Calendars")
    jm = CalendarsR.holidays(
        jcal,
        start,
        int(length),
        np.array(nonworking, dtype=np.int32),
        type,
        single,
    )
    result = jd2r_matrix(jm)
    assert result is not None
    return result


def long_term_mean(
    calendar: NationalCalendar,
    frequency: int,
    groups: list[int] | None = None,
    holiday: int = 7,
) -> np.ndarray:
    """Compute long-term means for calendar regressors.

    Args:
        calendar (NationalCalendar): The calendar.
        frequency (int): Annual frequency.
        groups (list[int] | None): Day-of-week grouping.
        holiday (int): Day to aggregate holidays with.

    Returns:
        np.ndarray: Matrix of long-term means per group and period.
    """
    _ensure_jvm()
    import jpype

    if groups is None:
        groups = [1, 2, 3, 4, 5, 6, 0]

    pcal = _r2p_calendar(calendar)
    jcal = _p2jd_calendar(pcal)
    CalendarsR = jpype.JClass("jdplus.toolkit.base.r.calendar.Calendars")
    jm = CalendarsR.longTermMean(
        jcal,
        int(frequency),
        np.array(groups, dtype=np.int32),
        int(holiday),
    )
    result = jd2r_matrix(jm)
    assert result is not None
    return result


def easter_dates(year0: int, year1: int, julian: bool = False) -> list[str]:
    """Return Easter Sunday dates for a range of years.

    Args:
        year0 (int): Start year (inclusive).
        year1 (int): End year (inclusive).
        julian (bool): Use Julian calendar.

    Returns:
        list[str]: List of date strings.
    """
    _ensure_jvm()
    import jpype

    CalendarsR = jpype.JClass("jdplus.toolkit.base.r.calendar.Calendars")
    dates = CalendarsR.easter(int(year0), int(year1), julian)
    return [str(d) for d in dates]


def stock_td(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    w: int = 31,
) -> pd.DataFrame:
    """Generate trading day regressors for stock series.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        w (int): Day of month for stock reporting (31=last day).

    Returns:
        pd.DataFrame: DataFrame with 6 trading day columns (Monday to Saturday).
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)

    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    jm = Variables.stockTradingDays(jdom, int(w))
    data = jd2r_matrix(jm)
    cols = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    return pd.DataFrame(data, columns=cols)
