from __future__ import annotations

import numpy as np
import pandas as pd

from pydemetra._converters import jd2r_matrix, r2jd_tsdomain
from pydemetra._java import _ensure_jvm
from pydemetra.timeseries import _ts_params


def easter_variable(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    duration: int = 6,
    endpos: int = -1,
    correction: str = "Simple",
) -> np.ndarray:
    """Generate an Easter effect regressor.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        duration (int): Duration of the Easter effect in days (1-20).
        endpos (int): End position relative to Easter (-1=before Sunday, 0=Sunday, 1=Monday).
        correction (str): ``"Simple"``, ``"PreComputed"``, ``"Theoretical"``, or ``"None"``.

    Returns:
        np.ndarray: Array with Easter regressor values.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    data = np.array(Variables.easter(jdom, int(duration), int(endpos), correction))
    return data


def julianeaster_variable(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    duration: int = 6,
) -> np.ndarray:
    """Generate a Julian Easter effect regressor.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        duration (int): Duration of the Easter effect in days.

    Returns:
        np.ndarray: Array with Julian Easter regressor values.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    data = np.array(Variables.julianEaster(jdom, int(duration)))
    return data


def lp_variable(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    type: str = "LeapYear",
) -> np.ndarray:
    """Generate a leap year or length-of-period regressor.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        type (str): ``"LeapYear"`` or ``"LengthOfPeriod"``.

    Returns:
        np.ndarray: Array with regressor values.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    lp = type == "LeapYear"
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    data = np.array(Variables.leapYear(jdom, lp))
    return data


def ao_variable(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    pos: int | None = None,
    date: str | None = None,
) -> np.ndarray:
    """Generate an additive outlier regressor.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        pos (int | None): 1-based position in the series.
        date (str | None): Date in ``"YYYY-MM-DD"`` format.

    Returns:
        np.ndarray: Array with outlier regressor values.

    Raises:
        ValueError: If neither *date* nor *pos* is provided.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    if date is not None:
        data = np.array(Variables.ao(jdom, str(date)))
    else:
        if pos is None:
            raise ValueError("Either date or pos must be provided")
        data = np.array(Variables.ao(jdom, int(pos - 1)))
    return data


def ls_variable(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    pos: int | None = None,
    date: str | None = None,
    zeroended: bool = True,
) -> np.ndarray:
    """Generate a level shift regressor.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        pos (int | None): 1-based position in the series.
        date (str | None): Date in ``"YYYY-MM-DD"`` format.
        zeroended (bool): If True, regressor ends at 0.

    Returns:
        np.ndarray: Array with level shift regressor values.

    Raises:
        ValueError: If neither *date* nor *pos* is provided.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    if date is not None:
        data = np.array(Variables.ls(jdom, str(date), zeroended))
    else:
        if pos is None:
            raise ValueError("Either date or pos must be provided")
        data = np.array(Variables.ls(jdom, int(pos - 1), zeroended))
    return data


def tc_variable(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    pos: int | None = None,
    date: str | None = None,
    rate: float = 0.7,
) -> np.ndarray:
    """Generate a transitory change regressor.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        pos (int | None): 1-based position in the series.
        date (str | None): Date in ``"YYYY-MM-DD"`` format.
        rate (float): Decay rate.

    Returns:
        np.ndarray: Array with transitory change regressor values.

    Raises:
        ValueError: If neither *date* nor *pos* is provided.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    if date is not None:
        data = np.array(Variables.tc(jdom, str(date), float(rate)))
    else:
        if pos is None:
            raise ValueError("Either date or pos must be provided")
        data = np.array(Variables.tc(jdom, int(pos - 1), float(rate)))
    return data


def so_variable(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    pos: int | None = None,
    date: str | None = None,
    zeroended: bool = True,
) -> np.ndarray:
    """Generate a seasonal outlier regressor.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        pos (int | None): 1-based position in the series.
        date (str | None): Date in ``"YYYY-MM-DD"`` format.
        zeroended (bool): If True, regressor ends at 0.

    Returns:
        np.ndarray: Array with seasonal outlier regressor values.

    Raises:
        ValueError: If neither *date* nor *pos* is provided.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    if date is not None:
        data = np.array(Variables.so(jdom, str(date), zeroended))
    else:
        if pos is None:
            raise ValueError("Either date or pos must be provided")
        data = np.array(Variables.so(jdom, int(pos - 1), zeroended))
    return data


def ramp_variable(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    range: tuple | list | None = None,
) -> np.ndarray:
    """Generate a ramp regressor.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        range (tuple | list | None): Length-2 sequence of dates (``"YYYY-MM-DD"``)
            or 1-based positions.

    Returns:
        np.ndarray: Array with ramp regressor values.

    Raises:
        ValueError: If *range* is not provided.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")

    if range is None:
        raise ValueError("range must be provided")
    if isinstance(range[0], str):
        data = np.array(Variables.ramp(jdom, str(range[0]), str(range[1])))
    else:
        data = np.array(Variables.ramp(jdom, int(range[0] - 1), int(range[1] - 1)))
    return data


def intervention_variable(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    starts: list[str] | list[int] | None = None,
    ends: list[str] | list[int] | None = None,
    delta: float = 0.0,
    seasonaldelta: float = 0.0,
) -> np.ndarray:
    """Generate an intervention variable.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        starts (list[str] | list[int] | None): Start dates or positions for intervention sequences.
        ends (list[str] | list[int] | None): End dates or positions for intervention sequences.
        delta (float): Regular differencing order.
        seasonaldelta (float): Seasonal differencing order.

    Returns:
        np.ndarray: Array with intervention variable values.

    Raises:
        ValueError: If *starts* and *ends* have different lengths.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    if starts is None or ends is None:
        raise ValueError("starts and ends must both be provided")
    if len(starts) != len(ends):
        raise ValueError("starts and ends must have the same length")

    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")

    if isinstance(starts[0], str):
        data = np.array(
            Variables.interventionVariable(
                jdom,
                float(delta),
                float(seasonaldelta),
                [str(s) for s in starts],
                [str(e) for e in ends],
            )
        )
    else:
        data = np.array(
            Variables.interventionVariable(
                jdom,
                float(delta),
                float(seasonaldelta),
                np.array([int(s) - 1 for s in starts], dtype=np.int32),
                np.array([int(e) - 1 for e in ends], dtype=np.int32),
            )
        )
    return data


def periodic_dummies(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
) -> np.ndarray:
    """Generate periodic dummy variables.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.

    Returns:
        np.ndarray: Matrix with one column per period.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    jm = Variables.periodicDummies(jdom)
    result = jd2r_matrix(jm)
    assert result is not None
    return result


def periodic_contrasts(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
) -> np.ndarray:
    """Generate periodic contrast variables.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.

    Returns:
        np.ndarray: Matrix with periodic contrasts.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")
    jm = Variables.periodicContrasts(jdom)
    result = jd2r_matrix(jm)
    assert result is not None
    return result


def trigonometric_variables(
    frequency: int,
    start: tuple[int, int],
    length: int,
    s: pd.Series | None = None,
    seasonal_frequency: list[int] | int | None = None,
) -> np.ndarray | None:
    """Generate trigonometric variables at seasonal frequencies.

    Args:
        frequency (int): Annual frequency.
        start (tuple[int, int]): (year, period) tuple.
        length (int): Number of periods.
        s (pd.Series | None): Optional time series to derive parameters from.
        seasonal_frequency (list[int] | int | None): Specific seasonal frequencies,
            or None for all harmonics.

    Returns:
        np.ndarray | None: Matrix with cos/sin columns, or None.
    """
    _ensure_jvm()
    import jpype

    frequency, start, length = _ts_params(s, frequency, start, length)
    jdom = r2jd_tsdomain(frequency, start[0], start[1], length)
    Variables = jpype.JClass("jdplus.toolkit.base.r.modelling.Variables")

    if seasonal_frequency is not None:
        if isinstance(seasonal_frequency, int):
            seasonal_frequency = [seasonal_frequency]
        sf = np.array(seasonal_frequency, dtype=np.int32)
    else:
        sf = None

    jm = Variables.trigonometricVariables(jdom, sf)
    data = jd2r_matrix(jm)
    if data is not None and data.shape[1] % 2 == 1:
        data = np.column_stack([data, np.zeros(data.shape[0])])
    return data
