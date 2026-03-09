from __future__ import annotations

import numpy as np
import pandas as pd

from pydemetra._java import _ensure_jvm


def do_stationary(data: np.ndarray | pd.Series, period: int | None = None) -> dict:
    """Automatic stationary transformation using Tramo-style differencing.

    Args:
        data (np.ndarray | pd.Series): Series to be differenced.
        period (int | None): Period of the series.

    Returns:
        dict: Dict with ``"ddata"`` (differenced data), ``"mean"`` (mean correction flag),
            and ``"differences"`` (lag/order matrix).
    """
    _ensure_jvm()
    import jpype

    from pydemetra._proto import modelling_pb2

    if period is None and isinstance(data, pd.Series) and isinstance(data.index, pd.PeriodIndex):
        freq = data.index.freqstr
        period = 12 if "M" in freq else (4 if "Q" in freq else 1)

    vals = np.asarray(data, dtype=np.float64)
    Differencing = jpype.JClass("jdplus.toolkit.base.r.modelling.Differencing")
    jst = Differencing.doStationary(vals, int(period))
    buf = bytes(Differencing.toBuffer(jst))

    msg = modelling_pb2.StationaryTransformation()
    msg.ParseFromString(buf)

    if msg.differences:
        diffs = np.array([[d.lag, d.order] for d in msg.differences]).T
    else:
        diffs = np.empty((2, 0))

    result = {
        "ddata": np.array(list(msg.stationary_series)),
        "mean": msg.mean_correction,
        "differences": diffs,
    }
    return result


def differencing_fast(
    data: np.ndarray | pd.Series,
    period: int | None = None,
    mad: bool = True,
    centile: float = 90.0,
    k: float = 1.2,
) -> dict:
    """Fast automatic differencing based on variance decrease.

    Args:
        data (np.ndarray | pd.Series): Series to be differenced.
        period (int | None): Period of the series.
        mad (bool): Use MAD for variance computation.
        centile (float): Percentage of data used for variance.
        k (float): Tolerance for variance decrease.

    Returns:
        dict: Dict with ``"ddata"``, ``"mean"``, and ``"differences"`` keys.
    """
    _ensure_jvm()
    import jpype

    from pydemetra._proto import modelling_pb2

    if period is None and isinstance(data, pd.Series) and isinstance(data.index, pd.PeriodIndex):
        freq = data.index.freqstr
        period = 12 if "M" in freq else (4 if "Q" in freq else 1)

    vals = np.asarray(data, dtype=np.float64)
    Differencing = jpype.JClass("jdplus.toolkit.base.r.modelling.Differencing")
    jst = Differencing.fastDifferencing(vals, int(period), mad, float(centile), float(k))
    buf = bytes(Differencing.toBuffer(jst))

    msg = modelling_pb2.StationaryTransformation()
    msg.ParseFromString(buf)

    if msg.differences:
        diffs = np.array([[d.lag, d.order] for d in msg.differences]).T
    else:
        diffs = np.empty((2, 0))

    return {
        "ddata": np.array(list(msg.stationary_series)),
        "mean": msg.mean_correction,
        "differences": diffs,
    }


def differences(
    data: np.ndarray | pd.Series,
    lags: list[int] | int = 1,
    mean: bool = True,
) -> np.ndarray:
    """Difference a series at specified lags.

    Args:
        data (np.ndarray | pd.Series): Series to be differenced.
        lags (list[int] | int): Single lag or list of lags.
        mean (bool): Apply mean correction after differencing.

    Returns:
        np.ndarray: The differenced series.
    """
    _ensure_jvm()
    import jpype

    if isinstance(lags, int):
        lags = [lags]

    vals = np.asarray(data, dtype=np.float64)
    Differencing = jpype.JClass("jdplus.toolkit.base.r.modelling.Differencing")
    result = np.array(
        Differencing.differences(
            vals,
            np.array(lags, dtype=np.int32),
            mean,
        )
    )
    return result


def rangemean_tstat(
    data: np.ndarray | pd.Series,
    period: int = 0,
    groupsize: int = 0,
    trim: int = 0,
) -> float:
    """Range-mean regression T-statistic for log transformation decision.

    Args:
        data (np.ndarray | pd.Series): Data to test.
        period (int): Periodicity.
        groupsize (int): Number of observations per group (0=automatic).
        trim (int): Number of trimmed observations.

    Returns:
        float: T-statistic of the slope.
    """
    _ensure_jvm()
    import jpype

    if period == 0 and isinstance(data, pd.Series) and isinstance(data.index, pd.PeriodIndex):
        freq = data.index.freqstr
        period = 12 if "M" in freq else (4 if "Q" in freq else 1)

    vals = np.asarray(data, dtype=np.float64)
    AutoModelling = jpype.JClass("jdplus.toolkit.base.r.modelling.AutoModelling")
    return float(AutoModelling.rangeMean(vals, int(period), int(groupsize), int(trim)))
