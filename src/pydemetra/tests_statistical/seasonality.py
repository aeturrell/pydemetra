from __future__ import annotations

import numpy as np
import pandas as pd

from pydemetra._java import _ensure_jvm
from pydemetra._models import StatisticalTest
from pydemetra._results import _jd2r_test


def _seas_class():
    import jpype

    return jpype.JClass("jdplus.sa.base.r.SeasonalityTests")


def _to_1d(data: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError(
                f"Expected a Series or single-column DataFrame, got {data.shape[1]} columns. "
                "Pass a single column, e.g. df['column_name']."
            )
        data = data.iloc[:, 0]
    return np.asarray(data, dtype=np.float64).ravel()


def _get_period(data: np.ndarray | pd.Series, period: int | None) -> int:
    if period is not None and period > 0:
        return period
    if isinstance(data, pd.Series) and isinstance(data.index, pd.PeriodIndex):
        freq = data.index.freqstr
        if "M" in freq:
            return 12
        if "Q" in freq:
            return 4
    raise ValueError("period must be specified for non-ts data")


def seasonality_qs(
    data: np.ndarray | pd.Series,
    period: int | None = None,
    nyears: int = 0,
    type: int = 1,
) -> StatisticalTest:
    """QS (seasonal Ljung-Box) test.

    Args:
        data (np.ndarray | pd.Series): Input data.
        period (int | None): Tested periodicity.
        nyears (int): Number of years at end of series (0=full sample).
        type (int): 1 for positive, -1 for negative, 0 for all auto-correlations.

    Returns:
        StatisticalTest: StatisticalTest result.
    """
    _ensure_jvm()
    period = _get_period(data, period)
    vals = _to_1d(data)
    jtest = _seas_class().qsTest(vals, int(period), int(nyears), int(type))
    return _jd2r_test(jtest)


def seasonality_modified_qs(
    data: np.ndarray | pd.Series,
    period: int | None = None,
    nyears: int = 0,
) -> float:
    """Modified QS seasonality test (Maravall).

    Args:
        data (np.ndarray | pd.Series): Input data.
        period (int | None): Tested periodicity.
        nyears (int): Number of years at end of series.

    Returns:
        float: The test statistic value.
    """
    _ensure_jvm()
    period = _get_period(data, period)
    vals = _to_1d(data)
    return float(_seas_class().modifiedQsTest(vals, int(period), int(nyears)))


def seasonality_kruskal_wallis(
    data: np.ndarray | pd.Series,
    period: int | None = None,
    nyears: int = 0,
) -> StatisticalTest:
    """Kruskal-Wallis seasonality test."""
    _ensure_jvm()
    period = _get_period(data, period)
    vals = _to_1d(data)
    jtest = _seas_class().kruskalWallisTest(vals, int(period), int(nyears))
    return _jd2r_test(jtest)


def seasonality_periodogram(
    data: np.ndarray | pd.Series,
    period: int | None = None,
    nyears: int = 0,
) -> StatisticalTest:
    """Periodogram seasonality test."""
    _ensure_jvm()
    period = _get_period(data, period)
    vals = _to_1d(data)
    jtest = _seas_class().periodogramTest(vals, int(period), int(nyears))
    return _jd2r_test(jtest)


def seasonality_friedman(
    data: np.ndarray | pd.Series,
    period: int | None = None,
    nyears: int = 0,
) -> StatisticalTest:
    """Friedman seasonality test."""
    _ensure_jvm()
    period = _get_period(data, period)
    vals = _to_1d(data)
    jtest = _seas_class().friedmanTest(vals, int(period), int(nyears))
    return _jd2r_test(jtest)


def seasonality_f(
    data: np.ndarray | pd.Series,
    period: int | None = None,
    model: str = "AR",
    nyears: int = 0,
) -> StatisticalTest:
    """F-test on seasonal dummies.

    Args:
        data (np.ndarray | pd.Series): Input data.
        period (int | None): Tested periodicity.
        model (str): ``"AR"``, ``"D1"``, or ``"WN"``.
        nyears (int): Number of years at end of series.

    Returns:
        StatisticalTest: StatisticalTest result.
    """
    _ensure_jvm()
    period = _get_period(data, period)
    vals = _to_1d(data)
    jtest = _seas_class().fTest(vals, int(period), model, int(nyears))
    return _jd2r_test(jtest)


def seasonality_combined(
    data: np.ndarray | pd.Series,
    period: int | None = None,
    firstperiod: int = 1,
    mul: bool = True,
) -> dict:
    """Combined seasonality test (X12-style).

    Args:
        data (np.ndarray | pd.Series): Input data.
        period (int | None): Tested periodicity.
        firstperiod (int): Position of the first observation in a cycle.
        mul (bool): Multiplicative decomposition.

    Returns:
        dict: Dict with ``"seasonality"``, ``"kruskalwallis"``, ``"stable"``, ``"evolutive"`` keys.
    """
    _ensure_jvm()

    from pydemetra._converters import p2r_test
    from pydemetra._proto import sa_pb2
    from pydemetra._protobuf import enum_extract

    period = _get_period(data, period)
    vals = _to_1d(data)

    SeasonalityTests = _seas_class()
    jctest = SeasonalityTests.combinedTest(vals, int(period), int(firstperiod - 1), mul)
    buf = bytes(SeasonalityTests.toBuffer(jctest))

    msg = sa_pb2.CombinedSeasonalityTest()
    msg.ParseFromString(buf)

    from scipy.stats import f as f_dist

    def _p2r_anova(anova):
        ssm, dfm, ssr, dfr = anova.SSM, anova.dfm, anova.SSR, anova.dfr
        if dfm > 0 and dfr > 0 and ssr > 0:
            val = (ssm / dfm) * (dfr / ssr)
            pval = 1.0 - float(f_dist.cdf(val, dfm, dfr))
        else:
            val, pval = 0.0, 1.0
        return {
            "SSM": ssm,
            "dfM": dfm,
            "SSR": ssr,
            "dfR": dfr,
            "test": StatisticalTest(val, pval, f"F({dfm},{dfr})"),
        }

    return {
        "seasonality": enum_extract(sa_pb2.IdentifiableSeasonality, msg.seasonality),
        "kruskalwallis": p2r_test(msg.kruskal_wallis),
        "stable": _p2r_anova(msg.stable_seasonality),
        "evolutive": _p2r_anova(msg.evolutive_seasonality),
    }


def seasonality_canova_hansen_trigs(
    data: np.ndarray | pd.Series,
    periods: np.ndarray | list[float],
    lag1: bool = True,
    kernel: str = "Bartlett",
    order: int = -1,
    original: bool = False,
) -> np.ndarray:
    """Canova-Hansen test using trigonometric variables.

    Args:
        data (np.ndarray | pd.Series): Input data.
        periods (np.ndarray | list[float]): Periodicities to test.
        lag1 (bool): Include lagged variable.
        kernel (str): Kernel for Newey-West covariance.
        order (int): Truncation parameter (-1 for automatic).
        original (bool): Use original algorithm.

    Returns:
        np.ndarray: Array of test statistics.
    """
    _ensure_jvm()
    vals = _to_1d(data)
    periods_arr = np.asarray(periods, dtype=np.float64)
    result = _seas_class().canovaHansenTrigs(
        vals,
        periods_arr,
        lag1,
        kernel,
        int(order),
        original,
    )
    return np.array(result)


def seasonality_canova_hansen(
    data: np.ndarray | pd.Series,
    period: int,
    type: str = "Contrast",
    lag1: bool = True,
    kernel: str = "Bartlett",
    order: int = -1,
    start: int = 1,
) -> dict:
    """Canova-Hansen seasonality test.

    Args:
        data (np.ndarray | pd.Series): Input data.
        period (int): Periodicity.
        type (str): ``"Contrast"``, ``"Dummy"``, or ``"Trigonometric"``.
        lag1 (bool): Include lagged variable.
        kernel (str): Kernel for Newey-West covariance.
        order (int): Truncation parameter (-1 for automatic).
        start (int): Position of first observation.

    Returns:
        dict: Dict with ``"seasonality"``, ``"joint"``, and ``"details"`` keys.
    """
    _ensure_jvm()
    vals = _to_1d(data)
    q = np.array(
        _seas_class().canovaHansen(
            vals,
            int(period),
            type,
            lag1,
            kernel,
            int(order),
            int(start - 1),
        )
    )
    last = len(q)
    return {
        "seasonality": {"value": q[last - 2], "pvalue": q[last - 1]},
        "joint": q[last - 3],
        "details": q[: last - 3],
    }
