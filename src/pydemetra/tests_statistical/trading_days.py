from __future__ import annotations

import numpy as np
import pandas as pd

from pydemetra._converters import r2jd_tsdata
from pydemetra._java import _ensure_jvm
from pydemetra._models import StatisticalTest
from pydemetra._results import _jd2r_test


def td_f(
    s: pd.Series,
    model: str = "D1",
    nyears: int = 0,
) -> StatisticalTest:
    """Residual trading days F-test.

    Args:
        s (pd.Series): Input time series.
        model (str): ``"D1"``, ``"DY"``, ``"DYD1"``, ``"WN"``, ``"AIRLINE"``,
            ``"R011"``, ``"R100"``.
        nyears (int): Number of years at end of series (0=full sample).

    Returns:
        StatisticalTest: StatisticalTest result.
    """
    _ensure_jvm()
    import jpype

    jts = r2jd_tsdata(s)
    TDTests = jpype.JClass("jdplus.toolkit.base.r.modelling.TradingDaysTests")
    jtest = TDTests.fTest(jts, model, int(nyears))
    result = _jd2r_test(jtest)
    assert result is not None
    return result


def td_canova_hansen(
    s: pd.Series,
    differencing: list[int],
    kernel: str = "Bartlett",
    order: int = -1,
) -> dict:
    """Canova-Hansen test for stable trading days.

    Args:
        s (pd.Series): Input time series.
        differencing (list[int]): Differencing lags.
        kernel (str): Kernel for Newey-West covariance.
        order (int): Truncation parameter (-1 for automatic).

    Returns:
        dict: Dict with ``"td"``, ``"joint"``, and ``"details"`` keys.
    """
    _ensure_jvm()
    import jpype

    jts = r2jd_tsdata(s)
    TDTests = jpype.JClass("jdplus.toolkit.base.r.modelling.TradingDaysTests")
    q = np.array(
        TDTests.canovaHansen(
            jts,
            np.array(differencing, dtype=np.int32),
            kernel,
            int(order),
        )
    )
    last = len(q)
    return {
        "td": {"value": q[last - 2], "pvalue": q[last - 1]},
        "joint": q[last - 3],
        "details": q[: last - 3],
    }


def td_time_varying(
    s: pd.Series,
    groups: list[int] | None = None,
    contrasts: bool = False,
) -> StatisticalTest:
    """Likelihood ratio test on time-varying trading days.

    Args:
        s (pd.Series): Input time series.
        groups (list[int] | None): Day-of-week grouping (length 7).
        contrasts (bool): Use contrasts for covariance matrix.

    Returns:
        StatisticalTest: StatisticalTest result.
    """
    _ensure_jvm()
    import jpype

    if groups is None:
        groups = [1, 2, 3, 4, 5, 6, 0]

    jts = r2jd_tsdata(s)
    TDTests = jpype.JClass("jdplus.toolkit.base.r.modelling.TradingDaysTests")
    jtest = TDTests.timeVaryingTradingDaysTest(
        jts,
        np.array(groups, dtype=np.int32),
        contrasts,
    )
    result = _jd2r_test(jtest)
    assert result is not None
    return result
