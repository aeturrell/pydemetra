from __future__ import annotations

import numpy as np

from pydemetra._java import _ensure_jvm
from pydemetra._models import StatisticalTest
from pydemetra._results import _jd2r_test


def _tests_class():
    import jpype

    return jpype.JClass("jdplus.toolkit.base.r.stats.Tests")


def ljungbox(
    data: np.ndarray,
    k: int = 1,
    lag: int = 1,
    nhp: int = 0,
    sign: int = 0,
    mean: bool = True,
) -> StatisticalTest:
    """Ljung-Box test for independence.

    Args:
        data (np.ndarray): Input data.
        k (int): Number of auto-correlations.
        lag (int): Lag between auto-correlations.
        nhp (int): Number of hyper-parameters (degree of freedom correction).
        sign (int): 1 for positive, -1 for negative, 0 for all auto-correlations.
        mean (bool): If True, compute mean-corrected auto-correlations.

    Returns:
        StatisticalTest: StatisticalTest result.
    """
    _ensure_jvm()
    jtest = _tests_class().ljungBox(
        np.asarray(data, dtype=np.float64),
        int(k),
        int(lag),
        int(nhp),
        int(sign),
        mean,
    )
    return _jd2r_test(jtest)


def bowman_shenton(data: np.ndarray) -> StatisticalTest:
    """Bowman-Shenton normality test."""
    _ensure_jvm()
    jtest = _tests_class().bowmanShenton(np.asarray(data, dtype=np.float64))
    return _jd2r_test(jtest)


def doornik_hansen(data: np.ndarray) -> StatisticalTest:
    """Doornik-Hansen normality test."""
    _ensure_jvm()
    jtest = _tests_class().doornikHansen(np.asarray(data, dtype=np.float64))
    return _jd2r_test(jtest)


def jarque_bera(data: np.ndarray, k: int = 0, sample: bool = True) -> StatisticalTest:
    """Jarque-Bera normality test.

    Args:
        data (np.ndarray): Input data.
        k (int): Degrees of freedom to subtract for residuals.
        sample (bool): Use unbiased empirical moments.

    Returns:
        StatisticalTest: StatisticalTest result.
    """
    _ensure_jvm()
    jtest = _tests_class().jarqueBera(np.asarray(data, dtype=np.float64), int(k), sample)
    return _jd2r_test(jtest)


def skewness(data: np.ndarray) -> StatisticalTest:
    """Skewness test."""
    _ensure_jvm()
    jtest = _tests_class().skewness(np.asarray(data, dtype=np.float64))
    return _jd2r_test(jtest)


def kurtosis(data: np.ndarray) -> StatisticalTest:
    """Kurtosis test."""
    _ensure_jvm()
    jtest = _tests_class().kurtosis(np.asarray(data, dtype=np.float64))
    return _jd2r_test(jtest)


def test_of_runs(data: np.ndarray, mean: bool = True, number: bool = True) -> StatisticalTest:
    """Runs test around mean or median.

    Args:
        data (np.ndarray): Input data.
        mean (bool): If True, runs around the mean; otherwise around the median.
        number (bool): If True, test number of runs; otherwise test lengths.

    Returns:
        StatisticalTest: StatisticalTest result.
    """
    _ensure_jvm()
    jtest = _tests_class().testOfRuns(np.asarray(data, dtype=np.float64), mean, number)
    return _jd2r_test(jtest)


def test_of_up_down_runs(data: np.ndarray, number: bool = True) -> StatisticalTest:
    """Up-and-down runs test.

    Args:
        data (np.ndarray): Input data.
        number (bool): If True, test number of runs; otherwise test lengths.

    Returns:
        StatisticalTest: StatisticalTest result.
    """
    _ensure_jvm()
    jtest = _tests_class().testOfUpDownRuns(np.asarray(data, dtype=np.float64), number)
    return _jd2r_test(jtest)


def autocorrelations(data: np.ndarray, mean: bool = True, n: int = 15) -> np.ndarray:
    """Compute autocorrelation function.

    Args:
        data (np.ndarray): Input data.
        mean (bool): Mean correction.
        n (int): Maximum lag.

    Returns:
        np.ndarray: Array of autocorrelation values.
    """
    _ensure_jvm()
    return np.array(
        _tests_class().autocorrelations(np.asarray(data, dtype=np.float64), mean, int(n))
    )


def autocorrelations_partial(data: np.ndarray, mean: bool = True, n: int = 15) -> np.ndarray:
    """Compute partial autocorrelation function.

    Args:
        data (np.ndarray): Input data.
        mean (bool): Mean correction.
        n (int): Maximum lag.

    Returns:
        np.ndarray: Array of partial autocorrelation values.
    """
    _ensure_jvm()
    return np.array(
        _tests_class().partialAutocorrelations(np.asarray(data, dtype=np.float64), mean, int(n))
    )


def autocorrelations_inverse(data: np.ndarray, nar: int = 30, n: int = 15) -> np.ndarray:
    """Compute inverse autocorrelation function.

    Args:
        data (np.ndarray): Input data.
        nar (int): Number of AR lags for computation.
        n (int): Maximum lag.

    Returns:
        np.ndarray: Array of inverse autocorrelation values.
    """
    _ensure_jvm()
    return np.array(
        _tests_class().inverseAutocorrelations(
            np.asarray(data, dtype=np.float64), int(nar), int(n)
        )
    )


def mad(data: np.ndarray, centile: float = 50.0, median_corrected: bool = True) -> float:
    """Compute robust median absolute deviation.

    Args:
        data (np.ndarray): Input data.
        centile (float): Percentage of data used.
        median_corrected (bool): If True, correct for median.

    Returns:
        float: The MAD value.
    """
    _ensure_jvm()
    return float(
        _tests_class().mad(np.asarray(data, dtype=np.float64), float(centile), median_corrected)
    )
