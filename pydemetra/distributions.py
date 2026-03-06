from __future__ import annotations

import numpy as np

from pydemetra._java import _ensure_jvm


def _dist_class():
    import jpype

    return jpype.JClass("jdplus.toolkit.base.r.stats.Distributions")


def random_t(df: float, n: int) -> np.ndarray:
    """Generate random variates from a Student t-distribution.

    Args:
        df: Degrees of freedom.
        n: Number of observations.

    Returns:
        Array of random values.
    """
    _ensure_jvm()
    return np.array(_dist_class().randomsT(float(df), int(n)))


def density_t(df: float, x: np.ndarray | list[float]) -> np.ndarray:
    """Compute the PDF of a Student t-distribution.

    Args:
        df: Degrees of freedom.
        x: Quantiles.

    Returns:
        Array of density values.
    """
    _ensure_jvm()
    return np.array(_dist_class().densityT(float(df), np.asarray(x, dtype=np.float64)))


def cdf_t(df: float, x: np.ndarray | list[float]) -> np.ndarray:
    """Compute the CDF of a Student t-distribution.

    Args:
        df: Degrees of freedom.
        x: Quantiles.

    Returns:
        Array of cumulative probabilities.
    """
    _ensure_jvm()
    return np.array(_dist_class().cdfT(float(df), np.asarray(x, dtype=np.float64)))


def random_chi2(df: float, n: int) -> np.ndarray:
    """Generate random variates from a chi-squared distribution."""
    _ensure_jvm()
    return np.array(_dist_class().randomsChi2(float(df), int(n)))


def density_chi2(df: float, x: np.ndarray | list[float]) -> np.ndarray:
    """Compute the PDF of a chi-squared distribution."""
    _ensure_jvm()
    return np.array(_dist_class().densityChi2(float(df), np.asarray(x, dtype=np.float64)))


def cdf_chi2(df: float, x: np.ndarray | list[float]) -> np.ndarray:
    """Compute the CDF of a chi-squared distribution."""
    _ensure_jvm()
    return np.array(_dist_class().cdfChi2(float(df), np.asarray(x, dtype=np.float64)))


def random_gamma(shape: float, scale: float, n: int) -> np.ndarray:
    """Generate random variates from a Gamma distribution."""
    _ensure_jvm()
    return np.array(_dist_class().randomsGamma(float(shape), float(scale), int(n)))


def density_gamma(shape: float, scale: float, x: np.ndarray | list[float]) -> np.ndarray:
    """Compute the PDF of a Gamma distribution."""
    _ensure_jvm()
    return np.array(
        _dist_class().densityGamma(float(shape), float(scale), np.asarray(x, dtype=np.float64))
    )


def cdf_gamma(shape: float, scale: float, x: np.ndarray | list[float]) -> np.ndarray:
    """Compute the CDF of a Gamma distribution."""
    _ensure_jvm()
    return np.array(
        _dist_class().cdfGamma(float(shape), float(scale), np.asarray(x, dtype=np.float64))
    )


def random_inverse_gamma(shape: float, scale: float, n: int) -> np.ndarray:
    """Generate random variates from an inverse-Gamma distribution."""
    _ensure_jvm()
    return np.array(_dist_class().randomsInverseGamma(float(shape), float(scale), int(n)))


def density_inverse_gamma(shape: float, scale: float, x: np.ndarray | list[float]) -> np.ndarray:
    """Compute the PDF of an inverse-Gamma distribution."""
    _ensure_jvm()
    return np.array(
        _dist_class().densityInverseGamma(
            float(shape), float(scale), np.asarray(x, dtype=np.float64)
        )
    )


def cdf_inverse_gamma(shape: float, scale: float, x: np.ndarray | list[float]) -> np.ndarray:
    """Compute the CDF of an inverse-Gamma distribution."""
    _ensure_jvm()
    return np.array(
        _dist_class().cdfInverseGamma(float(shape), float(scale), np.asarray(x, dtype=np.float64))
    )


def random_inverse_gaussian(shape: float, scale: float, n: int) -> np.ndarray:
    """Generate random variates from an inverse-Gaussian distribution."""
    _ensure_jvm()
    return np.array(_dist_class().randomsInverseGaussian(float(shape), float(scale), int(n)))


def density_inverse_gaussian(
    shape: float, scale: float, x: np.ndarray | list[float]
) -> np.ndarray:
    """Compute the PDF of an inverse-Gaussian distribution."""
    _ensure_jvm()
    return np.array(
        _dist_class().densityInverseGaussian(
            float(shape), float(scale), np.asarray(x, dtype=np.float64)
        )
    )


def cdf_inverse_gaussian(shape: float, scale: float, x: np.ndarray | list[float]) -> np.ndarray:
    """Compute the CDF of an inverse-Gaussian distribution."""
    _ensure_jvm()
    return np.array(
        _dist_class().cdfInverseGaussian(
            float(shape), float(scale), np.asarray(x, dtype=np.float64)
        )
    )
