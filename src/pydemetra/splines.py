from __future__ import annotations

import numpy as np

from pydemetra._converters import jd2r_matrix
from pydemetra._java import _ensure_jvm


def periodic_bsplines(
    order: int = 4,
    period: float = 1.0,
    knots: list[float] | np.ndarray | tuple[()] = (),
    pos: list[float] | np.ndarray | tuple[()] = (),
) -> np.ndarray:
    """Compute periodic B-splines.

    Args:
        order (int): Order of the splines (4 for cubic).
        period (float): Period of the splines.
        knots (list[float] | np.ndarray | tuple[()]): Knot positions in [0, period).
        pos (list[float] | np.ndarray | tuple[()]): Evaluation positions in [0, period).

    Returns:
        np.ndarray: Matrix of shape (len(pos), len(knots)).
    """
    _ensure_jvm()
    import jpype

    BSplines = jpype.JClass("jdplus.toolkit.base.r.math.BSplines")
    jm = BSplines.periodic(
        int(order),
        float(period),
        np.asarray(knots, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
    )
    result = jd2r_matrix(jm)
    assert result is not None
    return result


def bsplines(
    order: int = 4,
    knots: list[float] | np.ndarray | tuple[()] = (),
    pos: list[float] | np.ndarray | tuple[()] = (),
) -> np.ndarray:
    """Compute B-splines.

    Args:
        order (int): Order of the splines (4 for cubic).
        knots (list[float] | np.ndarray | tuple[()]): Knot positions.
        pos (list[float] | np.ndarray | tuple[()]): Evaluation positions.

    Returns:
        np.ndarray: Matrix of shape (len(pos), n_basis).
    """
    _ensure_jvm()
    import jpype

    BSplines = jpype.JClass("jdplus.toolkit.base.r.math.BSplines")
    jm = BSplines.of(
        int(order),
        np.asarray(knots, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
    )
    result = jd2r_matrix(jm)
    assert result is not None
    return result


def natural_cspline(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    pos: list[float] | np.ndarray,
) -> np.ndarray:
    """Natural cubic spline interpolation.

    Args:
        x (list[float] | np.ndarray): Knot abscissas.
        y (list[float] | np.ndarray): Knot ordinates.
        pos (list[float] | np.ndarray): Evaluation positions.

    Returns:
        np.ndarray: Array of spline values at the requested positions.
    """
    _ensure_jvm()
    import jpype

    CubicSplines = jpype.JClass("jdplus.toolkit.base.r.math.CubicSplines")
    return np.array(
        CubicSplines.natural(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
        )
    )


def monotonic_cspline(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    pos: list[float] | np.ndarray,
) -> np.ndarray:
    """Monotonic cubic spline interpolation.

    Args:
        x (list[float] | np.ndarray): Knot abscissas.
        y (list[float] | np.ndarray): Knot ordinates.
        pos (list[float] | np.ndarray): Evaluation positions.

    Returns:
        np.ndarray: Array of spline values at the requested positions.
    """
    _ensure_jvm()
    import jpype

    CubicSplines = jpype.JClass("jdplus.toolkit.base.r.math.CubicSplines")
    return np.array(
        CubicSplines.monotonic(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
        )
    )


def periodic_cspline(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    pos: list[float] | np.ndarray,
) -> np.ndarray:
    """Periodic cubic spline interpolation.

    Args:
        x (list[float] | np.ndarray): Knot abscissas.
        y (list[float] | np.ndarray): Knot ordinates.
        pos (list[float] | np.ndarray): Evaluation positions.

    Returns:
        np.ndarray: Array of spline values at the requested positions.
    """
    _ensure_jvm()
    import jpype

    CubicSplines = jpype.JClass("jdplus.toolkit.base.r.math.CubicSplines")
    return np.array(
        CubicSplines.periodic(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
        )
    )


def periodic_csplines(
    x: list[float] | np.ndarray,
    pos: list[float] | np.ndarray,
) -> np.ndarray:
    """Periodic cardinal cubic splines.

    Args:
        x (list[float] | np.ndarray): Knot abscissas.
        pos (list[float] | np.ndarray): Evaluation positions.

    Returns:
        np.ndarray: Matrix of shape (len(pos), len(x)).
    """
    _ensure_jvm()
    import jpype

    CubicSplines = jpype.JClass("jdplus.toolkit.base.r.math.CubicSplines")
    jm = CubicSplines.periodicCardinalSplines(
        np.asarray(x, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
    )
    result = jd2r_matrix(jm)
    assert result is not None
    return result
