from __future__ import annotations

import numpy as np

from pydemetra._models import Likelihood


def regarima_coef(result: dict, component: str = "regression") -> np.ndarray | None:
    """Extract coefficients from RegARIMA results.

    Args:
        result: RegARIMA estimation result dict.
        component: ``"regression"``, ``"arima"``, or ``"both"``.

    Returns:
        Array of coefficients.
    """
    if component == "regression":
        return result.get("b")
    elif component == "arima":
        params = result.get("parameters")
        return params["val"] if params else None
    elif component == "both":
        b = result.get("b", np.array([]))
        params = result.get("parameters")
        p = params["val"] if params else np.array([])
        return np.concatenate([b, p])
    return None


def regarima_loglik(result: dict) -> Likelihood | None:
    """Extract log-likelihood from RegARIMA results.

    Args:
        result: RegARIMA estimation result dict.

    Returns:
        Likelihood object.
    """
    return result.get("likelihood")


def regarima_vcov(result: dict, component: str = "regression") -> np.ndarray | None:
    """Extract variance-covariance matrix from RegARIMA results.

    Args:
        result: RegARIMA estimation result dict.
        component: ``"regression"`` or ``"arima"``.

    Returns:
        Covariance matrix.
    """
    if component == "regression":
        return result.get("bcovariance")
    elif component == "arima":
        params = result.get("parameters")
        return params.get("cov") if params else None
    return None


def regarima_residuals(result: dict) -> np.ndarray | None:
    """Extract residuals from RegARIMA results.

    Args:
        result: RegARIMA estimation result dict.

    Returns:
        Array of residuals.
    """
    return result.get("residuals")


def regarima_summary(result: dict) -> str:
    """Generate a text summary of RegARIMA results.

    Args:
        result: RegARIMA estimation result dict.

    Returns:
        Formatted summary string.
    """
    lines = ["RegARIMA Estimation Summary", "=" * 40]

    orders = result.get("orders", {})
    if orders:
        o = orders.get("order", (0, 0, 0))
        s = orders.get("seasonal", {})
        so = s.get("order", (0, 0, 0)) if isinstance(s, dict) else s
        sp = s.get("period", 0) if isinstance(s, dict) else 0
        lines.append(f"ARIMA({o[0]},{o[1]},{o[2]})({so[0]},{so[1]},{so[2]})[{sp}]")

    ll = result.get("likelihood")
    if ll:
        lines.append(f"\nLog-likelihood: {ll.ll:.4f}")
        lines.append(f"AIC: {ll.aic:.4f}  BIC: {ll.bic:.4f}")
        lines.append(f"Observations: {ll.nobs}")

    b = result.get("b")
    if b is not None and len(b) > 0:
        lines.append(f"\nRegression coefficients: {b}")

    params = result.get("parameters")
    if params:
        lines.append(f"ARIMA parameters: {params['val']}")

    return "\n".join(lines)
