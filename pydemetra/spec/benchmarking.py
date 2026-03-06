from __future__ import annotations

import copy


def set_benchmarking(
    spec: dict,
    enabled: bool | None = None,
    target: str | None = None,
    rho: float | None = None,
    lambda_: float | None = None,
    forecast: bool | None = None,
    bias: str | None = None,
) -> dict:
    """Configure benchmarking in a specification.

    Args:
        spec: The specification dict.
        enabled: Enable benchmarking.
        target: ``"CalendarAdjusted"`` or ``"Original"``.
        rho: Rho parameter.
        lambda_: Lambda parameter.
        forecast: Include forecast.
        bias: ``"None"``, ``"Additive"``, or ``"Multiplicative"``.

    Returns:
        Modified specification dict.
    """
    spec = copy.deepcopy(spec)
    bench = spec.setdefault("benchmarking", {})
    if enabled is not None:
        bench["enabled"] = enabled
    if target is not None:
        bench["target"] = target
    if rho is not None:
        bench["rho"] = rho
    if lambda_ is not None:
        bench["lambda"] = lambda_
    if forecast is not None:
        bench["forecast"] = forecast
    if bias is not None:
        bench["bias"] = bias
    return spec
