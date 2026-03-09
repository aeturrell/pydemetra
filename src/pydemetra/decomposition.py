from __future__ import annotations

from pydemetra._models import SaDecomposition


def sa_decomposition(
    y: dict | None = None,
    sa: dict | None = None,
    t: dict | None = None,
    s: dict | None = None,
    i: dict | None = None,
    mul: bool = True,
) -> SaDecomposition:
    """Create a seasonal adjustment decomposition result.

    Args:
        y (dict | None): Original series component.
        sa (dict | None): Seasonally adjusted component.
        t (dict | None): Trend component.
        s (dict | None): Seasonal component.
        i (dict | None): Irregular component.
        mul (bool): True for multiplicative decomposition.

    Returns:
        SaDecomposition: SaDecomposition dataclass.
    """
    mode = "MULTIPLICATIVE" if mul else "ADDITIVE"
    return SaDecomposition(
        mode=mode,
        series=y,
        sa=sa,
        t=t,
        s=s,
        i=i,
    )
