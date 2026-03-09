from __future__ import annotations

from pydemetra._models import SaDecomposition


def sa_decomposition(
    y: object = None,
    sa: object = None,
    t: object = None,
    s: object = None,
    i: object = None,
    mul: bool = True,
) -> SaDecomposition:
    """Create a seasonal adjustment decomposition result.

    Args:
        y (object): Original series component.
        sa (object): Seasonally adjusted component.
        t (object): Trend component.
        s (object): Seasonal component.
        i (object): Irregular component.
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
