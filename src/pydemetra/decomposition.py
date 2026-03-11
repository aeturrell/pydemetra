from __future__ import annotations

import pandas as pd

from pydemetra._models import SaDecomposition


def sa_decomposition(
    y: pd.Series | None = None,
    sa: pd.Series | None = None,
    t: pd.Series | None = None,
    s: pd.Series | None = None,
    i: pd.Series | None = None,
    mul: bool = True,
) -> SaDecomposition:
    """Create a seasonal adjustment decomposition result.

    Args:
        y (pd.Series | None): Original series component.
        sa (pd.Series | None): Seasonally adjusted component.
        t (pd.Series | None): Trend component.
        s (pd.Series | None): Seasonal component.
        i (pd.Series | None): Irregular component.
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
