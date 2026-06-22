from __future__ import annotations

import pandas as pd
from pydemetra.decomposition import sa_decomposition


class TestSaDecomposition:
    def test_creation(self):
        idx = pd.period_range("2020-01", periods=12, freq="M")
        s = pd.Series(range(12), index=idx, dtype=float)
        d = sa_decomposition(y=s, sa=s, t=s, s=s, i=s, mul=False)
        assert d.mode == "ADDITIVE"
        assert d.series is not None
