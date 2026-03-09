from __future__ import annotations

import numpy as np
import pandas as pd

from pydemetra._converters import jd2r_matrix, jd2r_tsdata
from pydemetra._java import _ensure_jvm
from pydemetra._models import Likelihood, StatisticalTest


def _jd2r_test(jtest: object) -> StatisticalTest | None:
    """Convert a Java StatisticalTest to a Python StatisticalTest."""
    _ensure_jvm()
    if jtest is None:
        return None
    desc = str(jtest.getDescription())
    val = float(jtest.getValue())
    pval = float(jtest.getPvalue())
    return StatisticalTest(val, max(0.0, min(1.0, pval)), desc)


def proc_numeric(jrslt: object, item_id: str) -> float | None:
    """Extract a numeric value from Java results."""
    _ensure_jvm()
    import jpype

    try:
        result = jpype.JClass("jdplus.toolkit.base.r.util.Processors").numericOf(jrslt, item_id)
        return float(result) if result is not None else None
    except Exception:
        return None


def proc_vector(jrslt: object, item_id: str) -> np.ndarray | None:
    """Extract a numeric vector from Java results."""
    _ensure_jvm()
    import jpype

    try:
        result = jpype.JClass("jdplus.toolkit.base.r.util.Processors").doublesOf(jrslt, item_id)
        return np.array(result) if result is not None else None
    except Exception:
        return None


def proc_ts(jrslt: object, item_id: str) -> pd.Series | None:
    """Extract a time series from Java results."""
    _ensure_jvm()
    import jpype

    try:
        jts = jpype.JClass("jdplus.toolkit.base.r.util.Processors").tsOf(jrslt, item_id)
        return jd2r_tsdata(jts)
    except Exception:
        return None


def proc_matrix(jrslt: object, item_id: str) -> np.ndarray | None:
    """Extract a matrix from Java results."""
    _ensure_jvm()
    import jpype

    try:
        jm = jpype.JClass("jdplus.toolkit.base.r.util.Processors").matrixOf(jrslt, item_id)
        return jd2r_matrix(jm)
    except Exception:
        return None


def proc_test(jrslt: object, item_id: str) -> StatisticalTest | None:
    """Extract a statistical test from Java results."""
    _ensure_jvm()
    import jpype

    try:
        jtest = jpype.JClass("jdplus.toolkit.base.r.util.Processors").testOf(jrslt, item_id)
        return _jd2r_test(jtest)
    except Exception:
        return None


def proc_str(jrslt: object, item_id: str) -> str | None:
    """Extract a string from Java results."""
    _ensure_jvm()
    import jpype

    try:
        result = jpype.JClass("jdplus.toolkit.base.r.util.Processors").stringOf(jrslt, item_id)
        return str(result) if result is not None else None
    except Exception:
        return None


def proc_bool(jrslt: object, item_id: str) -> bool | None:
    """Extract a boolean from Java results."""
    _ensure_jvm()
    import jpype

    try:
        result = jpype.JClass("jdplus.toolkit.base.r.util.Processors").boolOf(jrslt, item_id)
        return bool(result) if result is not None else None
    except Exception:
        return None


def proc_data(jrslt: object, item_id: str) -> dict | None:
    """Extract data items from Java results."""
    _ensure_jvm()
    import jpype

    try:
        jdict = jpype.JClass("jdplus.toolkit.base.r.util.Processors").dictionaryOf(jrslt)
        if jdict is None:
            return None
        items = list(jdict)
        return {str(item): proc_numeric(jrslt, str(item)) for item in items}
    except Exception:
        return None


def proc_dictionary(jrslt: object) -> list[str]:
    """Get the available item names from Java results."""
    _ensure_jvm()
    import jpype

    try:
        jdict = jpype.JClass("jdplus.toolkit.base.r.util.Processors").dictionaryOf(jrslt)
        return [str(item) for item in jdict] if jdict is not None else []
    except Exception:
        return []


def proc_likelihood(jrslt: object, prefix: str = "") -> Likelihood:
    """Extract likelihood information from Java results."""
    nobs = proc_numeric(jrslt, f"{prefix}likelihood.nobs") or 0
    neffobs = proc_numeric(jrslt, f"{prefix}likelihood.neffectiveobs") or int(nobs)
    nparams = proc_numeric(jrslt, f"{prefix}likelihood.nparams") or 0
    ll = proc_numeric(jrslt, f"{prefix}likelihood.ll") or 0.0
    adjustedll = proc_numeric(jrslt, f"{prefix}likelihood.adjustedll") or ll
    aic = proc_numeric(jrslt, f"{prefix}likelihood.aic") or 0.0
    aicc = proc_numeric(jrslt, f"{prefix}likelihood.aicc") or 0.0
    bic = proc_numeric(jrslt, f"{prefix}likelihood.bic") or 0.0
    bicc = proc_numeric(jrslt, f"{prefix}likelihood.bicc") or 0.0
    ssq = proc_numeric(jrslt, f"{prefix}likelihood.ssq") or 0.0

    return Likelihood(
        nobs=int(nobs),
        neffectiveobs=int(neffobs),
        nparams=int(nparams),
        ll=ll,
        adjustedll=adjustedll,
        aic=aic,
        aicc=aicc,
        bic=bic,
        bicc=bicc,
        ssq=ssq,
    )
