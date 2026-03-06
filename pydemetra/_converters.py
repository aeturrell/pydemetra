from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from pydemetra._java import _ensure_jvm
from pydemetra._models import (
    ArimaModel,
    Likelihood,
    Parameter,
    SarimaModel,
    Span,
    StatisticalTest,
    UcarimaModel,
)
from pydemetra._proto import modelling_pb2, sa_pb2, toolkit_pb2
from pydemetra._protobuf import enum_extract, enum_of

# ---------------------------------------------------------------------------
# JPype direct converters (Java <-> Python)
# ---------------------------------------------------------------------------


def r2jd_tsdata(s: pd.Series) -> object:
    """Convert a pandas Series with PeriodIndex to a Java TsData object."""
    _ensure_jvm()
    import jpype

    if not isinstance(s.index, pd.PeriodIndex):
        raise TypeError("Series must have a PeriodIndex")

    freq = s.index.freqstr
    freq_map = {"M": 12, "Q": 4, "A": 1, "Q-DEC": 4, "M-DEC": 12, "Y-DEC": 1}
    annual_freq = freq_map.get(freq)
    if annual_freq is None:
        if hasattr(s.index, "freq") and hasattr(s.index.freq, "n"):
            if "M" in freq:
                annual_freq = 12
            elif "Q" in freq:
                annual_freq = 4
            elif "Y" in freq or "A" in freq:
                annual_freq = 1
    if annual_freq is None:
        raise ValueError(f"Unsupported frequency: {freq}")

    start_year = s.index[0].year
    if annual_freq == 12:
        start_period = s.index[0].month
    elif annual_freq == 4:
        start_period = s.index[0].quarter
    else:
        start_period = 1

    TsUtility = jpype.JClass("jdplus.toolkit.base.r.timeseries.TsUtility")
    return TsUtility.of(
        int(annual_freq),
        int(start_year),
        int(start_period),
        np.asarray(s.values, dtype=np.float64),
    )


def jd2r_tsdata(jts: object) -> pd.Series | None:
    """Convert a Java TsData object to a pandas Series with PeriodIndex."""
    _ensure_jvm()
    import jpype

    if jts is None:
        return None

    TsUtility = jpype.JClass("jdplus.toolkit.base.r.timeseries.TsUtility")
    pstart = TsUtility.startPeriod(jts)
    freq = int(pstart[0])
    year = int(pstart[1])
    period = int(pstart[2])

    jvalues = jts.getValues()
    values = np.array(jvalues.toArray())

    if len(values) == 0:
        return None

    freq_str_map = {12: "M", 4: "Q", 1: "Y"}
    freq_str = freq_str_map.get(freq, f"{freq}M")

    month = period if freq == 12 else (period - 1) * (12 // freq) + 1
    start_period = pd.Period(year=year, month=month, freq=freq_str)
    index = pd.period_range(start=start_period, periods=len(values), freq=freq_str)
    return pd.Series(values, index=index)


def r2jd_tsdomain(period: int, start_year: int, start_period: int, length: int) -> object:
    """Create a Java TsDomain."""
    _ensure_jvm()
    import jpype

    TsUtility = jpype.JClass("jdplus.toolkit.base.r.timeseries.TsUtility")
    return TsUtility.of(int(period), int(start_year), int(start_period), int(length))


def jd2r_matrix(jm: object) -> np.ndarray | None:
    """Convert a Java Matrix to a numpy array."""
    _ensure_jvm()
    if jm is None:
        return None
    nr = int(jm.getRowsCount())
    nc = int(jm.getColumnsCount())
    d = np.array(jm.toArray())
    return d.reshape((nr, nc), order="F")


def r2jd_matrix(m: np.ndarray | None) -> object:
    """Convert a numpy array to a Java Matrix."""
    _ensure_jvm()
    import jpype

    Matrix = jpype.JClass("jdplus.toolkit.base.api.math.matrices.Matrix")
    if m is None:
        return jpype.JObject(None, Matrix)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    return Matrix.of(
        np.asarray(m, dtype=np.float64, order="F").ravel(),
        int(m.shape[0]),
        int(m.shape[1]),
    )


# ---------------------------------------------------------------------------
# Protobuf <-> Python converters
# ---------------------------------------------------------------------------


def p2r_tsdata(p: toolkit_pb2.TsData) -> pd.Series | None:
    """Convert a protobuf TsData to a pandas Series."""
    if len(p.values) == 0:
        return None

    freq = p.annual_frequency
    freq_str_map = {12: "M", 4: "Q", 1: "Y"}
    freq_str = freq_str_map.get(freq, f"{freq}M")

    if freq == 12:
        month = p.start_period
    elif freq == 4:
        month = (p.start_period - 1) * 3 + 1
    else:
        month = 1

    start = pd.Period(year=p.start_year, month=month, freq=freq_str)
    index = pd.period_range(start=start, periods=len(p.values), freq=freq_str)
    s = pd.Series(list(p.values), index=index, name=p.name or None)
    return s


def r2p_tsdata(s: pd.Series) -> toolkit_pb2.TsData:
    """Convert a pandas Series to a protobuf TsData."""
    p = toolkit_pb2.TsData()
    if not isinstance(s.index, pd.PeriodIndex):
        raise TypeError("Series must have a PeriodIndex")

    freq_str = s.index.freqstr
    if "M" in freq_str:
        p.annual_frequency = 12
        p.start_period = s.index[0].month
    elif "Q" in freq_str:
        p.annual_frequency = 4
        p.start_period = s.index[0].quarter
    else:
        p.annual_frequency = 1
        p.start_period = 1

    p.start_year = s.index[0].year
    p.values.extend(s.values.tolist())
    if s.name:
        p.name = str(s.name)
    return p


def p2r_parameter(p: toolkit_pb2.Parameter) -> Parameter | None:
    """Convert a protobuf Parameter to a Python Parameter."""
    if p.type == toolkit_pb2.PARAMETER_UNUSED:
        return None
    return Parameter(
        value=p.value,
        type=enum_extract(toolkit_pb2.ParameterType, p.type),
    )


def r2p_parameter(r: Parameter | None) -> toolkit_pb2.Parameter:
    """Convert a Python Parameter to a protobuf Parameter."""
    p = toolkit_pb2.Parameter()
    if r is None:
        p.value = 0
        p.type = enum_of(toolkit_pb2.ParameterType, "UNUSED", "PARAMETER")
    else:
        p.value = r.value
        p.type = enum_of(toolkit_pb2.ParameterType, r.type, "PARAMETER")
    return p


def p2r_parameters(params: list) -> list[Parameter]:
    """Convert a list of protobuf Parameters to Python Parameters."""
    return [p2r_parameter(p) for p in params]


def p2r_test(p: toolkit_pb2.StatisticalTest | None) -> StatisticalTest | None:
    """Convert a protobuf StatisticalTest to a Python StatisticalTest."""
    if p is None:
        return None
    return StatisticalTest(
        value=p.value,
        pvalue=p.pvalue,
        distribution=p.description or None,
    )


def p2r_matrix(p: toolkit_pb2.Matrix) -> np.ndarray:
    """Convert a protobuf Matrix to a numpy array."""
    return np.array(list(p.values)).reshape((p.nrows, p.ncols), order="F")


def r2p_matrix(m: np.ndarray) -> toolkit_pb2.Matrix:
    """Convert a numpy array to a protobuf Matrix."""
    p = toolkit_pb2.Matrix()
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    p.nrows = m.shape[0]
    p.ncols = m.shape[1]
    p.values.extend(np.asarray(m, order="F").ravel().tolist())
    return p


def p2r_likelihood(p: toolkit_pb2.LikelihoodStatistics) -> Likelihood:
    """Convert a protobuf LikelihoodStatistics to a Python Likelihood."""
    return Likelihood(
        nobs=p.nobs,
        neffectiveobs=p.neffectiveobs,
        nparams=p.nparams,
        ll=p.log_likelihood,
        adjustedll=p.adjusted_log_likelihood,
        aic=p.aic,
        aicc=p.aicc,
        bic=p.bic,
        bicc=p.bicc,
        ssq=p.ssq,
    )


def p2r_date(p: toolkit_pb2.Date) -> datetime.date | None:
    """Convert a protobuf Date to a Python date."""
    if p.year == 0:
        return None
    return datetime.date(p.year, p.month or 1, p.day or 1)


def r2p_date(d: datetime.date | str | None) -> toolkit_pb2.Date:
    """Convert a Python date to a protobuf Date."""
    p = toolkit_pb2.Date()
    if d is None:
        return p
    if isinstance(d, str):
        d = datetime.date.fromisoformat(d)
    p.year = d.year
    p.month = d.month
    p.day = d.day
    return p


def p2r_span(span) -> Span:
    """Convert a protobuf TimeSelector to a Python Span."""
    return Span(
        type=enum_extract(toolkit_pb2.SelectionType, span.type),
        d0=p2r_date(span.d0),
        d1=p2r_date(span.d1),
        n0=span.n0,
        n1=span.n1,
    )


def p2r_sarima(p: modelling_pb2.SarimaModel) -> SarimaModel:
    """Convert a protobuf SarimaModel to a Python SarimaModel."""
    return SarimaModel(
        name=p.name,
        period=p.period,
        phi=list(p.phi),
        d=p.d,
        theta=list(p.theta),
        bphi=list(p.bphi),
        bd=p.bd,
        btheta=list(p.btheta),
    )


def p2r_arima(p: modelling_pb2.ArimaModel) -> ArimaModel:
    """Convert a protobuf ArimaModel to a Python ArimaModel."""
    return ArimaModel(
        name=p.name,
        ar=list(p.ar),
        delta=list(p.delta),
        ma=list(p.ma),
        var=p.innovation_variance,
    )


def p2r_ucarima(p: modelling_pb2.UcarimaModel) -> UcarimaModel:
    """Convert a protobuf UcarimaModel to a Python UcarimaModel."""
    return UcarimaModel(
        model=p2r_arima(p.model),
        components=[p2r_arima(c) for c in p.components],
        complements=[p2r_arima(c) for c in p.complements],
    )


def p2r_sa_decomposition(p: sa_pb2.SaDecomposition, full: bool = False) -> dict:
    """Convert a protobuf SaDecomposition to a Python dict."""
    mode = enum_extract(sa_pb2.DecompositionMode, p.mode)
    converter = _p2r_sa_component if full else _p2r_component

    return {
        "mode": mode,
        "series": converter(p.series),
        "sa": converter(p.seasonally_adjusted),
        "t": converter(p.trend),
        "s": converter(p.seasonal),
        "i": converter(p.irregular),
    }


def _p2r_component(p) -> dict | None:
    """Convert a protobuf TsComponent to a basic dict."""
    values = list(p.data.values)
    n = len(values)
    if n == 0:
        return None

    freq = p.data.annual_frequency
    start_year = p.data.start_year
    start_period = p.data.start_period
    nb = p.nbcasts
    nf = p.nfcasts

    result = {"data": np.array(values[nb : n - nf])}
    if nb > 0:
        result["bcasts"] = np.array(values[:nb])
    if nf > 0:
        result["fcasts"] = np.array(values[n - nf :])
    result["frequency"] = freq
    result["start_year"] = start_year
    result["start_period"] = start_period
    result["nbcasts"] = nb
    return result


def _p2r_sa_component(p) -> dict | None:
    """Convert a protobuf TsComponent with standard errors to a dict."""
    result = _p2r_component(p)
    if result is None:
        return None

    stde = list(p.stde)
    if stde:
        n = len(list(p.data.values))
        nb = p.nbcasts
        nf = p.nfcasts
        result["data_stde"] = np.array(stde[nb : n - nf])
        if nb > 0:
            result["bcasts_stde"] = np.array(stde[:nb])
        if nf > 0:
            result["fcasts_stde"] = np.array(stde[n - nf :])
    return result


def p2r_spec_benchmarking(p: sa_pb2.BenchmarkingSpec) -> dict:
    """Convert a protobuf BenchmarkingSpec to a Python dict."""
    return {
        "enabled": p.enabled,
        "target": enum_extract(sa_pb2.BenchmarkingTarget, p.target),
        "lambda": p.lambd if hasattr(p, "lambd") else p.lambda_,
        "rho": p.rho,
        "bias": enum_extract(sa_pb2.BenchmarkingBias, p.bias),
        "forecast": p.forecast,
    }


def r2p_spec_benchmarking(r: dict) -> sa_pb2.BenchmarkingSpec:
    """Convert a Python dict to a protobuf BenchmarkingSpec."""
    p = sa_pb2.BenchmarkingSpec()
    p.enabled = r["enabled"]
    p.target = enum_of(sa_pb2.BenchmarkingTarget, r["target"], "BENCH_TARGET")
    setattr(p, "lambda", r["lambda"])
    p.rho = r["rho"]
    p.bias = enum_of(sa_pb2.BenchmarkingBias, r["bias"], "BENCH_BIAS")
    p.forecast = r["forecast"]
    return p
