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
    values = np.asarray(s.values, dtype=np.float64)
    jvalues = jpype.JArray(jpype.JDouble)(values.tolist())
    return TsUtility.of(
        int(annual_freq),
        int(start_year),
        int(start_period),
        jvalues,
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
        "lambda": getattr(p, "lambda"),
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


# ---------------------------------------------------------------------------
# Span: Python -> Protobuf
# ---------------------------------------------------------------------------


def r2p_span(rspan: Span) -> toolkit_pb2.TimeSelector:
    """Convert a Python Span to a protobuf TimeSelector."""
    p = toolkit_pb2.TimeSelector()
    p.type = enum_of(toolkit_pb2.SelectionType, rspan.type, "SPAN")
    p.n0 = rspan.n0
    p.n1 = rspan.n1
    p.d0.CopyFrom(r2p_date(rspan.d0))
    p.d1.CopyFrom(r2p_date(rspan.d1))
    return p


# ---------------------------------------------------------------------------
# SARIMA spec round-trip (for regarima spec parameters)
# ---------------------------------------------------------------------------


def p2r_spec_sarima(spec) -> dict:
    """Convert a protobuf SarimaSpec to a Python dict."""
    return {
        "period": spec.period,
        "d": spec.d,
        "bd": spec.bd,
        "phi": p2r_parameters(list(spec.phi)),
        "theta": p2r_parameters(list(spec.theta)),
        "bphi": p2r_parameters(list(spec.bphi)),
        "btheta": p2r_parameters(list(spec.btheta)),
    }


def r2p_spec_sarima(r: dict):
    """Convert a Python dict to a protobuf SarimaSpec."""
    from pydemetra._proto import regarima_pb2

    p = regarima_pb2.SarimaSpec()
    p.period = r["period"]
    p.d = r["d"]
    p.bd = r["bd"]
    for param in r2p_parameters(r["phi"]):
        p.phi.append(param)
    for param in r2p_parameters(r["theta"]):
        p.theta.append(param)
    for param in r2p_parameters(r["bphi"]):
        p.bphi.append(param)
    for param in r2p_parameters(r["btheta"]):
        p.btheta.append(param)
    return p


def r2p_parameters(params: list) -> list:
    """Convert a list of Python Parameters to protobuf Parameters."""
    if not params:
        return []
    return [r2p_parameter(p) for p in params if p is not None]


# ---------------------------------------------------------------------------
# Outlier / Ramp / UserVar / InterventionVariable round-trips
# ---------------------------------------------------------------------------


def p2r_outlier(p) -> dict:
    """Convert a protobuf Outlier to a Python dict."""
    return {
        "name": p.name,
        "pos": p2r_date(p.position),
        "code": p.code,
        "coef": p2r_parameter(p.coefficient),
    }


def r2p_outlier(r: dict):
    """Convert a Python outlier dict to a protobuf Outlier."""
    p = modelling_pb2.Outlier()
    p.name = r["name"]
    p.code = r["code"]
    p.position.CopyFrom(r2p_date(r["pos"]))
    p.coefficient.CopyFrom(r2p_parameter(r["coef"]))
    return p


def p2r_outliers(p) -> list[dict] | None:
    """Convert a list of protobuf Outliers to Python dicts."""
    if len(p) == 0:
        return None
    return [p2r_outlier(o) for o in p]


def r2p_outliers(r: list[dict] | None) -> list:
    """Convert a list of Python outlier dicts to protobuf Outliers."""
    if not r:
        return []
    return [r2p_outlier(o) for o in r]


def p2r_ramp(p) -> dict:
    """Convert a protobuf Ramp to a Python dict."""
    return {
        "name": p.name,
        "start": p2r_date(p.start),
        "end": p2r_date(p.end),
        "coef": p2r_parameter(p.coefficient),
    }


def r2p_ramp(r: dict):
    """Convert a Python ramp dict to a protobuf Ramp."""
    p = modelling_pb2.Ramp()
    p.name = r["name"]
    p.start.CopyFrom(r2p_date(r["start"]))
    p.end.CopyFrom(r2p_date(r["end"]))
    p.coefficient.CopyFrom(r2p_parameter(r["coef"]))
    return p


def p2r_ramps(p) -> list[dict] | None:
    """Convert a list of protobuf Ramps to Python dicts."""
    if len(p) == 0:
        return None
    return [p2r_ramp(r) for r in p]


def r2p_ramps(r: list[dict] | None) -> list:
    """Convert a list of Python ramp dicts to protobuf Ramps."""
    if not r:
        return []
    return [r2p_ramp(ramp) for ramp in r]


def _regeffect(metadata) -> str:
    """Extract regeffect from protobuf metadata."""
    for entry in metadata:
        if entry.key == "regeffect":
            return entry.value
    return "Undefined"


def p2r_uservar(p) -> dict:
    """Convert a protobuf TsVariable to a Python dict."""
    return {
        "id": p.id,
        "name": p.name,
        "lag": p.lag,
        "coef": p2r_parameter(p.coefficient),
        "regeffect": _regeffect(p.metadata),
    }


def r2p_uservar(r: dict):
    """Convert a Python user variable dict to a protobuf TsVariable."""
    p = modelling_pb2.TsVariable()
    p.name = r["name"]
    p.id = r["id"]
    p.lag = r["lag"]
    p.coefficient.CopyFrom(r2p_parameter(r["coef"]))
    entry = p.metadata.add()
    entry.key = "regeffect"
    entry.value = r.get("regeffect", "Undefined")
    return p


def p2r_uservars(p) -> list[dict] | None:
    """Convert a list of protobuf TsVariables to Python dicts."""
    if len(p) == 0:
        return None
    return [p2r_uservar(u) for u in p]


def r2p_uservars(r: list[dict] | None) -> list:
    """Convert a list of Python user variable dicts to protobuf TsVariables."""
    if not r:
        return []
    return [r2p_uservar(u) for u in r]


def _p2r_sequences(seqs) -> list[dict]:
    """Convert protobuf InterventionVariable sequences to Python dicts."""
    return [{"start": p2r_date(s.start), "end": p2r_date(s.end)} for s in seqs]


def _r2p_sequences(seqs: list[dict]) -> list:
    """Convert Python sequence dicts to protobuf sequences."""
    from pydemetra._proto import modelling_pb2 as _m

    result = []
    for s in seqs:
        seq = _m.InterventionVariable.Sequence()
        seq.start.CopyFrom(r2p_date(s["start"]))
        seq.end.CopyFrom(r2p_date(s["end"]))
        result.append(seq)
    return result


def p2r_iv(p) -> dict:
    """Convert a protobuf InterventionVariable to a Python dict."""
    return {
        "name": p.name,
        "sequences": _p2r_sequences(p.sequences),
        "delta": p.delta,
        "seasonaldelta": p.seasonal_delta,
        "coef": p2r_parameter(p.coefficient),
        "regeffect": _regeffect(p.metadata),
    }


def r2p_iv(r: dict):
    """Convert a Python intervention variable dict to protobuf."""
    p = modelling_pb2.InterventionVariable()
    p.name = r["name"]
    for seq in _r2p_sequences(r["sequences"]):
        p.sequences.append(seq)
    p.delta = r.get("delta", 0.0)
    p.seasonal_delta = r.get("seasonaldelta", 0.0)
    p.coefficient.CopyFrom(r2p_parameter(r["coef"]))
    entry = p.metadata.add()
    entry.key = "regeffect"
    entry.value = r.get("regeffect", "Undefined")
    return p


def p2r_ivs(p) -> list[dict] | None:
    """Convert a list of protobuf InterventionVariables to Python dicts."""
    if len(p) == 0:
        return None
    return [p2r_iv(iv) for iv in p]


def r2p_ivs(r: list[dict] | None) -> list:
    """Convert a list of Python intervention variable dicts to protobuf."""
    if not r:
        return []
    return [r2p_iv(iv) for iv in r]


# ---------------------------------------------------------------------------
# RegARIMA results (protobuf -> Python)
# ---------------------------------------------------------------------------


def _p2r_variables(variables) -> list[dict]:
    """Convert protobuf RegressionVariables to Python dicts."""
    return [
        {
            "name": v.name,
            "var_type": enum_extract(modelling_pb2.VariableType, v.var_type),
            "coefficients": p2r_parameters(list(v.coefficients)),
            "metadata": dict(v.metadata),
        }
        for v in variables
    ]


def _p2r_parameters_estimation(p) -> dict | None:
    """Convert protobuf ParametersEstimation to a Python dict."""
    if p is None:
        return None
    return {
        "values": list(p.value),
        "score": list(p.score),
        "covariance": p2r_matrix(p.covariance) if p.covariance.nrows > 0 else None,
        "description": p.description or None,
    }


def p2r_regarima_rslts(p) -> dict:
    """Convert a protobuf RegArimaModel to a Python dict."""
    description = {
        "log": p.description.log,
        "preadjustment": enum_extract(modelling_pb2.LengthOfPeriod, p.description.preadjustment),
        "arima": p2r_spec_sarima(p.description.arima),
        "variables": _p2r_variables(p.description.variables),
    }
    estimation = {
        "y": list(p.estimation.y),
        "X": p2r_matrix(p.estimation.x) if p.estimation.x.nrows > 0 else None,
        "parameters": _p2r_parameters_estimation(p.estimation.parameters),
        "b": list(p.estimation.b),
        "bvar": (
            p2r_matrix(p.estimation.bcovariance) if p.estimation.bcovariance.nrows > 0 else None
        ),
        "likelihood": p2r_likelihood(p.estimation.likelihood),
        "res": list(p.estimation.residuals),
    }
    diagnostics = {}
    for key, value in p.diagnostics.residuals_tests.items():
        diagnostics[key] = p2r_test(value)
    return {
        "description": description,
        "estimation": estimation,
        "diagnostics": diagnostics,
    }


def p2r_sa_diagnostics(p) -> dict:
    """Convert a protobuf sa.Diagnostics to a Python dict."""
    return {
        "vardecomposition": {
            "cycle": p.variance_decomposition.cycle,
            "seasonal": p.variance_decomposition.seasonal,
            "irregular": p.variance_decomposition.irregular,
            "calendar": p.variance_decomposition.calendar,
            "others": p.variance_decomposition.others,
            "total": p.variance_decomposition.total,
        },
        "seas.ftest.i": p2r_test(p.seasonal_ftest_on_irregular),
        "seas.ftest.sa": p2r_test(p.seasonal_ftest_on_sa),
        "seas.qstest.i": p2r_test(p.seasonal_qtest_on_irregular),
        "seas.qstest.sa": p2r_test(p.seasonal_qtest_on_sa),
        "td.ftest.i": p2r_test(p.td_ftest_on_irregular),
        "td.ftest.sa": p2r_test(p.td_ftest_on_sa),
    }
