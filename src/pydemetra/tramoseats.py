from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pandas as pd

from pydemetra._converters import (
    p2r_arima,
    p2r_ivs as _p2r_ivs,
    p2r_outliers as _p2r_outliers,
    p2r_parameter,
    p2r_parameters,
    p2r_ramps as _p2r_ramps,
    p2r_regarima_rslts,
    p2r_sa_decomposition,
    p2r_sa_diagnostics,
    p2r_span,
    p2r_spec_benchmarking,
    p2r_spec_sarima,
    p2r_tsdata,
    p2r_ucarima,
    p2r_uservars,
    r2jd_tsdata,
    r2p_ivs as _r2p_ivs,
    r2p_outliers as _r2p_outliers,
    r2p_parameter,
    r2p_parameters,
    r2p_ramps as _r2p_ramps,
    r2p_span,
    r2p_spec_benchmarking,
    r2p_spec_sarima,
    r2p_uservars as _r2p_uservars,
)
from pydemetra._java import _ensure_jvm
from pydemetra._protobuf import enum_extract, enum_of, enum_sextract, enum_sof

if TYPE_CHECKING:
    modelling_pb2: Any
    tramoseats_pb2: Any
else:
    from pydemetra._proto import modelling_pb2, tramoseats_pb2

_TRAMO_SPEC_NAMES = ("trfull", "tr0", "tr1", "tr2", "tr3", "tr4", "tr5")
_TRAMOSEATS_SPEC_NAMES = ("rsafull", "rsa0", "rsa1", "rsa2", "rsa3", "rsa4", "rsa5")

_APPROXIMATIONS = {
    "NONE": "APP_NONE",
    "LEGACY": "APP_LEGACY",
    "NOISY": "APP_NOISY",
}
_ALGORITHMS = {
    "BURMAN": "ALG_BURMAN",
    "KALMANSMOOTHER": "ALG_KALMANSMOOTHER",
}


def tramo_spec(name: str = "trfull") -> dict:
    """Create a default TRAMO specification.

    Args:
        name (str): Predefined specification name. One of
            ``"trfull"``, ``"tr0"``--``"tr5"``.

    Returns:
        dict: A TRAMO specification dict.

    Raises:
        ValueError: If *name* is not a recognised specification name.
    """
    _ensure_jvm()
    import jpype

    name = name.lower().replace("rsa", "tr")
    if name not in _TRAMO_SPEC_NAMES:
        raise ValueError(f"Unknown TRAMO spec: {name!r}. Choose from {_TRAMO_SPEC_NAMES}")

    TramoSpec = jpype.JClass("jdplus.tramoseats.base.api.tramo.TramoSpec")
    jspec = TramoSpec.fromString(name)
    return _jd2r_spec_tramo(jspec)


def tramoseats_spec(name: str = "rsafull") -> dict:
    """Create a default TRAMO-SEATS specification.

    Args:
        name (str): Predefined specification name. One of
            ``"rsafull"``, ``"rsa0"``--``"rsa5"``.

    Returns:
        dict: A specification dict with ``tramo``, ``seats``, and ``benchmarking`` keys.

    Raises:
        ValueError: If *name* is not a recognised specification name.
    """
    _ensure_jvm()
    import jpype

    name = name.lower().replace("tr", "rsa")
    if name not in _TRAMOSEATS_SPEC_NAMES:
        raise ValueError(
            f"Unknown TRAMO-SEATS spec: {name!r}. Choose from {_TRAMOSEATS_SPEC_NAMES}"
        )

    TramoSeatsSpec = jpype.JClass("jdplus.tramoseats.base.api.tramoseats.TramoSeatsSpec")
    jspec = TramoSeatsSpec.fromString(name)
    return _jd2r_spec_tramoseats(jspec)


def set_seats(
    spec: dict,
    *,
    approximation: str | None = None,
    trend_boundary: float | None = None,
    seas_boundary: float | None = None,
    seas_boundary_unique: float | None = None,
    seas_tolerance: float | None = None,
    ma_boundary: float | None = None,
    fcasts: int | None = None,
    bcasts: int | None = None,
    algorithm: str | None = None,
    bias: bool | None = None,
) -> dict:
    """Modify the SEATS decomposition parameters in a specification.

    Works on both SEATS specs and full TRAMO-SEATS specs (modifying the ``seats`` sub-dict).

    Args:
        spec (dict): A SEATS or TRAMO-SEATS specification dict.
        approximation (str | None): Handling of non-admissible models
            (``"None"``, ``"Legacy"``, ``"Noisy"``).
        trend_boundary (float | None): Trend boundary [0, 1].
        seas_boundary (float | None): Seasonal boundary [0, 1].
        seas_boundary_unique (float | None): Seasonal boundary at pi [0, 1].
        seas_tolerance (float | None): Seasonal tolerance in degrees [0, 10].
        ma_boundary (float | None): MA unit root boundary [0.9, 1.0].
        fcasts (int | None): Number of forecasts (negative = years).
        bcasts (int | None): Number of backcasts (negative = years).
        algorithm (str | None): Decomposition algorithm
            (``"Burman"`` or ``"KalmanSmoother"``).
        bias (bool | None): Apply bias correction.

    Returns:
        dict: A modified copy of the specification.
    """
    spec = copy.deepcopy(spec)

    if "seats" in spec:
        spec["seats"] = _set_seats_inner(
            spec["seats"],
            approximation=approximation,
            trend_boundary=trend_boundary,
            seas_boundary=seas_boundary,
            seas_boundary_unique=seas_boundary_unique,
            seas_tolerance=seas_tolerance,
            ma_boundary=ma_boundary,
            fcasts=fcasts,
            bcasts=bcasts,
            algorithm=algorithm,
            bias=bias,
        )
        return spec

    return _set_seats_inner(
        spec,
        approximation=approximation,
        trend_boundary=trend_boundary,
        seas_boundary=seas_boundary,
        seas_boundary_unique=seas_boundary_unique,
        seas_tolerance=seas_tolerance,
        ma_boundary=ma_boundary,
        fcasts=fcasts,
        bcasts=bcasts,
        algorithm=algorithm,
        bias=bias,
    )


def _set_seats_inner(
    x: dict,
    approximation: str | None = None,
    trend_boundary: float | None = None,
    seas_boundary: float | None = None,
    seas_boundary_unique: float | None = None,
    seas_tolerance: float | None = None,
    ma_boundary: float | None = None,
    fcasts: int | None = None,
    bcasts: int | None = None,
    algorithm: str | None = None,
    bias: bool | None = None,
) -> dict:
    x = copy.deepcopy(x)

    if approximation is not None:
        approx_upper = approximation.upper()
        if approx_upper not in _APPROXIMATIONS:
            raise ValueError(
                f"Invalid approximation: {approximation!r}. Choose from {tuple(_APPROXIMATIONS)}"
            )
        x["approximation"] = _APPROXIMATIONS[approx_upper]

    if trend_boundary is not None:
        x["rmod"] = trend_boundary
    if seas_boundary is not None:
        x["sbound"] = seas_boundary
    if seas_boundary_unique is not None:
        x["sboundatpi"] = seas_boundary_unique
    if seas_tolerance is not None:
        x["epsphi"] = seas_tolerance
    if ma_boundary is not None:
        x["xl"] = ma_boundary
    if fcasts is not None:
        x["nfcasts"] = fcasts
    if bcasts is not None:
        x["nbcasts"] = bcasts

    if algorithm is not None:
        algo_upper = algorithm.upper()
        if algo_upper not in _ALGORITHMS:
            raise ValueError(
                f"Invalid algorithm: {algorithm!r}. Choose from {tuple(_ALGORITHMS)}"
            )
        x["algorithm"] = _ALGORITHMS[algo_upper]

    if bias is not None:
        x["bias"] = bool(bias)

    return x


def tramoseats(
    ts: pd.Series,
    spec: str | dict = "rsafull",
    context: Any | None = None,
    userdefined: list[str] | None = None,
) -> dict | None:
    """Run full TRAMO-SEATS seasonal adjustment.

    Args:
        ts (pd.Series): Univariate time series with PeriodIndex.
        spec (str | dict): Specification name or dict.
        context (Any | None): Modelling context with external regressors.
        userdefined (list[str] | None): Additional output variable names.

    Returns:
        dict | None: Dict with ``result``, ``estimation_spec``, and ``result_spec`` keys,
            or ``None`` if processing failed.

    Raises:
        ValueError: If *spec* is a string that is not a recognised specification name.
    """
    _ensure_jvm()
    import jpype

    jts = r2jd_tsdata(ts)
    TramoSeats = jpype.JClass("jdplus.tramoseats.base.r.TramoSeats")

    if isinstance(spec, str):
        spec_name = spec.lower().replace("tr", "rsa")
        if spec_name not in _TRAMOSEATS_SPEC_NAMES:
            raise ValueError(f"Unknown spec: {spec!r}")
        jrslt = TramoSeats.fullProcess(jts, spec_name)
    else:
        jspec = _r2jd_spec_tramoseats(spec)
        if context is None:
            ModellingContext = jpype.JClass(
                "jdplus.toolkit.base.api.timeseries.regression.ModellingContext"
            )
            jcontext = jpype.JObject(None, ModellingContext)
        else:
            from pydemetra.context import _r2jd_modellingcontext

            jcontext = _r2jd_modellingcontext(context)
        jrslt = TramoSeats.fullProcess(jts, jspec, jcontext)

    if jrslt is None:
        return None

    return _tramoseats_output(jrslt)


def tramoseats_fast(
    ts: pd.Series,
    spec: str | dict = "rsafull",
    context: Any | None = None,
    userdefined: list[str] | None = None,
) -> dict | None:
    """Run TRAMO-SEATS seasonal adjustment (results only, faster).

    Args:
        ts (pd.Series): Univariate time series with PeriodIndex.
        spec (str | dict): Specification name or dict.
        context (Any | None): Modelling context with external regressors.
        userdefined (list[str] | None): Additional output variable names.

    Returns:
        dict | None: Dict with TRAMO-SEATS results, or ``None`` if processing failed.

    Raises:
        ValueError: If *spec* is a string that is not a recognised specification name.
    """
    _ensure_jvm()
    import jpype

    jts = r2jd_tsdata(ts)
    TramoSeats = jpype.JClass("jdplus.tramoseats.base.r.TramoSeats")

    if isinstance(spec, str):
        spec_name = spec.lower().replace("tr", "rsa")
        if spec_name not in _TRAMOSEATS_SPEC_NAMES:
            raise ValueError(f"Unknown spec: {spec!r}")
        jrslt = TramoSeats.process(jts, spec_name)
    else:
        jspec = _r2jd_spec_tramoseats(spec)
        if context is None:
            ModellingContext = jpype.JClass(
                "jdplus.toolkit.base.api.timeseries.regression.ModellingContext"
            )
            jcontext = jpype.JObject(None, ModellingContext)
        else:
            from pydemetra.context import _r2jd_modellingcontext

            jcontext = _r2jd_modellingcontext(context)
        jrslt = TramoSeats.process(jts, jspec, jcontext)

    if jrslt is None:
        return None

    return _tramoseats_rslts(jrslt)


def tramo(
    ts: pd.Series,
    spec: str | dict = "trfull",
    context: Any | None = None,
    userdefined: list[str] | None = None,
) -> dict | None:
    """Run full TRAMO modelling.

    Args:
        ts (pd.Series): Univariate time series with PeriodIndex.
        spec (str | dict): Specification name or dict.
        context (Any | None): Modelling context with external regressors.
        userdefined (list[str] | None): Additional output variable names.

    Returns:
        dict | None: Dict with ``result``, ``estimation_spec``, and ``result_spec`` keys,
            or ``None`` if processing failed.

    Raises:
        ValueError: If *spec* is a string that is not a recognised specification name.
    """
    _ensure_jvm()
    import jpype

    jts = r2jd_tsdata(ts)
    Tramo = jpype.JClass("jdplus.tramoseats.base.r.Tramo")

    if isinstance(spec, str):
        spec_name = spec.lower().replace("rsa", "tr")
        if spec_name not in _TRAMO_SPEC_NAMES:
            raise ValueError(f"Unknown spec: {spec!r}")
        jrslt = Tramo.fullProcess(jts, spec_name)
    else:
        jspec = _r2jd_spec_tramo(spec)
        if context is None:
            ModellingContext = jpype.JClass(
                "jdplus.toolkit.base.api.timeseries.regression.ModellingContext"
            )
            jcontext = jpype.JObject(None, ModellingContext)
        else:
            from pydemetra.context import _r2jd_modellingcontext

            jcontext = _r2jd_modellingcontext(context)
        jrslt = Tramo.fullProcess(jts, jspec, jcontext)

    if jrslt is None:
        return None

    return _tramo_output(jrslt)


def tramo_fast(
    ts: pd.Series,
    spec: str | dict = "trfull",
    context: Any | None = None,
    userdefined: list[str] | None = None,
) -> dict | None:
    """Run TRAMO modelling (results only, faster).

    Args:
        ts (pd.Series): Univariate time series with PeriodIndex.
        spec (str | dict): Specification name or dict.
        context (Any | None): Modelling context with external regressors.
        userdefined (list[str] | None): Additional output variable names.

    Returns:
        dict | None: Dict with TRAMO results, or ``None`` if processing failed.

    Raises:
        ValueError: If *spec* is a string that is not a recognised specification name.
    """
    _ensure_jvm()
    import jpype

    jts = r2jd_tsdata(ts)
    Tramo = jpype.JClass("jdplus.tramoseats.base.r.Tramo")

    if isinstance(spec, str):
        spec_name = spec.lower().replace("rsa", "tr")
        if spec_name not in _TRAMO_SPEC_NAMES:
            raise ValueError(f"Unknown spec: {spec!r}")
        jrslt = Tramo.process(jts, spec_name)
    else:
        jspec = _r2jd_spec_tramo(spec)
        if context is None:
            ModellingContext = jpype.JClass(
                "jdplus.toolkit.base.api.timeseries.regression.ModellingContext"
            )
            jcontext = jpype.JObject(None, ModellingContext)
        else:
            from pydemetra.context import _r2jd_modellingcontext

            jcontext = _r2jd_modellingcontext(context)
        jrslt = Tramo.process(jts, jspec, jcontext)

    if jrslt is None:
        return None

    return _tramo_fast_rslts(jrslt)


def tramoseats_dictionary() -> list[str]:
    """Return all available TRAMO-SEATS output variable names."""
    _ensure_jvm()
    import jpype

    TramoSeats = jpype.JClass("jdplus.tramoseats.base.r.TramoSeats")
    return list(TramoSeats.dictionary())


def _jd2r_spec_tramo(jspec) -> dict:
    import jpype

    Tramo = jpype.JClass("jdplus.tramoseats.base.r.Tramo")
    b = bytes(Tramo.toBuffer(jspec))
    p = tramoseats_pb2.TramoSpec()
    p.ParseFromString(b)
    return _p2r_spec_tramo(p)


def _r2jd_spec_tramo(spec: dict):
    import jpype

    p = _r2p_spec_tramo(spec)
    b = p.SerializeToString()
    Tramo = jpype.JClass("jdplus.tramoseats.base.r.Tramo")
    return Tramo.specOf(b)


def _jd2r_spec_tramoseats(jspec) -> dict:
    import jpype

    TramoSeats = jpype.JClass("jdplus.tramoseats.base.r.TramoSeats")
    b = bytes(TramoSeats.toBuffer(jspec))
    p = tramoseats_pb2.Spec()
    p.ParseFromString(b)
    return _p2r_spec_tramoseats(p)


def _r2jd_spec_tramoseats(spec: dict):
    import jpype

    p = _r2p_spec_tramoseats(spec)
    b = p.SerializeToString()
    TramoSeats = jpype.JClass("jdplus.tramoseats.base.r.TramoSeats")
    return TramoSeats.specOf(b)


def _p2r_spec_tramo(pspec) -> dict:
    basic = {
        "span": p2r_span(pspec.basic.span),
        "preliminaryCheck": pspec.basic.preliminary_check,
    }
    transform = {
        "fn": enum_extract(modelling_pb2.Transformation, pspec.transform.transformation),
        "fct": pspec.transform.fct,
        "adjust": enum_extract(modelling_pb2.LengthOfPeriod, pspec.transform.adjust),
        "outliers": pspec.transform.outliers_correction,
    }
    automodel = {
        "enabled": pspec.automodel.enabled,
        "acceptdef": pspec.automodel.accept_def,
        "cancel": pspec.automodel.cancel,
        "ub1": pspec.automodel.ub1,
        "ub2": pspec.automodel.ub2,
        "pcr": pspec.automodel.pcr,
        "pc": pspec.automodel.pc,
        "tsig": pspec.automodel.tsig,
        "amicompare": pspec.automodel.ami_compare,
    }
    arima = p2r_spec_sarima(pspec.arima)
    outlier = {
        "enabled": pspec.outlier.enabled,
        "span": p2r_span(pspec.outlier.span),
        "ao": pspec.outlier.ao,
        "ls": pspec.outlier.ls,
        "tc": pspec.outlier.tc,
        "so": pspec.outlier.so,
        "va": pspec.outlier.va,
        "tcrate": pspec.outlier.tcrate,
        "ml": pspec.outlier.ml,
    }
    ptd = pspec.regression.td
    td = {
        "td": enum_sextract(modelling_pb2.TradingDays, ptd.td),
        "lp": enum_extract(modelling_pb2.LengthOfPeriod, ptd.lp),
        "holidays": ptd.holidays,
        "users": list(ptd.users),
        "w": ptd.w,
        "test": enum_extract(tramoseats_pb2.TradingDaysTest, ptd.test),
        "auto": enum_extract(tramoseats_pb2.AutomaticTradingDays, ptd.auto),
        "ptest": ptd.ptest,
        "autoadjust": ptd.auto_adjust,
        "tdcoefficients": p2r_parameters(list(ptd.tdcoefficients)),
        "lpcoefficient": p2r_parameter(ptd.lpcoefficient),
    }
    pee = pspec.regression.easter
    easter = {
        "type": enum_extract(tramoseats_pb2.EasterType, pee.type),
        "duration": pee.duration,
        "julian": pee.julian,
        "test": pee.test,
        "coefficient": p2r_parameter(pee.coefficient),
    }
    regression = {
        "mean": p2r_parameter(pspec.regression.mean),
        "check_mean": pspec.regression.check_mean,
        "td": td,
        "easter": easter,
        "outliers": _p2r_outliers(pspec.regression.outliers),
        "users": p2r_uservars(pspec.regression.users),
        "interventions": _p2r_ivs(pspec.regression.interventions),
        "ramps": _p2r_ramps(pspec.regression.ramps),
    }
    estimate = {
        "span": p2r_span(pspec.estimate.span),
        "ml": pspec.estimate.ml,
        "tol": pspec.estimate.tol,
        "ubp": pspec.estimate.ubp,
    }
    return {
        "basic": basic,
        "transform": transform,
        "outlier": outlier,
        "arima": arima,
        "automodel": automodel,
        "regression": regression,
        "estimate": estimate,
    }


def _r2p_spec_tramo(r: dict):
    p = tramoseats_pb2.TramoSpec()

    p.basic.span.CopyFrom(r2p_span(r["basic"]["span"]))
    p.basic.preliminary_check = r["basic"]["preliminaryCheck"]

    p.transform.transformation = enum_of(
        modelling_pb2.Transformation, r["transform"]["fn"], "FN"
    )
    p.transform.fct = r["transform"]["fct"]
    p.transform.adjust = enum_of(modelling_pb2.LengthOfPeriod, r["transform"]["adjust"], "LP")
    p.transform.outliers_correction = r["transform"]["outliers"]

    p.outlier.enabled = r["outlier"]["enabled"]
    p.outlier.span.CopyFrom(r2p_span(r["outlier"]["span"]))
    p.outlier.ao = r["outlier"]["ao"]
    p.outlier.ls = r["outlier"]["ls"]
    p.outlier.tc = r["outlier"]["tc"]
    p.outlier.so = r["outlier"]["so"]
    p.outlier.va = r["outlier"]["va"]
    p.outlier.tcrate = r["outlier"]["tcrate"]
    p.outlier.ml = r["outlier"]["ml"]

    p.automodel.enabled = r["automodel"]["enabled"]
    p.automodel.cancel = r["automodel"]["cancel"]
    p.automodel.ub1 = r["automodel"]["ub1"]
    p.automodel.ub2 = r["automodel"]["ub2"]
    p.automodel.pcr = r["automodel"]["pcr"]
    p.automodel.pc = r["automodel"]["pc"]
    p.automodel.tsig = r["automodel"]["tsig"]
    p.automodel.accept_def = r["automodel"]["acceptdef"]
    p.automodel.ami_compare = r["automodel"]["amicompare"]

    p.arima.CopyFrom(r2p_spec_sarima(r["arima"]))

    p.regression.mean.CopyFrom(r2p_parameter(r["regression"]["mean"]))
    p.regression.check_mean = r["regression"]["check_mean"]
    for o in _r2p_outliers(r["regression"]["outliers"]):
        p.regression.outliers.append(o)
    for u in _r2p_uservars(r["regression"]["users"]):
        p.regression.users.append(u)
    for iv in _r2p_ivs(r["regression"]["interventions"]):
        p.regression.interventions.append(iv)
    for ramp in _r2p_ramps(r["regression"]["ramps"]):
        p.regression.ramps.append(ramp)

    p.regression.td.td = enum_sof(modelling_pb2.TradingDays, r["regression"]["td"]["td"])
    p.regression.td.lp = enum_of(
        modelling_pb2.LengthOfPeriod, r["regression"]["td"]["lp"], "LP"
    )
    p.regression.td.holidays = r["regression"]["td"]["holidays"]
    for u in r["regression"]["td"]["users"]:
        p.regression.td.users.append(u)
    p.regression.td.w = r["regression"]["td"]["w"]
    p.regression.td.test = enum_of(
        tramoseats_pb2.TradingDaysTest, r["regression"]["td"]["test"], "TD"
    )
    td_r = r["regression"]["td"]
    p.regression.td.auto = enum_of(
        tramoseats_pb2.AutomaticTradingDays, td_r["auto"], "TD"
    )
    p.regression.td.auto_adjust = td_r["autoadjust"]
    p.regression.td.ptest = td_r["ptest"]
    for c in r2p_parameters(td_r["tdcoefficients"]):
        p.regression.td.tdcoefficients.append(c)
    p.regression.td.lpcoefficient.CopyFrom(r2p_parameter(td_r["lpcoefficient"]))

    easter_r = r["regression"]["easter"]
    p.regression.easter.type = enum_of(
        tramoseats_pb2.EasterType, easter_r["type"], "EASTER"
    )
    p.regression.easter.duration = easter_r["duration"]
    p.regression.easter.julian = easter_r["julian"]
    p.regression.easter.test = easter_r["test"]
    p.regression.easter.coefficient.CopyFrom(r2p_parameter(easter_r["coefficient"]))

    p.estimate.span.CopyFrom(r2p_span(r["estimate"]["span"]))
    p.estimate.ml = r["estimate"]["ml"]
    p.estimate.tol = r["estimate"]["tol"]
    p.estimate.ubp = r["estimate"]["ubp"]

    return p


def _p2r_spec_seats(spec) -> dict:
    return {
        "xl": spec.xl_boundary,
        "approximation": enum_extract(tramoseats_pb2.SeatsApproximation, spec.approximation),
        "epsphi": spec.seastolerance,
        "rmod": spec.trend_boundary,
        "sbound": spec.seas_boundary,
        "sboundatpi": spec.seas_boundary_at_pi,
        "bias": spec.bias_correction,
        "nfcasts": spec.nfcasts,
        "nbcasts": spec.nbcasts,
        "algorithm": enum_extract(tramoseats_pb2.SeatsAlgorithm, spec.algorithm),
    }


def _r2p_spec_seats(spec: dict):
    p = tramoseats_pb2.DecompositionSpec()
    p.xl_boundary = spec["xl"]
    p.approximation = enum_of(
        tramoseats_pb2.SeatsApproximation, spec["approximation"], "SEATS"
    )
    p.seastolerance = spec["epsphi"]
    p.trend_boundary = spec["rmod"]
    p.seas_boundary = spec["sbound"]
    p.seas_boundary_at_pi = spec["sboundatpi"]
    p.bias_correction = spec["bias"]
    p.nfcasts = spec["nfcasts"]
    p.nbcasts = spec["nbcasts"]
    p.algorithm = enum_of(tramoseats_pb2.SeatsAlgorithm, spec["algorithm"], "SEATS")
    return p


def _p2r_spec_tramoseats(pspec) -> dict:
    return {
        "tramo": _p2r_spec_tramo(pspec.tramo),
        "seats": _p2r_spec_seats(pspec.seats),
        "benchmarking": p2r_spec_benchmarking(pspec.benchmarking),
    }


def _r2p_spec_tramoseats(r: dict):
    p = tramoseats_pb2.Spec()
    p.tramo.CopyFrom(_r2p_spec_tramo(r["tramo"]))
    p.seats.CopyFrom(_r2p_spec_seats(r["seats"]))
    p.benchmarking.CopyFrom(r2p_spec_benchmarking(r["benchmarking"]))
    return p


def _tramoseats_output(jq) -> dict:
    import jpype

    TramoSeats = jpype.JClass("jdplus.tramoseats.base.r.TramoSeats")
    b = bytes(TramoSeats.toBuffer(jq))
    p = tramoseats_pb2.TramoSeatsOutput()
    p.ParseFromString(b)
    return {
        "result": _p2r_tramoseats_rslts(p.result),
        "estimation_spec": _p2r_spec_tramoseats(p.estimation_spec),
        "result_spec": _p2r_spec_tramoseats(p.result_spec),
    }


def _tramo_output(jq) -> dict:
    import jpype

    Tramo = jpype.JClass("jdplus.tramoseats.base.r.Tramo")
    b = bytes(Tramo.toBuffer(jq))
    p = tramoseats_pb2.TramoOutput()
    p.ParseFromString(b)
    return {
        "result": p2r_regarima_rslts(p.result),
        "estimation_spec": _p2r_spec_tramo(p.estimation_spec),
        "result_spec": _p2r_spec_tramo(p.result_spec),
    }


def _tramoseats_rslts(jrslt) -> dict:
    import jpype

    TramoSeats = jpype.JClass("jdplus.tramoseats.base.r.TramoSeats")
    b = bytes(TramoSeats.toBuffer(jrslt))
    p = tramoseats_pb2.TramoSeatsResults()
    p.ParseFromString(b)
    return _p2r_tramoseats_rslts(p)


def _tramo_fast_rslts(jrslt) -> dict:
    import jpype

    Tramo = jpype.JClass("jdplus.tramoseats.base.r.Tramo")
    b = bytes(Tramo.toBuffer(jrslt))
    from pydemetra._proto import regarima_pb2

    p = regarima_pb2.RegArimaModel()
    p.ParseFromString(b)
    return p2r_regarima_rslts(p)


def _p2r_tramoseats_rslts(p) -> dict:
    return {
        "preprocessing": p2r_regarima_rslts(p.preprocessing),
        "decomposition": _p2r_seats_rslts(p.decomposition),
        "final": p2r_sa_decomposition(p.final),
        "diagnostics": p2r_sa_diagnostics(p.diagnostics_sa),
    }


def _p2r_seats_rslts(p) -> dict:
    return {
        "seatsmodel": p2r_arima(p.seats_arima),
        "canonicaldecomposition": p2r_ucarima(p.canonical_decomposition),
        "stochastics": p2r_sa_decomposition(p.stochastics, True),
    }
