from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pandas as pd

from pydemetra._converters import (
    p2r_ivs as _p2r_ivs,
)
from pydemetra._converters import (
    p2r_outliers as _p2r_outliers,
)
from pydemetra._converters import (
    p2r_parameter,
    p2r_parameters,
    p2r_regarima_rslts,
    p2r_sa_diagnostics,
    p2r_span,
    p2r_spec_benchmarking,
    p2r_spec_sarima,
    p2r_tsdata,
    p2r_uservars,
    r2jd_tsdata,
    r2p_parameter,
    r2p_parameters,
    r2p_span,
    r2p_spec_benchmarking,
    r2p_spec_sarima,
)
from pydemetra._converters import (
    p2r_ramps as _p2r_ramps,
)
from pydemetra._converters import (
    r2p_ivs as _r2p_ivs,
)
from pydemetra._converters import (
    r2p_outliers as _r2p_outliers,
)
from pydemetra._converters import (
    r2p_ramps as _r2p_ramps,
)
from pydemetra._converters import (
    r2p_uservars as _r2p_uservars,
)
from pydemetra._java import _ensure_jvm
from pydemetra._protobuf import enum_extract, enum_of, enum_sextract, enum_sof

if TYPE_CHECKING:
    modelling_pb2: Any
    x13_pb2: Any
else:
    from pydemetra._proto import modelling_pb2, x13_pb2

# ---------------------------------------------------------------------------
# Predefined spec names
# ---------------------------------------------------------------------------

_X13_SPEC_NAMES = ("rsa0", "rsa1", "rsa2c", "rsa3", "rsa4", "rsa5c")
_REGARIMA_SPEC_NAMES = ("rg0", "rg1", "rg2c", "rg3", "rg4", "rg5c")

# ---------------------------------------------------------------------------
# Specification creation
# ---------------------------------------------------------------------------


def x13_spec(name: str = "rsa4") -> dict:
    """Create a default X-13 specification.

    Args:
        name (str): Predefined specification name. One of
            ``"rsa0"``, ``"rsa1"``, ``"rsa2c"``, ``"rsa3"``, ``"rsa4"``, ``"rsa5c"``.

    Returns:
        dict: A specification dict with ``regarima``, ``x11``, and ``benchmarking`` keys.

    Raises:
        ValueError: If *name* is not a recognised specification name.
    """
    _ensure_jvm()
    import jpype

    name = name.lower().replace("g", "sa")
    if name not in _X13_SPEC_NAMES:
        raise ValueError(f"Unknown X-13 spec name: {name!r}. Choose from {_X13_SPEC_NAMES}")

    X13Spec = jpype.JClass("jdplus.x13.base.api.x13.X13Spec")
    jspec = X13Spec.fromString(name)
    return _jd2r_spec_x13(jspec)


def regarima_spec(name: str = "rg4") -> dict:
    """Create a default RegARIMA specification (X-13 flavour).

    Args:
        name (str): Predefined specification name. One of
            ``"rg0"``, ``"rg1"``, ``"rg2c"``, ``"rg3"``, ``"rg4"``, ``"rg5c"``.

    Returns:
        dict: A specification dict with ``basic``, ``transform``, ``outlier``, ``arima``,
            ``automodel``, ``regression``, and ``estimate`` keys.

    Raises:
        ValueError: If *name* is not a recognised specification name.
    """
    _ensure_jvm()
    import jpype

    name = name.lower().replace("sa", "g")
    if name not in _REGARIMA_SPEC_NAMES:
        msg = f"Unknown RegARIMA spec: {name!r}. Choose from {_REGARIMA_SPEC_NAMES}"
        raise ValueError(msg)

    RegArimaSpec = jpype.JClass("jdplus.x13.base.api.regarima.RegArimaSpec")
    jspec = RegArimaSpec.fromString(name)
    return _jd2r_spec_regarima(jspec)


def x11_spec() -> dict:
    """Create the default X-11 decomposition specification.

    Returns:
        dict: A specification dict with X-11 decomposition parameters.
    """
    _ensure_jvm()
    import jpype

    X11Spec = jpype.JClass("jdplus.x13.base.api.x11.X11Spec")
    jspec = X11Spec.DEFAULT
    return _jd2r_spec_x11(jspec)


# ---------------------------------------------------------------------------
# set_x11
# ---------------------------------------------------------------------------

_DECOMPOSITION_MODES = ("UNKNOWN", "ADDITIVE", "MULTIPLICATIVE", "LOGADDITIVE", "PSEUDOADDITIVE")
_SEASONAL_FILTERS = ("MSR", "STABLE", "X11DEFAULT", "S3X1", "S3X3", "S3X5", "S3X9", "S3X15")
_CALENDAR_SIGMAS = ("NONE", "SIGNIF", "ALL", "SELECT")


def set_x11(
    spec: dict,
    *,
    mode: str | None = None,
    seasonal_comp: bool | None = None,
    seasonal_filter: str | list[str] | None = None,
    henderson_filter: int | None = None,
    lsigma: float | None = None,
    usigma: float | None = None,
    fcasts: int | None = None,
    bcasts: int | None = None,
    calendar_sigma: str | None = None,
    sigma_vector: list[int] | None = None,
    exclude_forecast: bool | None = None,
    bias: str | None = None,
) -> dict:
    """Modify the X-11 decomposition parameters in a specification.

    Works on both X-11 specs and full X-13 specs (modifying the ``x11`` sub-dict).

    Args:
        spec (dict): An X-11 or X-13 specification dict.
        mode (str | None): Decomposition mode.
        seasonal_comp (bool | None): Whether to compute a seasonal component.
        seasonal_filter (str | list[str] | None): Seasonal filter(s). A single string
            or list of strings.
        henderson_filter (int | None): Henderson filter length (odd 3-101, or 0 for auto).
        lsigma (float | None): Lower sigma threshold for extreme value detection.
        usigma (float | None): Upper sigma threshold for extreme value detection.
        fcasts (int | None): Number of forecasts (negative = years).
        bcasts (int | None): Number of backcasts (negative = years).
        calendar_sigma (str | None): Calendar sigma option.
        sigma_vector (list[int] | None): Sigma group assignments (values 1 or 2).
        exclude_forecast (bool | None): Exclude forecasts from extreme value detection.
        bias (str | None): Bias correction (``"LEGACY"``).

    Returns:
        dict: A modified copy of the specification.
    """
    spec = copy.deepcopy(spec)

    if "x11" in spec:
        spec["x11"] = _set_x11_inner(
            spec["x11"],
            mode=mode,
            seasonal_comp=seasonal_comp,
            seasonal_filter=seasonal_filter,
            henderson_filter=henderson_filter,
            lsigma=lsigma,
            usigma=usigma,
            fcasts=fcasts,
            bcasts=bcasts,
            calendar_sigma=calendar_sigma,
            sigma_vector=sigma_vector,
            exclude_forecast=exclude_forecast,
            bias=bias,
        )
        return spec

    return _set_x11_inner(
        spec,
        mode=mode,
        seasonal_comp=seasonal_comp,
        seasonal_filter=seasonal_filter,
        henderson_filter=henderson_filter,
        lsigma=lsigma,
        usigma=usigma,
        fcasts=fcasts,
        bcasts=bcasts,
        calendar_sigma=calendar_sigma,
        sigma_vector=sigma_vector,
        exclude_forecast=exclude_forecast,
        bias=bias,
    )


def _set_x11_inner(
    x: dict,
    mode: str | None = None,
    seasonal_comp: bool | None = None,
    seasonal_filter: str | list[str] | None = None,
    henderson_filter: int | None = None,
    lsigma: float | None = None,
    usigma: float | None = None,
    fcasts: int | None = None,
    bcasts: int | None = None,
    calendar_sigma: str | None = None,
    sigma_vector: list[int] | None = None,
    exclude_forecast: bool | None = None,
    bias: str | None = None,
) -> dict:
    x = copy.deepcopy(x)

    if mode is not None:
        mode_upper = mode.upper()
        if mode_upper == "UNDEFINED":
            mode_upper = "UNKNOWN"
        if mode_upper not in _DECOMPOSITION_MODES:
            raise ValueError(f"Invalid mode: {mode!r}. Choose from {_DECOMPOSITION_MODES}")
        x["mode"] = mode_upper

    if seasonal_comp is not None:
        x["seasonal"] = bool(seasonal_comp)

    if seasonal_filter is not None:
        if isinstance(seasonal_filter, str):
            seasonal_filter = [seasonal_filter]
        validated = []
        for sf in seasonal_filter:
            sf_upper = sf.upper()
            if sf_upper not in _SEASONAL_FILTERS:
                msg = f"Invalid seasonal filter: {sf!r}. Choose from {_SEASONAL_FILTERS}"
                raise ValueError(msg)
            validated.append(f"FILTER_{sf_upper}")
        x["sfilters"] = validated

    if henderson_filter is not None:
        if henderson_filter != 0 and henderson_filter % 2 == 0:
            raise ValueError("henderson_filter must be an odd number or 0")
        x["henderson"] = henderson_filter

    if lsigma is not None:
        x["lsig"] = lsigma
    if usigma is not None:
        x["usig"] = usigma
    if fcasts is not None:
        x["nfcasts"] = fcasts
    if bcasts is not None:
        x["nbcasts"] = bcasts

    if calendar_sigma is not None:
        cs_upper = calendar_sigma.upper()
        if cs_upper not in _CALENDAR_SIGMAS:
            raise ValueError(f"Invalid calendar_sigma: {calendar_sigma!r}")
        x["sigma"] = cs_upper

    if sigma_vector is not None:
        if not all(v in (1, 2) for v in sigma_vector):
            raise ValueError("sigma_vector values must be 1 or 2")
        x["sigma"] = "SELECT"
        x["vsigmas"] = [int(v) for v in sigma_vector]

    if exclude_forecast is not None:
        x["excludefcasts"] = bool(exclude_forecast)

    if bias is not None:
        x["bias"] = bias.upper()

    return x


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------


def x13(
    ts: pd.Series,
    spec: str | dict = "rsa4",
    context: Any | None = None,
    userdefined: list[str] | None = None,
) -> dict | None:
    """Run full X-13ARIMA-SEATS seasonal adjustment.

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
    X13 = jpype.JClass("jdplus.x13.base.r.X13")

    if isinstance(spec, str):
        spec_name = spec.lower().replace("g", "sa")
        if spec_name not in _X13_SPEC_NAMES:
            raise ValueError(f"Unknown spec: {spec!r}")
        jrslt = X13.fullProcess(jts, spec_name)
    else:
        jspec = _r2jd_spec_x13(spec)
        if context is None:
            ModellingContext = jpype.JClass(
                "jdplus.toolkit.base.api.timeseries.regression.ModellingContext"
            )
            jcontext = jpype.JObject(None, ModellingContext)
        else:
            from pydemetra.context import _r2jd_modellingcontext

            jcontext = _r2jd_modellingcontext(context)
        jrslt = X13.fullProcess(jts, jspec, jcontext)

    if jrslt is None:
        return None

    return _x13_output(jrslt)


def x13_fast(
    ts: pd.Series,
    spec: str | dict = "rsa4",
    context: Any | None = None,
    userdefined: list[str] | None = None,
) -> dict | None:
    """Run X-13 seasonal adjustment (results only, faster).

    Args:
        ts (pd.Series): Univariate time series with PeriodIndex.
        spec (str | dict): Specification name or dict.
        context (Any | None): Modelling context with external regressors.
        userdefined (list[str] | None): Additional output variable names.

    Returns:
        dict | None: Dict with X-13 results, or ``None`` if processing failed.

    Raises:
        ValueError: If *spec* is a string that is not a recognised specification name.
    """
    _ensure_jvm()
    import jpype

    jts = r2jd_tsdata(ts)
    X13 = jpype.JClass("jdplus.x13.base.r.X13")

    if isinstance(spec, str):
        spec_name = spec.lower().replace("g", "sa")
        if spec_name not in _X13_SPEC_NAMES:
            raise ValueError(f"Unknown spec: {spec!r}")
        jrslt = X13.process(jts, spec_name)
    else:
        jspec = _r2jd_spec_x13(spec)
        if context is None:
            ModellingContext = jpype.JClass(
                "jdplus.toolkit.base.api.timeseries.regression.ModellingContext"
            )
            jcontext = jpype.JObject(None, ModellingContext)
        else:
            from pydemetra.context import _r2jd_modellingcontext

            jcontext = _r2jd_modellingcontext(context)
        jrslt = X13.process(jts, jspec, jcontext)

    if jrslt is None:
        return None

    return _x13_rslts(jrslt)


def x11(
    ts: pd.Series,
    spec: dict | None = None,
    userdefined: list[str] | None = None,
) -> dict | None:
    """Run pure X-11 decomposition (no RegARIMA pre-processing).

    Args:
        ts (pd.Series): Univariate time series with PeriodIndex.
        spec (dict | None): X-11 specification dict. Defaults to ``x11_spec()``.
        userdefined (list[str] | None): Additional output variable names.

    Returns:
        dict | None: Dict with X-11 D-series components, or ``None`` if processing failed.
    """
    _ensure_jvm()
    import jpype

    if spec is None:
        spec = x11_spec()

    jts = r2jd_tsdata(ts)
    jspec = _r2jd_spec_x11(spec)
    X11 = jpype.JClass("jdplus.x13.base.r.X11")
    jrslt = X11.process(jts, jspec)

    if jrslt is None:
        return None

    return _x11_rslts(jrslt)


def x13_regarima(
    ts: pd.Series,
    spec: str | dict = "rg4",
    context: Any | None = None,
    userdefined: list[str] | None = None,
) -> dict | None:
    """Run full RegARIMA model (X-13 flavour).

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
    RegArima = jpype.JClass("jdplus.x13.base.r.RegArima")

    if isinstance(spec, str):
        spec_name = spec.lower().replace("sa", "g")
        if spec_name not in _REGARIMA_SPEC_NAMES:
            raise ValueError(f"Unknown spec: {spec!r}")
        jrslt = RegArima.fullProcess(jts, spec_name)
    else:
        jspec = _r2jd_spec_regarima(spec)
        if context is None:
            ModellingContext = jpype.JClass(
                "jdplus.toolkit.base.api.timeseries.regression.ModellingContext"
            )
            jcontext = jpype.JObject(None, ModellingContext)
        else:
            from pydemetra.context import _r2jd_modellingcontext

            jcontext = _r2jd_modellingcontext(context)
        jrslt = RegArima.fullProcess(jts, jspec, jcontext)

    if jrslt is None:
        return None

    return _regarima_output(jrslt)


# ---------------------------------------------------------------------------
# Dictionary functions
# ---------------------------------------------------------------------------


def x13_dictionary() -> list[str]:
    """Return all available X-13 output variable names."""
    _ensure_jvm()
    import jpype

    X13 = jpype.JClass("jdplus.x13.base.r.X13")
    return list(X13.dictionary())


# ---------------------------------------------------------------------------
# Java <-> Protobuf <-> Python: Spec converters
# ---------------------------------------------------------------------------


def _jd2r_spec_x11(jspec) -> dict:
    import jpype

    X11 = jpype.JClass("jdplus.x13.base.r.X11")
    b = bytes(X11.toBuffer(jspec))
    p = x13_pb2.X11Spec()
    p.ParseFromString(b)
    return _p2r_spec_x11(p)


def _r2jd_spec_x11(spec: dict):
    import jpype

    p = _r2p_spec_x11(spec)
    b = p.SerializeToString()
    X11 = jpype.JClass("jdplus.x13.base.r.X11")
    return X11.of(b)


def _jd2r_spec_regarima(jspec) -> dict:
    import jpype

    RegArima = jpype.JClass("jdplus.x13.base.r.RegArima")
    b = bytes(RegArima.toBuffer(jspec))
    p = x13_pb2.RegArimaSpec()
    p.ParseFromString(b)
    return _p2r_spec_regarima(p)


def _r2jd_spec_regarima(spec: dict):
    import jpype

    p = _r2p_spec_regarima(spec)
    b = p.SerializeToString()
    RegArima = jpype.JClass("jdplus.x13.base.r.RegArima")
    return RegArima.specOf(b)


def _jd2r_spec_x13(jspec) -> dict:
    import jpype

    X13 = jpype.JClass("jdplus.x13.base.r.X13")
    b = bytes(X13.toBuffer(jspec))
    p = x13_pb2.Spec()
    p.ParseFromString(b)
    return _p2r_spec_x13(p)


def _r2jd_spec_x13(spec: dict):
    import jpype

    p = _r2p_spec_x13(spec)
    b = p.SerializeToString()
    X13 = jpype.JClass("jdplus.x13.base.r.X13")
    return X13.specOf(b)


# ---------------------------------------------------------------------------
# Protobuf <-> Python: Spec converters
# ---------------------------------------------------------------------------


def _p2r_spec_x11(p) -> dict:
    return {
        "mode": enum_extract(x13_pb2.DecompositionMode, p.mode),
        "seasonal": p.seasonal,
        "henderson": p.henderson,
        "sfilters": [enum_extract(x13_pb2.SeasonalFilter, z) for z in p.sfilters],
        "lsig": p.lsig,
        "usig": p.usig,
        "nfcasts": p.nfcasts,
        "nbcasts": p.nbcasts,
        "sigma": enum_extract(x13_pb2.CalendarSigma, p.sigma),
        "vsigmas": list(p.vsigmas),
        "excludefcasts": p.exclude_fcasts,
        "bias": enum_extract(x13_pb2.BiasCorrection, p.bias),
    }


def _r2p_spec_x11(r: dict):
    p = x13_pb2.X11Spec()
    p.mode = enum_of(x13_pb2.DecompositionMode, r["mode"], "MODE")
    p.seasonal = r["seasonal"]
    p.henderson = r["henderson"]
    for sf in r["sfilters"]:
        p.sfilters.append(enum_of(x13_pb2.SeasonalFilter, sf, "SEASONAL"))
    p.lsig = r["lsig"]
    p.usig = r["usig"]
    p.nfcasts = r["nfcasts"]
    p.nbcasts = r["nbcasts"]
    p.sigma = enum_of(x13_pb2.CalendarSigma, r["sigma"], "SIGMA")
    for v in r["vsigmas"]:
        p.vsigmas.append(v)
    p.exclude_fcasts = r["excludefcasts"]
    p.bias = enum_of(x13_pb2.BiasCorrection, r["bias"], "BIAS")
    return p


def _p2r_spec_regarima(pspec) -> dict:
    basic = {
        "span": p2r_span(pspec.basic.span),
        "preprocessing": pspec.basic.preprocessing,
        "preliminaryCheck": pspec.basic.preliminary_check,
    }
    transform = {
        "fn": enum_extract(modelling_pb2.Transformation, pspec.transform.transformation),
        "adjust": enum_extract(modelling_pb2.LengthOfPeriod, pspec.transform.adjust),
        "aicdiff": pspec.transform.aicdiff,
        "outliers": pspec.transform.outliers_correction,
    }
    automodel = {
        "enabled": pspec.automodel.enabled,
        "ljungbox": pspec.automodel.ljungbox,
        "tsig": pspec.automodel.tsig,
        "predcv": pspec.automodel.predcv,
        "ubfinal": pspec.automodel.ubfinal,
        "ub1": pspec.automodel.ub1,
        "ub2": pspec.automodel.ub2,
        "cancel": pspec.automodel.cancel,
        "fct": pspec.automodel.fct,
        "acceptdef": pspec.automodel.acceptdef,
        "mixed": pspec.automodel.mixed,
        "balanced": pspec.automodel.balanced,
    }
    arima = p2r_spec_sarima(pspec.arima)
    outlier = {
        "outliers": [{"type": z.code, "va": z.va} for z in pspec.outlier.outliers],
        "span": p2r_span(pspec.outlier.span),
        "defva": pspec.outlier.defva,
        "method": enum_extract(x13_pb2.OutlierMethod, pspec.outlier.method),
        "monthlytcrate": pspec.outlier.monthly_tc_rate,
        "maxiter": pspec.outlier.maxiter,
        "lsrun": pspec.outlier.lsrun,
    }
    td = {
        "td": enum_sextract(modelling_pb2.TradingDays, pspec.regression.td.td),
        "lp": enum_extract(modelling_pb2.LengthOfPeriod, pspec.regression.td.lp),
        "holidays": pspec.regression.td.holidays,
        "users": list(pspec.regression.td.users),
        "w": pspec.regression.td.w,
        "test": enum_extract(x13_pb2.RegressionTest, pspec.regression.td.test),
        "auto": enum_extract(x13_pb2.AutomaticTradingDays, pspec.regression.td.auto),
        "autoadjust": pspec.regression.td.auto_adjust,
        "tdcoefficients": p2r_parameters(list(pspec.regression.td.tdcoefficients)),
        "lpcoefficient": p2r_parameter(pspec.regression.td.lpcoefficient),
        "ptest1": pspec.regression.td.ptest1,
        "ptest2": pspec.regression.td.ptest2,
    }
    easter = {
        "type": enum_extract(x13_pb2.EasterType, pspec.regression.easter.type),
        "duration": pspec.regression.easter.duration,
        "test": enum_extract(x13_pb2.RegressionTest, pspec.regression.easter.test),
        "coefficient": p2r_parameter(pspec.regression.easter.coefficient),
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
        "tol": pspec.estimate.tol,
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


def _r2p_spec_regarima(r: dict):
    p = x13_pb2.RegArimaSpec()

    p.basic.preliminary_check = r["basic"]["preliminaryCheck"]
    p.basic.preprocessing = r["basic"]["preprocessing"]
    p.basic.span.CopyFrom(r2p_span(r["basic"]["span"]))

    p.transform.transformation = enum_of(modelling_pb2.Transformation, r["transform"]["fn"], "FN")
    p.transform.adjust = enum_of(modelling_pb2.LengthOfPeriod, r["transform"]["adjust"], "LP")
    p.transform.aicdiff = r["transform"]["aicdiff"]
    p.transform.outliers_correction = r["transform"]["outliers"]

    for z in r["outlier"]["outliers"]:
        ot = p.outlier.outliers.add()
        ot.code = z["type"]
        ot.va = z["va"]
    p.outlier.span.CopyFrom(r2p_span(r["outlier"]["span"]))
    p.outlier.defva = r["outlier"]["defva"]
    p.outlier.method = enum_of(x13_pb2.OutlierMethod, r["outlier"]["method"], "OUTLIER")
    p.outlier.monthly_tc_rate = r["outlier"]["monthlytcrate"]
    p.outlier.maxiter = r["outlier"]["maxiter"]
    p.outlier.lsrun = r["outlier"]["lsrun"]

    p.automodel.enabled = r["automodel"]["enabled"]
    p.automodel.ljungbox = r["automodel"]["ljungbox"]
    p.automodel.tsig = r["automodel"]["tsig"]
    p.automodel.predcv = r["automodel"]["predcv"]
    p.automodel.ubfinal = r["automodel"]["ubfinal"]
    p.automodel.ub1 = r["automodel"]["ub1"]
    p.automodel.ub2 = r["automodel"]["ub2"]
    p.automodel.cancel = r["automodel"]["cancel"]
    p.automodel.fct = r["automodel"]["fct"]
    p.automodel.acceptdef = r["automodel"]["acceptdef"]
    p.automodel.mixed = r["automodel"]["mixed"]
    p.automodel.balanced = r["automodel"]["balanced"]

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
    p.regression.td.lp = enum_of(modelling_pb2.LengthOfPeriod, r["regression"]["td"]["lp"], "LP")
    p.regression.td.holidays = r["regression"]["td"]["holidays"]
    for u in r["regression"]["td"]["users"]:
        p.regression.td.users.append(u)
    p.regression.td.w = r["regression"]["td"]["w"]
    p.regression.td.test = enum_of(x13_pb2.RegressionTest, r["regression"]["td"]["test"], "TEST")
    td_r = r["regression"]["td"]
    p.regression.td.auto = enum_of(x13_pb2.AutomaticTradingDays, td_r["auto"], "TD")
    p.regression.td.auto_adjust = td_r["autoadjust"]
    for c in r2p_parameters(td_r["tdcoefficients"]):
        p.regression.td.tdcoefficients.append(c)
    p.regression.td.lpcoefficient.CopyFrom(r2p_parameter(td_r["lpcoefficient"]))
    p.regression.td.ptest1 = td_r["ptest1"]
    p.regression.td.ptest2 = td_r["ptest2"]

    easter_r = r["regression"]["easter"]
    p.regression.easter.type = enum_of(x13_pb2.EasterType, easter_r["type"], "EASTER")
    p.regression.easter.duration = easter_r["duration"]
    p.regression.easter.test = enum_of(x13_pb2.RegressionTest, easter_r["test"], "TEST")
    p.regression.easter.coefficient.CopyFrom(r2p_parameter(easter_r["coefficient"]))

    p.estimate.span.CopyFrom(r2p_span(r["estimate"]["span"]))
    p.estimate.tol = r["estimate"]["tol"]

    return p


def _p2r_spec_x13(pspec) -> dict:
    return {
        "regarima": _p2r_spec_regarima(pspec.regarima),
        "x11": _p2r_spec_x11(pspec.x11),
        "benchmarking": p2r_spec_benchmarking(pspec.benchmarking),
    }


def _r2p_spec_x13(r: dict):
    p = x13_pb2.Spec()
    p.regarima.CopyFrom(_r2p_spec_regarima(r["regarima"]))
    p.x11.CopyFrom(_r2p_spec_x11(r["x11"]))
    p.benchmarking.CopyFrom(r2p_spec_benchmarking(r["benchmarking"]))
    return p


# ---------------------------------------------------------------------------
# Results extraction (Java -> Protobuf -> Python)
# ---------------------------------------------------------------------------


def _x13_output(jq) -> dict:
    import jpype

    X13 = jpype.JClass("jdplus.x13.base.r.X13")
    b = bytes(X13.toBuffer(jq))
    p = x13_pb2.X13Output()
    p.ParseFromString(b)
    return {
        "result": _p2r_x13_rslts(p.result),
        "estimation_spec": _p2r_spec_x13(p.estimation_spec),
        "result_spec": _p2r_spec_x13(p.result_spec),
    }


def _regarima_output(jq) -> dict:
    import jpype

    RegArima = jpype.JClass("jdplus.x13.base.r.RegArima")
    b = bytes(RegArima.toBuffer(jq))
    p = x13_pb2.RegArimaOutput()
    p.ParseFromString(b)
    return {
        "result": p2r_regarima_rslts(p.result),
        "estimation_spec": _p2r_spec_regarima(p.estimation_spec),
        "result_spec": _p2r_spec_regarima(p.result_spec),
    }


def _x13_rslts(jrslt) -> dict:
    import jpype

    X13 = jpype.JClass("jdplus.x13.base.r.X13")
    b = bytes(X13.toBuffer(jrslt))
    p = x13_pb2.X13Results()
    p.ParseFromString(b)
    return _p2r_x13_rslts(p)


def _x11_rslts(jrslt) -> dict:
    import jpype

    X11 = jpype.JClass("jdplus.x13.base.r.X11")
    b = bytes(X11.toBuffer(jrslt))
    p = x13_pb2.X11Results()
    p.ParseFromString(b)
    return _p2r_x11_rslts(p)


def _p2r_x13_rslts(p) -> dict:
    mstats_p = p.diagnostics_x13.mstatistics
    mstats = {
        "m1": mstats_p.m1,
        "m2": mstats_p.m2,
        "m3": mstats_p.m3,
        "m4": mstats_p.m4,
        "m5": mstats_p.m5,
        "m6": mstats_p.m6,
        "m7": mstats_p.m7,
        "m8": mstats_p.m8,
        "m9": mstats_p.m9,
        "m10": mstats_p.m10,
        "m11": mstats_p.m11,
        "q": mstats_p.q,
        "qm2": mstats_p.qm2,
    }
    return {
        "preprocessing": p2r_regarima_rslts(p.preprocessing),
        "preadjust": _p2r_x13_preadjust(p.preadjustment),
        "decomposition": _p2r_x11_rslts(p.decomposition),
        "final": _p2r_x13_final(p.final),
        "mstats": mstats,
        "diagnostics": p2r_sa_diagnostics(p.diagnostics_sa),
    }


def _p2r_x11_rslts(p) -> dict:
    return {
        "d1": p2r_tsdata(p.d1),
        "d2": p2r_tsdata(p.d2),
        "d4": p2r_tsdata(p.d4),
        "d5": p2r_tsdata(p.d5),
        "d6": p2r_tsdata(p.d6),
        "d7": p2r_tsdata(p.d7),
        "d8": p2r_tsdata(p.d8),
        "d9": p2r_tsdata(p.d9),
        "d10": p2r_tsdata(p.d10),
        "d11": p2r_tsdata(p.d11),
        "d12": p2r_tsdata(p.d12),
        "d13": p2r_tsdata(p.d13),
        "final_seasonal": list(p.final_seasonal_filters),
        "final_henderson": p.final_henderson_filter,
    }


def _p2r_x13_final(p) -> dict:
    return {
        "d11final": p2r_tsdata(p.d11final),
        "d12final": p2r_tsdata(p.d12final),
        "d13final": p2r_tsdata(p.d13final),
        "d16": p2r_tsdata(p.d16),
        "d18": p2r_tsdata(p.d18),
        "d11a": p2r_tsdata(p.d11a),
        "d12a": p2r_tsdata(p.d12a),
        "d16a": p2r_tsdata(p.d16a),
        "d18a": p2r_tsdata(p.d18a),
        "e1": p2r_tsdata(p.e1),
        "e2": p2r_tsdata(p.e2),
        "e3": p2r_tsdata(p.e3),
        "e11": p2r_tsdata(p.e11),
    }


def _p2r_x13_preadjust(p) -> dict:
    return {
        "a1": p2r_tsdata(p.a1),
        "a1a": p2r_tsdata(p.a1a),
        "a1b": p2r_tsdata(p.a1b),
        "a6": p2r_tsdata(p.a6),
        "a7": p2r_tsdata(p.a7),
        "a8": p2r_tsdata(p.a8),
        "a9": p2r_tsdata(p.a9),
    }
