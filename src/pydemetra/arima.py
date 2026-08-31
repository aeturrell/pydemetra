from __future__ import annotations

from typing import Any

import numpy as np

from pydemetra._converters import (
    jd2r_matrix,
    p2r_arima,
    p2r_sarima,
    p2r_ucarima,
    r2jd_matrix,
)
from pydemetra._java import _ensure_jvm
from pydemetra._models import ArimaModel, SarimaModel, UcarimaModel


def sarima_model(
    name: str = "sarima",
    period: int = 1,
    phi: list[float] | None = None,
    d: int = 0,
    theta: list[float] | None = None,
    bphi: list[float] | None = None,
    bd: int = 0,
    btheta: list[float] | None = None,
) -> SarimaModel:
    """Create a seasonal ARIMA model (Box-Jenkins).

    Args:
        name (str): Name of the model.
        period (int): Period of the model.
        phi (list[float] | None): Coefficients of the regular AR polynomial. True signs.
        d (int): Regular differencing order.
        theta (list[float] | None): Coefficients of the regular MA polynomial. True signs.
        bphi (list[float] | None): Coefficients of the seasonal AR polynomial. True signs.
        bd (int): Seasonal differencing order.
        btheta (list[float] | None): Coefficients of the seasonal MA polynomial. True signs.

    Returns:
        SarimaModel: A SarimaModel dataclass.
    """
    return SarimaModel(
        name=name,
        period=period,
        phi=phi or [],
        d=d,
        theta=theta or [],
        bphi=bphi or [],
        bd=bd,
        btheta=btheta or [],
    )


def _r2jd_sarima(model: SarimaModel) -> object:
    """Convert a SarimaModel to a Java SarimaModel."""
    _ensure_jvm()
    import jpype

    SarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.SarimaModels")
    return SarimaModels.of(
        int(model.period),
        np.array(model.phi, dtype=np.float64),
        int(model.d),
        np.array(model.theta, dtype=np.float64),
        np.array(model.bphi, dtype=np.float64),
        int(model.bd),
        np.array(model.btheta, dtype=np.float64),
    )


def _jd2r_sarima(jsarima: object) -> SarimaModel:
    """Convert a Java SarimaModel to a Python SarimaModel via protobuf."""
    import jpype

    SarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.SarimaModels")
    buf = bytes(SarimaModels.toBuffer(jsarima))
    from pydemetra._proto import modelling_pb2

    msg = modelling_pb2.SarimaModel()
    msg.ParseFromString(buf)
    return p2r_sarima(msg)


def _r2jd_arima(model: ArimaModel) -> object:
    """Convert an ArimaModel to a Java ArimaModel."""
    _ensure_jvm()
    import jpype

    ArimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.ArimaModels")
    return ArimaModels.of(
        np.array(model.ar, dtype=np.float64),
        np.array(model.delta, dtype=np.float64),
        np.array(model.ma, dtype=np.float64),
        float(model.var),
        False,
    )


def _jd2r_arima(jarima: object) -> ArimaModel:
    """Convert a Java ArimaModel to a Python ArimaModel via protobuf."""
    import jpype

    ArimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.ArimaModels")
    buf = bytes(ArimaModels.toBuffer(jarima))
    from pydemetra._proto import modelling_pb2

    msg = modelling_pb2.ArimaModel()
    msg.ParseFromString(buf)
    return p2r_arima(msg)


def _r2jd_ucarima(ucm: UcarimaModel) -> object:
    """Convert a UcarimaModel to a Java UcarimaModel."""
    _ensure_jvm()
    import jpype

    UcarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.UcarimaModels")
    assert ucm.model is not None
    jmodel = _r2jd_arima(ucm.model)
    ArimaModelClass = jpype.JClass("jdplus.toolkit.base.core.arima.ArimaModel")
    jcmps = jpype.JArray(ArimaModelClass)([_r2jd_arima(c) for c in ucm.components])  # type: ignore[call-non-callable]
    return UcarimaModels.of(jmodel, jcmps)


def _jd2r_ucarima(jucm: object) -> UcarimaModel:
    """Convert a Java UcarimaModel to a Python UcarimaModel via protobuf."""
    import jpype

    UcarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.UcarimaModels")
    buf = bytes(UcarimaModels.toBuffer(jucm))
    from pydemetra._proto import modelling_pb2

    msg = modelling_pb2.UcarimaModel()
    msg.ParseFromString(buf)
    return p2r_ucarima(msg)


def sarima_properties(model: SarimaModel, nspectrum: int = 601, nacf: int = 36) -> dict:
    """Compute the ACF and spectrum of a SARIMA model.

    Args:
        model (SarimaModel): A SarimaModel.
        nspectrum (int): Number of spectrum points in [0, pi].
        nacf (int): Maximum lag for ACF.

    Returns:
        dict: Dict with keys ``"acf"`` and ``"spectrum"``.
    """
    _ensure_jvm()
    import jpype

    jmodel = _r2jd_sarima(model)
    SarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.SarimaModels")
    spectrum = np.array(SarimaModels.spectrum(jmodel, int(nspectrum)))
    acf = np.array(SarimaModels.acf(jmodel, int(nacf)))
    return {"acf": acf, "spectrum": spectrum}


def sarima_random(
    model: SarimaModel,
    length: int,
    stde: float = 1.0,
    tdegree: int = 0,
    seed: int = -1,
) -> np.ndarray:
    """Simulate a SARIMA series.

    Args:
        model (SarimaModel): A SarimaModel.
        length (int): Length of the output series.
        stde (float): Standard deviation of innovations.
        tdegree (int): Degrees of freedom for T distribution (0 for normal).
        seed (int): Random seed (negative for random).

    Returns:
        np.ndarray: A numpy array with the simulated series.
    """
    _ensure_jvm()
    import jpype

    SarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.SarimaModels")
    result = SarimaModels.random(
        int(length),
        int(model.period),
        np.array(model.phi, dtype=np.float64),
        int(model.d),
        np.array(model.theta, dtype=np.float64),
        np.array(model.bphi, dtype=np.float64),
        int(model.bd),
        np.array(model.btheta, dtype=np.float64),
        float(stde),
        int(tdegree),
        int(seed),
    )
    return np.array(result)


def sarima_decompose(
    model: SarimaModel, rmod: float = 0.0, epsphi: float = 0.0
) -> UcarimaModel | None:
    """Decompose a SARIMA model into trend, seasonal, and irregular components.

    Args:
        model (SarimaModel): A SarimaModel.
        rmod (float): Trend threshold.
        epsphi (float): Seasonal tolerance (in degrees).

    Returns:
        UcarimaModel | None: A UcarimaModel, or None if decomposition fails.
    """
    _ensure_jvm()
    import jpype

    jmodel = _r2jd_sarima(model)
    UcarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.UcarimaModels")
    jucm = UcarimaModels.decompose(jmodel, float(rmod), float(epsphi))
    if jucm is None:
        return None
    return _jd2r_ucarima(jucm)


def arima_model(
    name: str = "arima",
    ar: list[float] | None = None,
    delta: list[float] | None = None,
    ma: list[float] | None = None,
    variance: float = 1.0,
) -> ArimaModel:
    """Create an ARIMA model.

    Args:
        name (str): Name of the model.
        ar (list[float] | None): AR polynomial coefficients. True signs.
        delta (list[float] | None): Non-stationary AR polynomial.
        ma (list[float] | None): MA polynomial coefficients. True signs.
        variance (float): Innovation variance.

    Returns:
        ArimaModel: An ArimaModel dataclass.
    """
    return ArimaModel(
        name=name,
        ar=ar if ar is not None else [1.0],
        delta=delta if delta is not None else [1.0],
        ma=ma if ma is not None else [1.0],
        var=variance,
    )


def arima_sum(*components: ArimaModel) -> ArimaModel:
    """Sum ARIMA models with independent innovations.

    Args:
        *components (ArimaModel): ARIMA models to sum.

    Returns:
        ArimaModel: The summed ArimaModel.
    """
    _ensure_jvm()
    import jpype

    ArimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.ArimaModels")
    ArimaModelClass = jpype.JClass("jdplus.toolkit.base.core.arima.ArimaModel")
    jarr = jpype.JArray(ArimaModelClass)([_r2jd_arima(c) for c in components])  # type: ignore[call-non-callable]
    jsum = ArimaModels.sum(jarr)
    return _jd2r_arima(jsum)


def arima_difference(left: ArimaModel, right: ArimaModel, simplify: bool = True) -> ArimaModel:
    """Subtract one ARIMA model from another.

    Args:
        left (ArimaModel): Left operand.
        right (ArimaModel): Right operand.
        simplify (bool): Simplify common roots.

    Returns:
        ArimaModel: The difference ArimaModel.
    """
    _ensure_jvm()
    jleft = _r2jd_arima(left)
    jright = _r2jd_arima(right)
    jdiff = jleft.minus(jright, simplify)  # type: ignore[unresolved-attribute]
    return _jd2r_arima(jdiff)


def arima_properties(model: ArimaModel, nspectrum: int = 601, nac: int = 36) -> dict:
    """Compute ACF and spectrum of an ARIMA model.

    Args:
        model (ArimaModel): An ArimaModel.
        nspectrum (int): Number of spectrum points.
        nac (int): Maximum lag for auto-covariances.

    Returns:
        dict: Dict with keys ``"acf"`` and ``"spectrum"``.
    """
    _ensure_jvm()
    import jpype

    jmodel = _r2jd_arima(model)
    ArimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.ArimaModels")
    spectrum = np.array(ArimaModels.spectrum(jmodel, int(nspectrum)))
    acf = np.array(ArimaModels.acf(jmodel, int(nac)))
    return {"acf": acf, "spectrum": spectrum}


def ucarima_model(
    model: ArimaModel | None = None,
    components: list[ArimaModel] | None = None,
    complements: list[ArimaModel] | None = None,
    checkmodel: bool = False,
) -> UcarimaModel:
    """Create a UCARIMA model from ARIMA components with independent innovations.

    Args:
        model (ArimaModel | None): The reduced model. If None, computed as the sum of components.
        components (list[ArimaModel] | None): The ARIMA component models.
        complements (list[ArimaModel] | None): Complements of some components.
        checkmodel (bool): Whether to validate the model.

    Returns:
        UcarimaModel: A UcarimaModel.
    """
    components = components or []
    complements = complements or []
    if model is None:
        model = arima_sum(*components)
    return UcarimaModel(model=model, components=components, complements=complements)


def ucarima_wk(
    ucm: UcarimaModel,
    cmp: int,
    signal: bool = True,
    nspectrum: int = 601,
    nwk: int = 300,
) -> dict:
    """Compute Wiener-Kolmogorov estimators for a UCARIMA component.

    Args:
        ucm (UcarimaModel): A UcarimaModel.
        cmp (int): 1-based index of the component.
        signal (bool): True for signal, False for noise.
        nspectrum (int): Number of spectrum points.
        nwk (int): Number of filter weights.

    Returns:
        dict: Dict with ``"spectrum"``, ``"filter"``, and ``"gain2"`` keys.
    """
    _ensure_jvm()
    import jpype

    UcarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.UcarimaModels")
    jucm = _r2jd_ucarima(ucm)
    jwks = UcarimaModels.wienerKolmogorovEstimators(jucm)
    jwk = UcarimaModels.finalEstimator(jwks, int(cmp - 1), signal)

    spectrum = np.array(UcarimaModels.spectrum(jwk, int(nspectrum)))
    wk = np.array(UcarimaModels.filter(jwk, int(nwk)))
    gain = np.array(UcarimaModels.gain(jwk, int(nspectrum)))

    return {"spectrum": spectrum, "filter": wk, "gain2": gain * gain}


def ucarima_canonical(ucm: UcarimaModel, cmp: int = 0, adjust: bool = True) -> UcarimaModel:
    """Make a UCARIMA model canonical.

    Args:
        ucm (UcarimaModel): A UcarimaModel.
        cmp (int): Index of component to receive noise (0 for new component).
        adjust (bool): Ensure positive pseudo-spectrum.

    Returns:
        UcarimaModel: A new canonical UcarimaModel.
    """
    _ensure_jvm()
    import jpype

    UcarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.UcarimaModels")
    jucm = _r2jd_ucarima(ucm)
    jnucm = UcarimaModels.doCanonical(jucm, int(cmp - 1), adjust)
    return _jd2r_ucarima(jnucm)


def ucarima_estimate(x: np.ndarray, ucm: UcarimaModel, stdev: bool = True) -> np.ndarray:
    """Estimate UCARIMA components from data.

    Args:
        x (np.ndarray): Input time series values.
        ucm (UcarimaModel): A UcarimaModel.
        stdev (bool): Whether to include standard deviations.

    Returns:
        np.ndarray: A matrix with component estimates (and standard deviations if requested).
    """
    _ensure_jvm()
    import jpype

    UcarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.UcarimaModels")
    jucm = _r2jd_ucarima(ucm)
    jcmps = UcarimaModels.estimate(np.asarray(x, dtype=np.float64), jucm, stdev)
    result = jd2r_matrix(jcmps)
    assert result is not None
    return result


def sarima_estimate(
    x: np.ndarray,
    order: tuple[int, int, int] = (0, 0, 0),
    seasonal: dict | tuple[int, int, int] | None = None,
    mean: bool = False,
    xreg: np.ndarray | None = None,
    eps: float = 1e-9,
) -> dict:
    """Estimate a SARIMA model by maximum likelihood.

    Args:
        x (np.ndarray): Input time series values.
        order (tuple[int, int, int]): (p, d, q) non-seasonal orders.
        seasonal (dict | tuple[int, int, int] | None): Dict with ``"order"`` and
            ``"period"`` keys, or a (P, D, Q) tuple.
        mean (bool): Include intercept.
        xreg (np.ndarray | None): External regressors matrix.
        eps (float): Convergence precision.

    Returns:
        dict: Dict with estimation results.
    """
    _ensure_jvm()
    import jpype

    from pydemetra._proto import regarima_pb2

    if seasonal is None:
        seasonal = {"order": (0, 0, 0), "period": 12}
    elif isinstance(seasonal, tuple):
        seasonal = {"order": seasonal, "period": 12}

    period = seasonal.get("period", 12)
    sorder = seasonal["order"]

    SarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.SarimaModels")
    jxreg = r2jd_matrix(xreg)

    jestim = SarimaModels.estimate(
        np.asarray(x, dtype=np.float64),
        np.array(order, dtype=np.int32),
        int(period),
        np.array(sorder, dtype=np.int32),
        mean,
        jxreg,
        jpype.JObject(None, jpype.JArray(jpype.JDouble)),
        float(eps),
    )

    buf = bytes(SarimaModels.toBuffer(jestim))
    msg = regarima_pb2.RegArimaModel.Estimation()
    msg.ParseFromString(buf)

    from pydemetra._converters import p2r_likelihood, p2r_matrix

    result: dict[str, Any] = {
        "y": np.array(list(msg.y)),
        "b": np.array(list(msg.b)),
        "residuals": np.array(list(msg.residuals)),
    }

    if msg.HasField("x"):
        result["x"] = p2r_matrix(msg.x)
    if msg.HasField("bcovariance"):
        result["bcovariance"] = p2r_matrix(msg.bcovariance)
    if msg.HasField("parameters"):
        result["parameters"] = {
            "val": np.array(list(msg.parameters.value)),
            "score": np.array(list(msg.parameters.score)),
        }
    if msg.HasField("likelihood"):
        result["likelihood"] = p2r_likelihood(msg.likelihood)

    result["orders"] = {"order": order, "seasonal": seasonal}
    return result


def sarima_hannan_rissanen(
    x: np.ndarray,
    order: tuple[int, int, int] = (0, 0, 0),
    seasonal: dict | tuple[int, int, int] | None = None,
    initialization: str = "Ols",
    bias_correction: bool = True,
    final_correction: bool = True,
) -> SarimaModel:
    """Estimate SARIMA model using the Hannan-Rissanen method.

    Args:
        x (np.ndarray): Input time series values.
        order (tuple[int, int, int]): (p, d, q) non-seasonal orders.
        seasonal (dict | tuple[int, int, int] | None): Dict with ``"order"`` and
            ``"period"`` keys, or (P, D, Q) tuple.
        initialization (str): ``"Ols"``, ``"Levinson"``, or ``"Burg"``.
        bias_correction (bool): Apply bias correction.
        final_correction (bool): Apply final correction (Tramo style).

    Returns:
        SarimaModel: An estimated SarimaModel.
    """
    _ensure_jvm()
    import jpype

    if seasonal is None:
        seasonal = {"order": (0, 0, 0), "period": 12}
    elif isinstance(seasonal, tuple):
        seasonal = {"order": seasonal, "period": 12}

    period = seasonal.get("period", 12)
    sorder = seasonal["order"]

    SarimaModels = jpype.JClass("jdplus.toolkit.base.r.arima.SarimaModels")
    jmodel = SarimaModels.hannanRissanen(
        np.asarray(x, dtype=np.float64),
        np.array(order, dtype=np.int32),
        int(period),
        np.array(sorder, dtype=np.int32),
        initialization,
        bias_correction,
        final_correction,
    )
    return _jd2r_sarima(jmodel)
