from typing import Any

from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

class _Message:
    def CopyFrom(self, other_msg: Any) -> None: ...
    def MergeFrom(self, other_msg: Any) -> None: ...
    def ParseFromString(self, s: bytes) -> int: ...
    def SerializeToString(self) -> bytes: ...
    def HasField(self, field_name: str) -> bool: ...

VariableType: EnumTypeWrapper
LengthOfPeriod: EnumTypeWrapper

class SarimaModel(_Message):
    name: str
    period: int
    phi: list[float]
    d: int
    theta: list[float]
    bphi: list[float]
    bd: int
    btheta: list[float]
    def __init__(self, **kwargs: Any) -> None: ...

class ArimaModel(_Message):
    name: str
    innovation_variance: float
    ar: list[float]
    delta: list[float]
    ma: list[float]
    def __init__(self, **kwargs: Any) -> None: ...

class UcarimaModel(_Message):
    model: ArimaModel
    components: list[ArimaModel]
    complements: list[ArimaModel]
    def __init__(self, **kwargs: Any) -> None: ...

class Outlier(_Message):
    name: str
    code: str
    position: Any
    coefficient: Any
    metadata: Any
    def __init__(self, **kwargs: Any) -> None: ...

class Ramp(_Message):
    name: str
    start: Any
    end: Any
    coefficient: Any
    metadata: Any
    def __init__(self, **kwargs: Any) -> None: ...

class TsVariable(_Message):
    name: str
    id: str
    lag: int
    coefficient: Any
    metadata: Any
    def __init__(self, **kwargs: Any) -> None: ...

class InterventionVariable(_Message):
    class Sequence(_Message):
        start: Any
        end: Any
        def __init__(self, **kwargs: Any) -> None: ...

    name: str
    sequences: list[Sequence]
    delta: float
    seasonal_delta: float
    coefficient: Any
    metadata: Any
    def __init__(self, **kwargs: Any) -> None: ...

class RegressionVariable(_Message):
    name: str
    var_type: int
    metadata: Any
    coefficients: list[Any]
    def __init__(self, **kwargs: Any) -> None: ...

class TsComponent(_Message):
    data: Any
    stde: list[float]
    nbcasts: int
    nfcasts: int
    def __init__(self, **kwargs: Any) -> None: ...

class StationaryTransformation(_Message):
    class Differencing:
        lag: int
        order: int

    mean_correction: bool
    differences: list[Differencing]
    stationary_series: list[float]
    def __init__(self, **kwargs: Any) -> None: ...
