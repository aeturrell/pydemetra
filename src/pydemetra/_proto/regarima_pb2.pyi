from typing import Any

class _Message:
    def CopyFrom(self, other_msg: Any) -> None: ...
    def MergeFrom(self, other_msg: Any) -> None: ...
    def ParseFromString(self, s: bytes) -> int: ...
    def SerializeToString(self) -> bytes: ...
    def HasField(self, field_name: str) -> bool: ...

class SarimaSpec(_Message):
    period: int
    d: int
    bd: int
    phi: list[Any]
    theta: list[Any]
    bphi: list[Any]
    btheta: list[Any]
    def __init__(self, **kwargs: Any) -> None: ...

class RegArimaModel(_Message):
    class Description:
        series: Any
        log: bool
        preadjustment: int
        variables: list[Any]
        arima: Any

    class Estimation(_Message):
        y: list[float]
        x: Any
        b: list[float]
        bcovariance: Any
        parameters: Any
        likelihood: Any
        residuals: list[float]
        missings: list[Any]

    description: Description
    estimation: Estimation
    diagnostics: Any
    def __init__(self, **kwargs: Any) -> None: ...
