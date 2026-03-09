from typing import Any

from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

class _Message:
    def CopyFrom(self, other_msg: Any) -> None: ...
    def MergeFrom(self, other_msg: Any) -> None: ...
    def ParseFromString(self, s: bytes) -> int: ...
    def SerializeToString(self) -> bytes: ...
    def HasField(self, field_name: str) -> bool: ...

DecompositionMode: EnumTypeWrapper
BenchmarkingTarget: EnumTypeWrapper
BenchmarkingBias: EnumTypeWrapper
IdentifiableSeasonality: EnumTypeWrapper

class SaDecomposition(_Message):
    mode: int
    series: Any
    seasonally_adjusted: Any
    trend: Any
    seasonal: Any
    irregular: Any
    def __init__(self, **kwargs: Any) -> None: ...

class BenchmarkingSpec(_Message):
    enabled: bool
    target: int
    rho: float
    bias: int
    forecast: bool
    def __init__(self, **kwargs: Any) -> None: ...

class CombinedSeasonalityTest(_Message):
    seasonality: int
    kruskal_wallis: Any
    stable_seasonality: Any
    evolutive_seasonality: Any
    def __init__(self, **kwargs: Any) -> None: ...
