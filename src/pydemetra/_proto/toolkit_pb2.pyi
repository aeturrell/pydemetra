from typing import Any

from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

class _Message:
    def CopyFrom(self, other_msg: Any) -> None: ...
    def MergeFrom(self, other_msg: Any) -> None: ...
    def ParseFromString(self, s: bytes) -> int: ...
    def SerializeToString(self) -> bytes: ...
    def HasField(self, field_name: str) -> bool: ...

SelectionType: EnumTypeWrapper
ParameterType: EnumTypeWrapper
CalendarEvent: EnumTypeWrapper

PARAMETER_UNUSED: int

class Date(_Message):
    year: int
    month: int
    day: int
    def __init__(self, *, year: int = ..., month: int = ..., day: int = ...) -> None: ...

class TimeSelector(_Message):
    type: int
    n0: int
    n1: int
    d0: Date
    d1: Date
    def __init__(self, **kwargs: Any) -> None: ...

class Parameter(_Message):
    value: float
    type: int
    description: str
    def __init__(self, **kwargs: Any) -> None: ...

class TsData(_Message):
    name: str
    annual_frequency: int
    start_year: int
    start_period: int
    values: list[float]
    def __init__(self, **kwargs: Any) -> None: ...

class Matrix(_Message):
    name: str
    nrows: int
    ncols: int
    values: list[float]
    def __init__(self, **kwargs: Any) -> None: ...

class ParametersEstimation(_Message):
    value: list[float]
    score: list[float]
    covariance: Matrix
    description: str
    def __init__(self, **kwargs: Any) -> None: ...

class StatisticalTest(_Message):
    value: float
    pvalue: float
    description: str
    def __init__(self, **kwargs: Any) -> None: ...

class OneWayAnova(_Message):
    SSM: float
    dfm: int
    SSR: float
    dfr: int
    test: StatisticalTest
    def __init__(self, **kwargs: Any) -> None: ...

class LikelihoodStatistics(_Message):
    nobs: int
    neffectiveobs: int
    nparams: int
    log_likelihood: float
    adjusted_log_likelihood: float
    aic: float
    aicc: float
    bic: float
    bicc: float
    ssq: float
    def __init__(self, **kwargs: Any) -> None: ...

class ValidityPeriod(_Message):
    start: Date
    end: Date
    def __init__(self, **kwargs: Any) -> None: ...

class FixedDay(_Message):
    month: int
    day: int
    weight: float
    validity: ValidityPeriod
    def __init__(
        self, *, month: int = ..., day: int = ..., weight: float = ..., **kwargs: Any
    ) -> None: ...

class EasterRelatedDay(_Message):
    offset: int
    julian: bool
    weight: float
    validity: ValidityPeriod
    def __init__(
        self, *, offset: int = ..., julian: bool = ..., weight: float = ..., **kwargs: Any
    ) -> None: ...

class PrespecifiedHoliday(_Message):
    event: int
    offset: int
    weight: float
    validity: ValidityPeriod
    def __init__(
        self, *, event: int = ..., offset: int = ..., weight: float = ..., **kwargs: Any
    ) -> None: ...

class FixedWeekDay(_Message):
    month: int
    position: int
    weekday: int
    weight: float
    validity: ValidityPeriod
    def __init__(
        self,
        *,
        month: int = ...,
        position: int = ...,
        weekday: int = ...,
        weight: float = ...,
        **kwargs: Any,
    ) -> None: ...

class SingleDate(_Message):
    date: Date
    weight: float
    def __init__(self, *, weight: float = ..., **kwargs: Any) -> None: ...

class Calendar(_Message):
    fixed_days: list[FixedDay]
    easter_related_days: list[EasterRelatedDay]
    fixed_week_days: list[FixedWeekDay]
    prespecified_holidays: list[PrespecifiedHoliday]
    single_dates: list[SingleDate]
    mean_correction: bool
    def __init__(self, **kwargs: Any) -> None: ...
