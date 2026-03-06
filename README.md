# pydemetra

Python bindings for JDemetra+ version 3.x seasonal adjustment and time series analysis algorithms.

## Requirements

- Python 3.11+
- Java 17+ (the JVM is started automatically on first use)

## Installation

```bash
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .
```

## Quick start

```python
import pandas as pd
import pydemetra as jd

# Create a SARIMA model specification
model = jd.sarima_model(period=12, phi=[0.8], d=1, theta=[-0.6], bd=1, btheta=[-0.5])
print(model)
# SARIMA(1,1,1)(0,1,1)[12]

# Build a calendar with holidays
cal = jd.national_calendar(days=[
    jd.fixed_day(month=1, day=1),       # New Year
    jd.fixed_day(month=12, day=25),      # Christmas
    jd.easter_day(offset=-2),            # Good Friday
])

# Generate trading day regressors
td_regs = jd.td(frequency=12, start=(2000, 1), length=120)

# Run seasonality tests (requires JVM)
result = jd.seasonality_qs(data, period=12)

# RegARIMA specification helpers
spec = {}
spec = jd.add_outlier(spec, type="AO", date="2020-03-01")
spec = jd.set_benchmarking(spec, enabled=True, target="Original")
```

Most functions that interact with JDemetra+ Java classes require a running JVM. The JVM is started lazily on the first call — no manual setup needed as long as Java 17+ is on your `PATH` or `JAVA_HOME` is set.

## Development

```bash
uv sync --group dev
uv run pytest tests/ -v
uv run ruff check pydemetra/ tests/
```


## Installing Java

### MacOS

```
brew install openjdk
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
```

restart terminal