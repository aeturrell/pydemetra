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

### X-13ARIMA-SEATS seasonal adjustment

```python
import pandas as pd
import numpy as np
import pydemetra as jd

# Create a sample monthly time series
index = pd.period_range("2000-01", periods=240, freq="M")
ts = pd.Series(
    np.cumsum(np.random.default_rng(42).normal(size=240)) + 100,
    index=index,
)

# Create a default X-13 specification and run seasonal adjustment
spec = jd.x13_spec("rsa5c")
result = jd.x13(ts, spec)

# Extract key components
sa = result["result"]["final"]["d11final"]       # seasonally adjusted series
trend = result["result"]["final"]["d12final"]     # trend
seasonal = result["result"]["final"]["d16"]       # seasonal component
irregular = result["result"]["final"]["d13final"] # irregular component

# Quality diagnostics (M-statistics; Q < 1 is acceptable)
print(result["result"]["mstats"]["q"])

# Use a string shorthand instead of a spec object
result = jd.x13(ts, "rsa5")
result["result"]["final"]["d11"]

# Modify the X-11 decomposition parameters
spec = jd.set_x11(spec, henderson_filter=13, seasonal_filter="S3X5")
result = jd.x13(ts, spec)

# Pure X-11 decomposition (no RegARIMA pre-processing)
x11_result = jd.x11(ts)
print(x11_result["d11"])  # seasonally adjusted (D11)

# RegARIMA modelling only
regarima_result = jd.x13_regarima(ts, "rg5c")
print(regarima_result["result"]["estimation"]["likelihood"])
```

### Customising specifications

```python
import pydemetra as jd

# Start from RSA5 and customise
spec = jd.x13_spec("rsa5c")

# Force log transformation (no automatic detection)
spec["regarima"]["transform"]["fn"] = "LOG"

# Disable transitory component (TC) outlier detection
spec["regarima"]["outlier"]["outliers"] = [
    o for o in spec["regarima"]["outlier"]["outliers"] if o["type"] != "TC"
]

# Disable trading day regressors (suitable for quarterly data)
spec["regarima"]["regression"]["td"]["td"] = "TD_NONE"
spec["regarima"]["regression"]["td"]["auto"] = "AUTO_NO"

result = jd.x13(ts, spec)
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