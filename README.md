# pydemetra

![](docs/icon.svg)

![](icon.svg)


Python front-end to [JDemetra+](https://github.com/jdemetra), which is a [Java](https://www.java.com/en/) package for **seasonal adjustment**.

[![PyPI](https://img.shields.io/pypi/v/pydemetra.svg)](https://pypi.org/project/pydemetra/)
[![Status](https://img.shields.io/pypi/status/pydemetra.svg)](https://pypi.org/project/pydemetra/)
[![Python Version](https://img.shields.io/pypi/pyversions/pydemetra)](https://pypi.org/project/pydemetra)
[![License](https://img.shields.io/pypi/l/pydemetra)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests](https://github.com/aeturrell/pydemetra/workflows/Tests/badge.svg)](https://github.com/aeturrell/pydemetra/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/aeturrell/pydemetra/branch/main/graph/badge.svg)](https://codecov.io/gh/aeturrell/pydemetra)
[![Read the documentation at https://aeturrell.github.io/pydemetra/](https://img.shields.io/badge/Go%20to%20the%20docs-purple?style=flat)](https://aeturrell.github.io/pydemetra/)
[![Downloads](https://static.pepy.tech/badge/pydemetra)](https://pepy.tech/projects/pydemetra)

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
[![Source](https://img.shields.io/badge/source%20code-github-lightgrey?style=for-the-badge)](https://github.com/aeturrell/pydemetra)

⚠️ This repo is under development and not fully tested. Use with caution ⚠️

Want to just get going? Head to the [Quick Start](quick_start.ipynb) page.

This project has no affiliation the original JDemetra+, but we're grateful to its creators!

## Functionality

pydemetra was inspired by the [rjdverse](https://github.com/rjdverse) R front-end to JDemetra+. Without that package, this one would not be possible. Not all of the functionality of either JDemetra+ or the R front-end is implemented in this package.

### Implemented

- **X-13ARIMA-SEATS** seasonal adjustment (X-13, X-11, RegARIMA)
- **TRAMO-SEATS** seasonal adjustment (TRAMO, SEATS)
- **Calendars and trading days** — national calendars, fixed/Easter/weekday holidays, trading day regressors
- **ARIMA modelling** — SARIMA, UCARIMA, estimation, decomposition, simulation
- **Regression variables** — outliers (AO, LS, TC, SO), ramps, interventions, Easter/leap-year effects, periodic dummies, trigonometric variables
- **Statistical tests** — seasonality, normality, trading days, autocorrelation
- **Time series utilities** — aggregation, interpolation, differencing, length-of-period adjustment
- **Distributions** — t, chi-squared, gamma, inverse gamma, inverse Gaussian
- **Splines** — B-splines, natural/monotonic/periodic cubic splines
- **Benchmarking specification**

### Not implemented


- **STL** decomposition
- **State space** models
- **High-frequency** seasonal adjustment
- **Extended X-11**
- **Benchmarking and temporal disaggregation**
- **Revision analysis**
- **Trend-cycle filters**

## Prerequisites

- Python 3.11+
- Java 17+ (the JVM is started automatically on first use)

Most functions that interact with JDemetra+ Java classes require a running JVM. The JVM is started lazily on the first call — no manual setup needed as long as Java 17+ is on your `PATH` or `JAVA_HOME` is set.

### Installing Java on MacOS

To install a recent version of Java, run

```
brew install openjdk
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
```

and then restart your terminal.

## Development

1. git clone repo
2. branch
3. install with development requirements using `uv sync`
4. do the work you need to
5. `uv run nox` for full test suite; everything should pass
