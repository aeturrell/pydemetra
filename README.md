# pydemetra

Python front-end to [JDemetra+](https://github.com/jdemetra), which is a Java package for **seasonal adjustment**.

Want to just get going? Head to the [Quick Start](quick_start.ipynb) page.

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
