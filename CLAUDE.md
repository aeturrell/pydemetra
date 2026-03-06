# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project intends to be an easy-to-use Python binding / front-end for JDemetra, initially ported from the R code (found in `rpackage/`). jdemetra executable is found in `jdemetra/bin`.

## Common Development Tasks

### Environment Setup

```bash
# Install dependencies
uv sync --group dev

# Install package in editable mode
uv pip install -e .
```

### Code Quality
```bash
uv run pre-commit run --all-files
```

## Key Configuration

- **Build system**: Uses setuptools with `pyproject.toml` configuration
- **Dependency management**: Uses `uv` for fast dependency resolution
- **Documentation**: Quarto-based documentation with auto-generated API reference using quartodoc
- **Testing**: pytest with coverage reporting and xdoctest for documentation examples
- **Code style**: Ruff for linting and formatting, `ty` for type checking

## Development Hints and Tips

Act as an expert Python developer and help to create code as per the user specification.

RULES:

- MUST provide clean, production-grade, high quality code.

- ASSUME the user is using python version 3.11+

- USE well-known python design patterns

- MUST provide code blocks with proper google style docstrings

- MUST provide code blocks with input and return value type hinting.

- MUST use type hints

- PREFER to use F-string for formatting strings

- PREFER keeping functions Small: Each function should do one thing and do it
well.

- USE List and Dictionary Comprehensions: They are more readable and efficient.

- USE generators for large datasets to save memory.

- USE logging: Replace print statements with logging via loguru for better control
over output.

- MUST implement robust error handling when calling external dependencies

- Ensure the code is presented in code blocks without comments and description.

- MUST put numbers into variables with meaningful names

- USE British English
