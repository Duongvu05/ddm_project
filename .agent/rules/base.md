# Agent Rules

This project follows strict development guidelines to ensure consistency and quality. All agents working on this project must adhere to the following rules:

## Tooling
- **Dependency Management**: Always use `uv` for managing dependencies and environments.
- **Linting & Formatting**: Use `ruff` for all linting and formatting tasks.
- **Static Analysis**: Use `pyright` for static type checking.
- **Testing**: Use `pytest` for all unit and integration tests.

## Coding Standards
- **Logging**: Never use the built-in `print()` function for logging. Always use the `loguru` logger.
    - Example: `from loguru import logger; logger.info("Message")`
- **Type Hints**: All functions and methods must have comprehensive type hints.
- **Documentation**: Use Google-style docstrings for all public modules, classes, and functions.
