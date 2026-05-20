# Contributing to GidronAI Core

Thank you for your interest in contributing to GidronAI. This document outlines the process for contributing to the project.

## Development Setup

```bash
git clone https://github.com/LeonardKleinX/gidronai-core.git
cd gidronai-core
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.
- Type hints are required for all public functions. We run `mypy --strict`.
- Docstrings follow the NumPy docstring convention.

## Running Tests

```bash
pytest
```

For coverage:

```bash
pytest --cov=gidronai --cov-report=html
```

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. Add tests for any new functionality.
3. Ensure the test suite passes and linting is clean (`ruff check .`).
4. Update documentation if you changed public APIs.
5. Open a pull request with a clear description of the changes.

## Reporting Issues

Open an issue on GitHub with:
- A minimal reproducible example
- Your Python version and OS
- The full traceback (if applicable)

## Code of Conduct

Be respectful and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
