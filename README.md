# not-again-ai

[![GitHub Actions][github-actions-badge]](https://github.com/johnthagen/python-blueprint/actions)
[![uv][uv-badge]](https://github.com/astral-sh/uv)
[![Nox][nox-badge]](https://github.com/wntrblm/nox)
[![Ruff][ruff-badge]](https://github.com/astral-sh/ruff)
[![Type checked with mypy][mypy-badge]](https://mypy-lang.org/)

[github-actions-badge]: https://github.com/johnthagen/python-blueprint/workflows/python/badge.svg
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[nox-badge]: https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[mypy-badge]: https://www.mypy-lang.org/static/mypy_badge.svg

**not-again-ai** is a collection of various building blocks that come up over and over again when developing AI products. 
The key goals of this package are to have simple, yet flexible interfaces and to minimize dependencies. 
It is encouraged to also **a)** use this as a template for your own Python package. 
**b)** instead of installing the package, copy and paste functions into your own projects. 
We make this easier by limiting the number of dependencies and use an MIT license.

**Documentation** available within individual **[notebooks](notebooks)** or docstrings within the source code.

# Installation

Requires: Python 3.11, or 3.12 which can be installed with [uv](https://docs.astral.sh/uv/getting-started/installation/) by running the command `uv python install 3.12`

Install the entire package from [PyPI](https://pypi.org/project/not-again-ai/) with: 

```bash
$ pip install not_again_ai[data,llm,statistics,viz]
```

The package is split into subpackages, so you can install only the parts you need.

### Base
1. `pip install not_again_ai`


### Data
1. `pip install not_again_ai[data]`
1. `crawl4ai-setup` to run crawl4ai post-installation setup.
1. Set the `BRAVE_SEARCH_API_KEY` environment variable to use the Brave Search API for web data extraction.
   1. Get the API key from https://api-dashboard.search.brave.com/app/keys. You must have at least the Free "Data for Search" subscription.


### LLM
1. `pip install not_again_ai[llm]`
1. Setup OpenAI API
   1. Go to https://platform.openai.com/settings/profile?tab=api-keys to get your API key.
   1. (Optional) Set the `OPENAI_API_KEY` and the `OPENAI_ORG_ID` environment variables.
1. Setup Azure OpenAI (AOAI)
   1. Using AOAI requires using Entra ID authentication. See https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity for how to set this up for your AOAI deployment.
      * Requires the correct role assigned to your user account and being signed into the Azure CLI.
   1. (Optional) Set the `AZURE_OPENAI_ENDPOINT` environment variable.
1. If you wish to use Ollama:
     1. Follow the instructions at https://github.com/ollama/ollama to install Ollama for your system. 
     1. (Optional) [Add Ollama as a startup service (recommended)](https://github.com/ollama/ollama/blob/main/docs/linux.md#adding-ollama-as-a-startup-service-recommended)
     1. (Optional) To make the Ollama service accessible on your local network from a Linux server, add the following to the `/etc/systemd/system/ollama.service` file which will make Ollama available at `http://<local_address>:11434`:
         ```bash
         [Service]
         ...
         Environment="OLLAMA_HOST=0.0.0.0"
         ```
     1. It is recommended to always have the latest version of Ollama. To update Ollama check the [docs](https://github.com/ollama/ollama/blob/main/docs/). The command for Linux is: `curl -fsSL https://ollama.com/install.sh | sh`


### Statistics
1. `pip install not_again_ai[statistics]`


### Visualization
1. `pip install not_again_ai[viz]`


# Development Information

This package uses [uv](https://docs.astral.sh/uv/) to manage dependencies and
isolated [Python virtual environments](https://docs.python.org/3/library/venv.html).

To proceed,
[install uv globally](https://docs.astral.sh/uv/getting-started/installation/)
onto your system.

To install a specific version of Python:

```shell
uv python install 3.12
```

## Dependencies

Dependencies are defined in [`pyproject.toml`](./pyproject.toml) and specific versions are locked
into [`uv.lock`](./uv.lock). This allows for exact reproducible environments across
all machines that use the project, both during development and in production.

To install all dependencies into an isolated virtual environment:

```shell
uv sync --all-extras --all-groups
```

To upgrade all dependencies to their latest versions:

```shell
uv lock --upgrade
```

## Packaging

This project is designed as a Python package, meaning that it can be bundled up and redistributed
as a single compressed file.

Packaging is configured by the [`pyproject.toml`](./pyproject.toml).

To package the project as both a 
[source distribution](https://packaging.python.org/en/latest/flow/#the-source-distribution-sdist) and
a [wheel](https://packaging.python.org/en/latest/specifications/binary-distribution-format/):

```bash
$ uv build
```

This will generate `dist/not-again-ai-<version>.tar.gz` and `dist/not_again_ai-<version>-py3-none-any.whl`.


## Publish Distributions to PyPI

Source and wheel redistributable packages can
be [published to PyPI](https://docs.astral.sh/uv/guides/package/) or installed
directly from the filesystem using `pip`.

```shell
uv publish
```

# Enforcing Code Quality

Automated code quality checks are performed using [Nox](https://nox.thea.codes/en/stable/). Nox
will automatically create virtual environments and run commands based on
[`noxfile.py`](./noxfile.py) for unit testing, PEP 8 style guide checking, type checking and
documentation generation.

To run all default sessions:

```shell
uv run nox
```

## Unit Testing

Unit testing is performed with [pytest](https://pytest.org/). pytest has become the de facto Python
unit testing framework. Some key advantages over the built-in
[unittest](https://docs.python.org/3/library/unittest.html) module are:

1. Significantly less boilerplate needed for tests.
2. PEP 8 compliant names (e.g. `pytest.raises()` instead of `self.assertRaises()`).
3. Vibrant ecosystem of plugins.

pytest will automatically discover and run tests by recursively searching for folders and `.py`
files prefixed with `test` for any functions prefixed by `test`.

The `tests` folder is created as a Python package (i.e. there is an `__init__.py` file within it)
because this helps `pytest` uniquely namespace the test files. Without this, two test files cannot
be named the same, even if they are in different subdirectories.

Code coverage is provided by the [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) plugin.

When running a unit test Nox session (e.g. `nox -s test`), an HTML report is generated in
the `htmlcov` folder showing each source file and which lines were executed during unit testing.
Open `htmlcov/index.html` in a web browser to view the report. Code coverage reports help identify
areas of the project that are currently not tested.

pytest and code coverage are configured in [`pyproject.toml`](./pyproject.toml).

To run selected tests:

```bash
(.venv) $ uv run nox -s test -- -k "test_web"
```

## Code Style Checking

[PEP 8](https://peps.python.org/pep-0008/) is the universally accepted style guide for Python
code. PEP 8 code compliance is verified using [Ruff][Ruff]. Ruff is configured in the
`[tool.ruff]` section of [`pyproject.toml`](./pyproject.toml).

[Ruff]: https://github.com/astral-sh/ruff

To lint code, run:

```bash
(.venv) $ uv run nox -s lint
```

To automatically fix fixable lint errors, run:

```bash
(.venv) $ uv run nox -s lint_fix
```

## Automated Code Formatting

[Ruff][Ruff] is used to automatically format code and group and sort imports.

To automatically format code, run:

```bash
(.venv) $ uv run nox -s fmt
```

To verify code has been formatted, such as in a CI job:

```bash
(.venv) $ uv run nox -s fmt_check
```

## Type Checking

[Type annotations](https://docs.python.org/3/library/typing.html) allows developers to include
optional static typing information to Python source code. This allows static analyzers such
as [mypy](http://mypy-lang.org/), [PyCharm](https://www.jetbrains.com/pycharm/),
or [Pyright](https://github.com/microsoft/pyright) to check that functions are used with the
correct types before runtime.


```python
def factorial(n: int) -> int:
    ...
```

mypy is configured in [`pyproject.toml`](./pyproject.toml). To type check code, run:

```bash
(.venv) $ uv run nox -s type_check
```

### Distributing Type Annotations

[PEP 561](https://www.python.org/dev/peps/pep-0561/) defines how a Python package should
communicate the presence of inline type annotations to static type
checkers. [mypy's documentation](https://mypy.readthedocs.io/en/stable/installed_packages.html)
provides further examples on how to do this.

Mypy looks for the existence of a file named [`py.typed`](./src/not-again-ai/py.typed) in the root of the
installed package to indicate that inline type annotations should be checked.

## Typos

Check for typos using [typos](https://github.com/crate-ci/typos)

```bash
(.venv) $ uv run nox -s typos
```

## Continuous Integration

Continuous integration is provided by [GitHub Actions](https://github.com/features/actions). This
runs all tests, lints, and type checking for every commit and pull request to the repository.

GitHub Actions is configured in [`.github/workflows/python.yml`](./.github/workflows/python.yml).

## [Visual Studio Code](https://code.visualstudio.com/docs/languages/python)

Install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) for VSCode.

Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) for VSCode.

Default settings are configured in [`.vscode/settings.json`](./.vscode/settings.json) which will enable Ruff with consistent settings.

# Attributions
[python-blueprint](https://github.com/johnthagen/python-blueprint) for the Python package skeleton.

This project uses Crawl4AI (https://github.com/unclecode/crawl4ai) for web data extraction.
