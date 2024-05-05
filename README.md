# not-again-ai

[![GitHub Actions][github-actions-badge]](https://github.com/johnthagen/python-blueprint/actions)
[![Packaged with Poetry][poetry-badge]](https://python-poetry.org/)
[![Nox][nox-badge]](https://github.com/wntrblm/nox)
[![Ruff][ruff-badge]](https://github.com/astral-sh/ruff)
[![Type checked with mypy][mypy-badge]](https://mypy-lang.org/)

[github-actions-badge]: https://github.com/johnthagen/python-blueprint/workflows/python/badge.svg
[poetry-badge]: https://img.shields.io/badge/packaging-poetry-cyan.svg
[nox-badge]: https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[mypy-badge]: https://www.mypy-lang.org/static/mypy_badge.svg

**not-again-ai** is a collection of various building blocks that come up over and over again when developing AI products. The key goals of this package are to have simple, but flexible interfaces and to minimize dependencies. Feel free to **a)** use this as a template for your own Python package. **b)** instead of installing the package, copy and paste functions into your own projects (this is made possible with the limited amount of dependencies and the MIT license).

**Documentation** available within individual **[notebooks](notebooks)**, docstrings within the source, or auto-generated at [DaveCoDev.github.io/not-again-ai/](https://DaveCoDev.github.io/not-again-ai/).

# Installation

Requires: Python 3.11, or 3.12

Install the entire package from [PyPI](https://pypi.org/project/not-again-ai/) with: 

```bash
$ pip install not_again_ai[llm,statistics,viz]
```

The package is split into subpackages, so you can install only the parts you need. See the **[notebooks](notebooks)** for examples.
* **Base only**: `pip install not_again_ai`
* **LLM**: `pip install not_again_ai[llm]`
    1. If you wish to use OpenAI
        1. Go to https://platform.openai.com/settings/profile?tab=api-keys to get your API key.
        1. (Optionally) Set the `OPENAI_API_KEY` and the `OPENAI_ORG_ID` environment variables.
    1. If you wish to use Ollama:
        1. follow the instructions to install ollama for your system: https://github.com/ollama/ollama
        1. [Add Ollama as a startup service (recommended)](https://github.com/ollama/ollama/blob/main/docs/linux.md#adding-ollama-as-a-startup-service-recommended)
        1. If you'd like to make the ollama service accessible on your local network and it is hosted on Linux, add the following to the `/etc/systemd/system/ollama.service` file:
            ```bash
            [Service]
            ...
            Environment="OLLAMA_HOST=0.0.0.0"
            ```
        Now ollama will be available at `http://<local_address>:11434`
* **Statistics**: `pip install not_again_ai[statistics]`
* **Visualization**: `pip install not_again_ai[viz]`


# Development Information

The following information is relevant if you would like to contribute or use this package as a template for yourself. 

This package uses [Poetry](https://python-poetry.org/) to manage dependencies and
isolated [Python virtual environments](https://docs.python.org/3/library/venv.html). To proceed, be sure to first install [pipx](https://github.com/pypa/pipx#install-pipx)
and then [install Poetry](https://python-poetry.org/docs/#installing-with-pipx).

Install Poetry Plugin: Export

```bash
$ pipx inject poetry poetry-plugin-export
```

(Optional) configure Poetry to use an in-project virtual environment.
```bash
$ poetry config virtualenvs.in-project true
```

## Dependencies

Dependencies are defined in [`pyproject.toml`](./pyproject.toml) and specific versions are locked
into [`poetry.lock`](./poetry.lock). This allows for exact reproducible environments across
all machines that use the project, both during development and in production.

To upgrade all dependencies to the versions defined in [`pyproject.toml`](./pyproject.toml):

```bash
$ poetry update
```

To install all dependencies (with all extra dependencies) into an isolated virtual environment:

> Append `--sync` to uninstall dependencies that are no longer in use from the virtual environment.

```bash
$ poetry install --all-extras
```

To [activate](https://python-poetry.org/docs/basic-usage#activating-the-virtual-environment) the
virtual environment that is automatically created by Poetry:

```bash
$ poetry shell
```

To deactivate the environment:

```bash
(.venv) $ exit
```

## Packaging

This project is designed as a Python package, meaning that it can be bundled up and redistributed
as a single compressed file.

Packaging is configured by:

- [`pyproject.toml`](./pyproject.toml)

To package the project as both a 
[source distribution](https://packaging.python.org/en/latest/flow/#the-source-distribution-sdist) and
a [wheel](https://packaging.python.org/en/latest/specifications/binary-distribution-format/):

```bash
$ poetry build
```

This will generate `dist/not-again-ai-<version>.tar.gz` and `dist/not_again_ai-<version>-py3-none-any.whl`.

Read more about the [advantages of wheels](https://pythonwheels.com/) to understand why generating
wheel distributions are important.

## Publish Distributions to PyPI

Source and wheel redistributable packages can
be [published to PyPI](https://python-poetry.org/docs/cli#publish) or installed
directly from the filesystem using `pip`.

```bash
$ poetry publish
```

# Enforcing Code Quality

Automated code quality checks are performed using 
[Nox](https://nox.thea.codes/en/stable/) and
[`nox-poetry`](https://nox-poetry.readthedocs.io/en/stable/). Nox will automatically create virtual
environments and run commands based on [`noxfile.py`](./noxfile.py) for unit testing, PEP 8 style
guide checking, type checking and documentation generation.

> Note: `nox` is installed into the virtual environment automatically by the `poetry install`
> command above. Run `poetry shell` to activate the virtual environment.

To run all default sessions:

```bash
(.venv) $ nox
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

To pass arguments to `pytest` through `nox`:

```bash
(.venv) $ nox -s test -- -k invalid_factorial
```

## Code Style Checking

[PEP 8](https://peps.python.org/pep-0008/) is the universally accepted style guide for Python
code. PEP 8 code compliance is verified using [Ruff][Ruff]. Ruff is configured in the
`[tool.ruff]` section of [`pyproject.toml`](./pyproject.toml).

[Ruff]: https://github.com/astral-sh/ruff

To lint code, run:

```bash
(.venv) $ nox -s lint
```

To automatically fix fixable lint errors, run:

```bash
(.venv) $ nox -s lint_fix
```

## Automated Code Formatting

[Ruff][Ruff] is used to automatically format code and group and sort imports.

To automatically format code, run:

```bash
(.venv) $ nox -s fmt
```

To verify code has been formatted, such as in a CI job:

```bash
(.venv) $ nox -s fmt_check
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
(.venv) $ nox -s type_check
```

See also [awesome-python-typing](https://github.com/typeddjango/awesome-python-typing).

### Distributing Type Annotations

[PEP 561](https://www.python.org/dev/peps/pep-0561/) defines how a Python package should
communicate the presence of inline type annotations to static type
checkers. [mypy's documentation](https://mypy.readthedocs.io/en/stable/installed_packages.html)
provides further examples on how to do this.

Mypy looks for the existence of a file named [`py.typed`](./src/not-again-ai/py.typed) in the root of the
installed package to indicate that inline type annotations should be checked.

## Continuous Integration

Continuous integration is provided by [GitHub Actions](https://github.com/features/actions). This
runs all tests, lints, and type checking for every commit and pull request to the repository.

GitHub Actions is configured in [`.github/workflows/python.yml`](./.github/workflows/python.yml).

## [Visual Studio Code](https://code.visualstudio.com/docs/languages/python)

Install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) for VSCode.

Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) for VSCode.

Default settings are configured in [`.vscode/settings.json`](./.vscode/settings.json). This will enable Ruff and black with consistent settings.

# Documentation

## Generating a User Guide

[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) is a powerful static site
generator that combines easy-to-write Markdown, with a number of Markdown extensions that increase
the power of Markdown. This makes it a great fit for user guides and other technical documentation.

The example MkDocs project included in this project is configured to allow the built documentation
to be hosted at any URL or viewed offline from the file system.

To build the user guide, run,

```bash
(.venv) $ nox -s docs
```

and open `docs/user_guide/site/index.html` using a web browser.

To build the user guide, additionally validating external URLs, run:

```bash
(.venv) $ nox -s docs_check_urls
```

To build the user guide in a format suitable for viewing directly from the file system, run:

```bash
(.venv) $ nox -s docs_offline
```

To build and serve the user guide with automatic rebuilding as you change the contents,
run:

```bash
(.venv) $ nox -s docs_serve
``` 

and open <http://127.0.0.1:8000> in a browser.

Each time the `main` Git branch is updated, the 
[`.github/workflows/pages.yml`](.github/workflows/pages.yml) GitHub Action will
automatically build the user guide and publish it to [GitHub Pages](https://pages.github.com/).
This is configured in the `docs_github_pages` Nox session.

## Generating API Documentation

This project uses [mkdocstrings](https://github.com/mkdocstrings/mkdocstrings) plugin for
MkDocs, which renders
[Google-style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
into an MkDocs project. Google-style docstrings provide a good mix of easy-to-read docstrings in
code as well as nicely-rendered output.

```python
"""Computes the factorial through a recursive algorithm.

Args:
    n: A positive input value.

Raises:
    InvalidFactorialError: If n is less than 0.

Returns:
    Computed factorial.
"""
```

## Misc

If you get a `Failed to create the collection: Prompt dismissed..` error when running `poetry update` on Ubuntu, try setting the following environment variable:

    ```bash
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
    ```

# Attributions
[python-blueprint](https://github.com/johnthagen/python-blueprint) for the Python package skeleton.
