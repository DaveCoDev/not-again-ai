import os

import nox
from nox import parametrize
from nox_poetry import Session, session

nox.options.error_on_external_run = True
nox.options.reuse_existing_virtualenvs = False
nox.options.sessions = [
    "lint",
    "type_check",
    "typos",
    "test",
    "docs",
]


@session(python=["3.11", "3.12"])
def test(s: Session) -> None:
    s.install(".[data,llm,statistics,viz]", "pytest", "pytest-asyncio", "pytest-cov", "pytest-randomly")

    # Skip tests in directories specified by the SKIP_TESTS_NAII environment variable.
    skip_tests = os.getenv("SKIP_TESTS_NAAI", "")
    skip_tests += " tests/llm/chat_completion/ tests/llm/embedding/"
    skip_args = [f"--ignore={dir}" for dir in skip_tests.split()] if skip_tests else []

    s.run(
        "python",
        "-m",
        "pytest",
        "--cov=not_again_ai",
        "--cov-report=html",
        "--cov-report=term",
        "tests",
        *skip_args,
        "-W ignore::DeprecationWarning",
        *s.posargs,
    )


# For some sessions, set venv_backend="none" to simply execute scripts within the existing Poetry
# environment. This requires that nox is run within `poetry shell` or using `poetry run nox ...`.
@session(venv_backend="none")
@parametrize(
    "command",
    [
        # During formatting, additionally sort imports and remove unused imports.
        [
            "ruff",
            "check",
            ".",
            "--select",
            "I",
            "--select",
            "F401",
            "--extend-fixable",
            "F401",
            "--fix",
        ],
        ["ruff", "format", "."],
    ],
)
def fmt(s: Session, command: list[str]) -> None:
    s.run(*command)


@session(venv_backend="none")
def fmt_check(s: Session) -> None:
    s.run("ruff", "check", ".", "--select", "I")


@session(venv_backend="none")
@parametrize(
    "command",
    [
        ["ruff", "check", "."],
        ["ruff", "format", "--check", "."],
    ],
)
def lint(s: Session, command: list[str]) -> None:
    s.run(*command)


@session(venv_backend="none")
def lint_fix(s: Session) -> None:
    s.run("ruff", "check", ".", "--extend-fixable", "F401", "--fix")


@session(venv_backend="none")
def type_check(s: Session) -> None:
    s.run("mypy", "src", "tests", "noxfile.py")


# Environment variable needed for mkdocstrings-python to locate source files.
doc_env = {"PYTHONPATH": "src"}


@session(venv_backend="none")
def docs(s: Session) -> None:
    s.run("mkdocs", "build", env=doc_env)


@session(venv_backend="none")
def docs_check_urls(s: Session) -> None:
    s.run("mkdocs", "build", env=doc_env | {"HTMLPROOFER_VALIDATE_EXTERNAL_URLS": str(True)})


@session(venv_backend="none")
def docs_offline(s: Session) -> None:
    s.run("mkdocs", "build", env=doc_env | {"MKDOCS_MATERIAL_OFFLINE": str(True)})


@session(venv_backend="none")
def docs_serve(s: Session) -> None:
    s.run("mkdocs", "serve", env=doc_env)


@session(venv_backend="none")
def docs_github_pages(s: Session) -> None:
    s.run("mkdocs", "gh-deploy", "--force", env=doc_env)


@session(venv_backend="none")
def typos(s: Session) -> None:
    s.run("typos", "-c", ".github/_typos.toml")
