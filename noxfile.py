import os

import nox
from nox import Session, param, parametrize, session

nox.options.error_on_external_run = True
nox.options.default_venv_backend = "uv"
nox.options.sessions = [
    "lint",
    "type_check",
    "typos",
    "test",
]


@session(python=["3.11", "3.12"])
def test(s: Session) -> None:
    s.install(".[data,llm,statistics,viz]", "pytest", "pytest-asyncio", "pytest-cov", "pytest-randomly")
    s.run_install(
        "uv",
        "sync",
        "--locked",
        "--all-extras",
        env={"UV_PROJECT_ENVIRONMENT": s.virtualenv.location},
    )

    # Skip tests in directories specified by the SKIP_TESTS_NAII environment variable.
    skip_tests = os.getenv("SKIP_TESTS_NAAI", "")
    skip_tests += " tests/llm/chat_completion/ tests/llm/embedding/ tests/llm/image_gen/"
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


# For some sessions, set venv_backend="none" to simply execute scripts within the existing
# uv-generated virtual environment, rather than have nox create a new one for each session.
@session(venv_backend="none")
@parametrize(
    "command",
    [
        param(
            [
                "ruff",
                "check",
                ".",
                "--select",
                "I",
                # Also remove unused imports.
                "--select",
                "F401",
                "--extend-fixable",
                "F401",
                "--fix",
            ],
            id="sort_imports",
        ),
        param(["ruff", "format", "."], id="format"),
    ],
)
def fmt(s: Session, command: list[str]) -> None:
    s.run(*command)


@session(venv_backend="none")
@parametrize(
    "command",
    [
        param(["ruff", "check", "."], id="lint_check"),
        param(["ruff", "format", "--check", "."], id="format_check"),
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


@session(venv_backend="none")
def typos(s: Session) -> None:
    s.run("typos", "-c", ".github/_typos.toml")
