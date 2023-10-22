import nox
from nox_poetry import Session, session

nox.options.error_on_external_run = True
nox.options.reuse_existing_virtualenvs = False
nox.options.sessions = ["fmt_check", "lint", "type_check", "test", "docs"]


@session(python=["3.10", "3.11", "3.12"])
def test(s: Session) -> None:
    s.install(".[statistics,viz]", "pytest", "pytest-cov", "pytest-randomly")
    s.run(
        "python",
        "-m",
        "pytest",
        "--cov=not_again_ai",
        "--cov-report=html",
        "--cov-report=term",
        "tests",
        "-W ignore::DeprecationWarning",
        *s.posargs,
    )


# For some sessions, set venv_backend="none" to simply execute scripts within the existing Poetry
# environment. This requires that nox is run within `poetry shell` or using `poetry run nox ...`.
@session(venv_backend="none")
def fmt(s: Session) -> None:
    s.run("ruff", "check", ".", "--select", "I", "--fix")
    s.run("black", ".")


@session(venv_backend="none")
def fmt_check(s: Session) -> None:
    s.run("ruff", "check", ".", "--select", "I")
    s.run("black", "--check", ".")


@session(venv_backend="none")
def lint(s: Session) -> None:
    s.run("ruff", "check", ".")


@session(venv_backend="none")
def lint_fix(s: Session) -> None:
    s.run("ruff", "check", ".", "--fix")


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
    s.run("mkdocs", "build", env={**doc_env, **{"HTMLPROOFER_VALIDATE_EXTERNAL_URLS": str(True)}})


@session(venv_backend="none")
def docs_offline(s: Session) -> None:
    s.run("mkdocs", "build", env={**doc_env, **{"MKDOCS_MATERIAL_OFFLINE": str(True)}})


@session(venv_backend="none")
def docs_serve(s: Session) -> None:
    s.run("mkdocs", "serve", env=doc_env)


@session(venv_backend="none")
def docs_github_pages(s: Session) -> None:
    s.run("mkdocs", "gh-deploy", "--force", env=doc_env)
