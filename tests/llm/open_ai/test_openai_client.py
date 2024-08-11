import os

from openai import OpenAI
import pytest

from not_again_ai.llm.openai_api.openai_client import InvalidOAIAPITypeError, openai_client


def test_openai_client_client_default_type() -> None:
    # Assuming the OpenAI client will not raise an error even if no real API key is given.
    client = openai_client()
    assert isinstance(client, OpenAI)


def test_openai_client_os_vars() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    organization = os.environ.get("OPENAI_ORGANIZATION")

    client = openai_client(api_key=api_key, organization=organization)
    assert isinstance(client, OpenAI)


def test_openai_client_invalid_type() -> None:
    with pytest.raises(InvalidOAIAPITypeError):
        openai_client("invalid_type")  # type: ignore


def test_openai_client_openai_type_with_parameters() -> None:
    client = openai_client(
        api_type="openai",
        api_key="example_api_key",
        organization="example_org",
        timeout=5.0,
        max_retries=3,
    )
    assert isinstance(client, OpenAI)
