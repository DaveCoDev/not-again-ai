import os

from openai import OpenAI
import pytest

from not_again_ai.llm.oai_client import InvalidOAIAPITypeError, oai_client


def test_oai_client_default_type() -> None:
    # Assuming the OpenAI client will not raise an error even if no real API key is given.
    client = oai_client()
    assert isinstance(client, OpenAI)


def test_oai_client_os_vars() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    organization = os.environ.get("OPENAI_ORGANIZATION")

    client = oai_client(api_key=api_key, organization=organization)
    assert isinstance(client, OpenAI)


def test_oai_client_invalid_type() -> None:
    with pytest.raises(InvalidOAIAPITypeError):
        oai_client("invalid_type")


def test_oai_client_azure_openai() -> None:
    with pytest.raises(NotImplementedError):
        oai_client("azure_openai")


def test_oai_client_openai_type_with_parameters() -> None:
    client = oai_client(
        api_type="openai",
        api_key="example_api_key",
        organization="example_org",
        timeout=5.0,
        max_retries=3,
    )
    assert isinstance(client, OpenAI)
