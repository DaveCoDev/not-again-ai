from ollama import Client

from not_again_ai.llm.ollama.client import client


def test_client() -> None:
    # Assumes that the OLLAMA_HOST environment variable is set.
    ollama_client = client()
    assert isinstance(ollama_client, Client)


def test_client_with_host() -> None:
    ollama_client = client(host="http://localhost:11434")
    assert isinstance(ollama_client, Client)


def test_client_with_timeout() -> None:
    ollama_client = client(timeout=5.0)
    assert isinstance(ollama_client, Client)


def test_client_all_parameters() -> None:
    ollama_client = client(
        host="http://localhost:11434",
        timeout=5.0,
    )
    assert isinstance(ollama_client, Client)
