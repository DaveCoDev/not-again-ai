from ollama import Client

from not_again_ai.local_llm.ollama.ollama_client import ollama_client


def test_client() -> None:
    # Assumes that the OLLAMA_HOST environment variable is set.
    client = ollama_client()
    assert isinstance(client, Client)


def test_client_with_host() -> None:
    client = ollama_client(host="http://localhost:11434")
    assert isinstance(client, Client)


def test_client_with_timeout() -> None:
    client = ollama_client(timeout=5.0)
    assert isinstance(client, Client)


def test_client_all_parameters() -> None:
    client = ollama_client(
        host="http://localhost:11434",
        timeout=5.0,
    )
    assert isinstance(client, Client)
