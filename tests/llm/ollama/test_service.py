import pytest

from not_again_ai.llm.ollama.client import client
from not_again_ai.llm.ollama.service import delete, is_model_available, list_models, pull, show

MODEL = "phi3"


def test_list_models() -> None:
    ollama_client = client()
    models = list_models(ollama_client)
    print(models)


def test_is_model_available() -> None:
    ollama_client = client()
    response = is_model_available(MODEL, ollama_client)
    assert response is True


def test_show() -> None:
    ollama_client = client()
    response = show(MODEL, ollama_client)
    print(response)


@pytest.mark.skip(reason="Time consuming")
def test_delete_pull() -> None:
    ollama_client = client()
    response = delete(MODEL, ollama_client)
    print(response)
    response = pull(MODEL, ollama_client)
    print(response)
