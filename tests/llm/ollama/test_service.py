import pytest

from not_again_ai.llm.ollama.ollama_client import ollama_client
from not_again_ai.llm.ollama.service import delete, is_model_available, list_models, pull, show

MODEL = "phi3"


def test_list_models() -> None:
    client = ollama_client()
    models = list_models(client)
    print(models)


def test_is_model_available() -> None:
    client = ollama_client()
    response = is_model_available(MODEL, client)
    assert response is True


def test_show() -> None:
    client = ollama_client()
    response = show(MODEL, client)
    print(response)


@pytest.mark.skip(reason="Time consuming")
def test_delete_pull() -> None:
    client = ollama_client()
    response = delete(MODEL, client)
    print(response)
    response = pull(MODEL, client)
    print(response)
