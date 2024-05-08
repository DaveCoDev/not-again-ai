from ollama import ResponseError
import pytest

from not_again_ai.llm.ollama.chat_completion import chat_completion
from not_again_ai.llm.ollama.ollama_client import ollama_client

MODELS = ["phi3", "llama3:8b"]


@pytest.fixture(params=MODELS)
def model(request):  # type: ignore
    return request.param


def test_chat_completion(model: str) -> None:
    client = ollama_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    response = chat_completion(messages, model=model, client=client)
    print(response)


def test_chat_completion_max_tokens(model: str) -> None:
    client = ollama_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    response = chat_completion(messages, model=model, client=client, max_tokens=2)
    print(response)


def test_chat_completion_context_window(model: str) -> None:
    client = ollama_client()
    messages = [
        {"role": "user", "content": "Orange, kiwi, watermelon. List the three fruits I just named."},
    ]

    response = chat_completion(messages, model=model, client=client, context_window=1, max_tokens=200)
    print(response)


def test_chat_completion_json_mode(model: str) -> None:
    client = ollama_client()
    messages = [
        {
            "role": "system",
            "content": """You are getting names of users and formatting them into json.
Example:
User: Jane Doe
Output: {"name": "Jane Doe"}""",
        },
        {
            "role": "user",
            "content": "John Doe",
        },
    ]

    response = chat_completion(messages, model=model, client=client, json_mode=True, max_tokens=200)
    print(response)


test_chat_completion_json_mode(MODELS[1])


def test_chat_completion_seed(model: str) -> None:
    client = ollama_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate a random number between 0 and 100."},
    ]

    response1 = chat_completion(messages, model=model, client=client, seed=6, temperature=2)
    response2 = chat_completion(messages, model=model, client=client, seed=6, temperature=2)

    assert response1["message"] == response2["message"]


def test_chat_completion_all(model: str) -> None:
    client = ollama_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Generate a random number between 0 and 100 and structure the response in using JSON.",
        },
    ]

    response = chat_completion(
        messages,
        model=model,
        client=client,
        max_tokens=300,
        context_window=1000,
        temperature=1.51,
        json_mode=True,
        seed=6,
    )
    print(response)


def test_chat_completion_model_not_found() -> None:
    client = ollama_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    with pytest.raises(ResponseError):
        chat_completion(messages, model="notamodel", client=client)
