from not_again_ai.llm.chat_completion import chat_completion
from not_again_ai.llm.ollama.ollama_client import ollama_client
from not_again_ai.llm.openai_api.openai_client import openai_client


def test_chat_completion_ollama() -> None:
    client = ollama_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    response = chat_completion(messages, model="phi3", client=client)
    print(response)


def test_chat_completion_openai() -> None:
    client = openai_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    response = chat_completion(messages, model="gpt-3.5-turbo", client=client)
    print(response)
