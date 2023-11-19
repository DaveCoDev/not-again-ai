from not_again_ai.llm.chat_completion import chat_completion
from not_again_ai.llm.openai_client import openai_client


def test_chat_completion() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(messages=messages, model="gpt-3.5-turbo", max_tokens=100, client=client)
    print(response)


def test_chat_completion_length() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(messages=messages, model="gpt-3.5-turbo", max_tokens=2, client=client)
    print(response)


def test_chat_completion_expected_function() -> None:
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
    client = openai_client()
    response = chat_completion(
        messages=messages, model="gpt-3.5-turbo", client=client, functions=functions, max_tokens=200, temperature=0.5
    )
    print(response)
