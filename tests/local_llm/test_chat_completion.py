import pytest

from not_again_ai.llm.gh_models.azure_ai_client import azure_ai_client
from not_again_ai.llm.openai_api.openai_client import openai_client
from not_again_ai.local_llm.chat_completion import chat_completion
from not_again_ai.local_llm.ollama.ollama_client import ollama_client


def test_chat_completion_ollama() -> None:
    client = ollama_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    response = chat_completion(messages, model="phi3:3.8b-mini-4k-instruct-q8_0", client=client)
    print(response)


def test_chat_completion_ollama_tools() -> None:
    client = ollama_client()
    messages = [{"role": "user", "content": "What is the flight time from New York (NYC) to Los Angeles (LAX)?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_flight_times",
                "description": "Get the flight times between two cities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "departure": {
                            "type": "string",
                            "description": "The departure city (airport code)",
                        },
                        "arrival": {
                            "type": "string",
                            "description": "The arrival city (airport code)",
                        },
                    },
                    "required": ["departure", "arrival"],
                },
            },
        },
    ]

    response = chat_completion(
        messages,
        model="llama3.1:8b-instruct-q4_0",
        client=client,
        tools=tools,
        temperature=0,
    )
    print(response)


def test_chat_completion_openai() -> None:
    client = openai_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    response = chat_completion(messages, model="gpt-4o-mini-2024-07-18", client=client)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_multiple_functions() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "system",
            "content": "Call the get_current_weather function once for each city that the user mentions.",
        },
        {
            "role": "user",
            "content": "What's the current weather like in Boston, MA and New York, NY today?",
        },
    ]
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-4o-mini-2024-07-18",
        client=client,
        tools=tools,
        max_tokens=400,
        temperature=0,
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_gh() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "system",
            "content": "Call the get_current_weather function once for each city that the user mentions.",
        },
        {
            "role": "user",
            "content": "What's the current weather like in Boston, MA and New York, NY today?",
        },
    ]
    client = azure_ai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-4o-mini",
        client=client,
        tools=tools,
        max_tokens=400,
        temperature=0,
    )
    print(response)


if __name__ == "__main__":
    test_chat_completion_gh()
    test_chat_completion_multiple_functions()
    test_chat_completion_ollama_tools()
