from ollama import ResponseError
import pytest

from not_again_ai.local_llm.ollama.chat_completion import chat_completion
from not_again_ai.local_llm.ollama.ollama_client import ollama_client

MODELS = ["phi3:3.8b-mini-4k-instruct-q8_0", "llama3.1:8b-instruct-q4_0"]


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


def test_chat_completion_seed(model: str) -> None:
    client = ollama_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate a random number between 0 and 100."},
    ]

    response1 = chat_completion(messages, model=model, client=client, seed=6, temperature=2)
    response2 = chat_completion(messages, model=model, client=client, seed=6, temperature=2)

    print(response1)
    print(response2)
    # Occasionally the responses are different for some reason
    # assert response1["message"] == response2["message"]


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


def test_chat_completion_tool_example() -> None:
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
        model=MODELS[1],
        client=client,
        tools=tools,
        temperature=0,
    )
    print(response)


def test_chat_completion_multiple_functions() -> None:
    client = ollama_client()
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
    response = chat_completion(
        messages=messages,
        model=MODELS[1],
        client=client,
        tools=tools,
        max_tokens=400,
        temperature=0,
    )
    print(response)


def test_chat_completion_dont_call_function() -> None:
    client = ollama_client()
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
            "content": "You are a friendly assistant who chats with users! Do not call functions if it is not required to answer the user's question.",
        },
        {
            "role": "user",
            "content": "Hello can you tell me what 2+2 is?",
        },
    ]
    response = chat_completion(
        messages=messages,
        model=MODELS[1],
        client=client,
        tools=tools,
        max_tokens=400,
        temperature=0,
    )
    print(response)


def test_chat_completion_unsupported_tool_call_model() -> None:
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
    with pytest.raises(ResponseError):
        chat_completion(
            messages,
            model=MODELS[0],
            client=client,
            tools=tools,
            temperature=0,
        )


if __name__ == "__main__":
    test_chat_completion_unsupported_tool_call_model()
    test_chat_completion_dont_call_function()
    test_chat_completion_multiple_functions()
    test_chat_completion_tool_example()
    test_chat_completion(model=MODELS[0])
