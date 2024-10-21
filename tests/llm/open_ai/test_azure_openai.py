from pathlib import Path
from typing import Any

from openai import AzureOpenAI
import pytest

from not_again_ai.llm.openai_api.chat_completion import chat_completion, chat_completion_stream
from not_again_ai.llm.openai_api.openai_client import openai_client
from not_again_ai.llm.openai_api.prompts import encode_image

image_dir = Path(__file__).parent.parent / "sample_images"
cat_image = image_dir / "cat.jpg"
dog_image = image_dir / "dog.jpg"
numbers_image = image_dir / "numbers.png"
sk_infographic = image_dir / "SKInfographic.png"
sk_diagram = image_dir / "SKDiagram.png"


def test_azure_openai() -> None:
    client = openai_client(api_type="azure_openai")
    assert isinstance(client, AzureOpenAI)


def test_aoai_chat_completion() -> None:
    client = openai_client(api_type="azure_openai")
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(messages=messages, model="gpt-4o-mini-2024-07-18", max_tokens=15, client=client)
    print(response)


def test_chat_completion_vision_different_fidelity() -> None:
    """Test sending one image with low fidelity and another with high fidelity."""
    client = openai_client(api_type="azure_openai")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Based on these infographics, can you summarize how Semantic Kernel works in exactly one sentence?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(sk_infographic)}", "detail": "high"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(sk_diagram)}", "detail": "low"},
                },
            ],
        },
    ]
    response = chat_completion(messages=messages, model="gpt-4o-2024-05-13", max_tokens=200, client=client)
    print(response)


def test_chat_completion_misc_1() -> None:
    client = openai_client(api_type="azure_openai")
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
            "role": "user",
            "content": "What's the current weather like in Boston, MA today? Call the get_current_weather function.",
        }
    ]
    client = openai_client(api_type="azure_openai")
    response = chat_completion(
        messages=messages,
        model="gpt-4o-mini-2024-07-18",
        client=client,
        tools=tools,
        max_tokens=300,
        temperature=0,
        logprobs=(True, 2),
        seed=42,
        n=2,
    )
    # NOTE: When a function is called, logprobs are not returned.
    print(response)


def test_chat_completion_stream_simple() -> None:
    client = openai_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = chat_completion_stream(
        messages=messages,
        model="gpt-4o-mini-2024-07-18",
        client=client,
        max_tokens=3,
        temperature=0.7,
        seed=42,
    )
    for chunk in response:
        print(chunk)


@pytest.mark.skip("API Cost")
def test_chat_completion_stream_multiple_functions() -> None:
    client = openai_client()
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
    response = chat_completion_stream(
        messages=messages,
        model="gpt-4o-mini-2024-07-18",
        client=client,
        tools=tools,
        max_tokens=400,
        temperature=0,
    )
    for chunk in response:
        print(chunk)


@pytest.mark.skip("API Cost")
def test_chat_completion_stream_message_with_tools() -> None:
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
            "content": "First greet the user by saying 'Hello!'. Then call the get_current_weather function for each city the user mentions. Finally, say goodbye to the user.",
        },
        {
            "role": "user",
            "content": "What is the weather like in Orlando, FL?",
        },
    ]
    client = openai_client()
    response = chat_completion_stream(
        messages=messages,
        model="gpt-4o-2024-08-06",
        client=client,
        tools=tools,
        tool_choice="auto",
        max_tokens=500,
        temperature=0.5,
    )
    for chunk in response:
        print(chunk)


if __name__ == "__main__":
    test_azure_openai()
    test_aoai_chat_completion()
    test_chat_completion_vision_different_fidelity()
    test_chat_completion_misc_1()
    test_chat_completion_stream_simple()
    test_chat_completion_stream_multiple_functions()
    test_chat_completion_stream_message_with_tools()
