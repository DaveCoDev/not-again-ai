from pathlib import Path
from typing import Any

import pytest

from not_again_ai.llm.openai_api.chat_completion import chat_completion
from not_again_ai.llm.openai_api.openai_client import openai_client
from not_again_ai.llm.openai_api.prompts import encode_image

image_dir = Path(__file__).parent.parent / "sample_images"
cat_image = image_dir / "cat.jpg"
dog_image = image_dir / "dog.jpg"
numbers_image = image_dir / "numbers.png"
sk_infographic = image_dir / "SKInfographic.png"
sk_diagram = image_dir / "SKDiagram.png"


def test_chat_completion() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(messages=messages, model="gpt-4o-2024-05-13", max_tokens=100, client=client)
    print(response)


def test_chat_completion_length() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(messages=messages, model="gpt-3.5-turbo-0125", max_tokens=2, client=client)
    print(response)


def test_chat_completion_expected_function() -> None:
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
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-3.5-turbo-0125",
        client=client,
        tools=tools,
        max_tokens=300,
        temperature=0,
    )
    print(response)


def test_chat_completion_tool_choice() -> None:
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
            "content": "What's the current weather like in Boston, MA today?",
        }
    ]
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-3.5-turbo-0125",
        client=client,
        tools=tools,
        tool_choice="get_current_weather",
        max_tokens=300,
        temperature=0,
    )
    print(response)


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
        model="gpt-3.5-turbo-0125",
        client=client,
        tools=tools,
        max_tokens=400,
        temperature=0,
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_dont_call_function() -> None:
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
        model="gpt-3.5-turbo-0125",
        client=client,
        tools=tools,
        tool_choice="none",
        max_tokens=200,
        temperature=0,
    )
    print(response)


def test_json_mode() -> None:
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
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-3.5-turbo-0125",
        client=client,
        max_tokens=300,
        temperature=0,
        json_mode=True,
    )
    print(response)


def test_chat_completion_n() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(messages=messages, model="gpt-3.5-turbo-0125", max_tokens=100, client=client, n=2)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_seed() -> None:
    client = openai_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate a random number between 0 and 100."},
    ]
    response_1 = chat_completion(
        messages=messages, model="gpt-3.5-turbo-0125", max_tokens=100, client=client, temperature=2, seed=42
    )

    response_2 = chat_completion(
        messages=messages, model="gpt-3.5-turbo-0125", max_tokens=100, client=client, temperature=2, seed=42
    )

    print(response_1)
    print(response_1["message"])
    print(response_2["message"])

    assert "system_fingerprint" in response_1

    # Add this assertion when outputs become guaranteed to be the same. It currently fails roughly 1 in 5 times.
    # if response_1["system_fingerprint"] == response_2["system_fingerprint"]:
    #   assert response_1["message"] == response_2["message"]


def test_chat_completion_logprobs() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(
        messages=messages, model="gpt-3.5-turbo-0125", max_tokens=100, client=client, logprobs=(True, None)
    )
    print(response)


def test_chat_completion_toplogprobs() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(
        messages=messages, model="gpt-3.5-turbo-0125", max_tokens=100, client=client, logprobs=(True, 3)
    )
    print(response)


def test_chat_completion_misc_1() -> None:
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
            "role": "user",
            "content": "What's the current weather like in Boston, MA today? Call the get_current_weather function.",
        }
    ]
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-3.5-turbo-0125",
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


@pytest.mark.skip("API Cost")
def test_chat_completion_misc_2() -> None:
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
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-4-turbo-2024-04-09",
        client=client,
        max_tokens=200,
        temperature=0,
        json_mode=True,
        logprobs=(True, None),
        seed=-5,
        n=2,
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_message_with_tools() -> None:
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
            "content": """You will be given a function called get_current_weather.
Before calling the function, first reason about which city the user is asking about. YOU MUST think step by step before calling the function.
For example, if the user asks 'What's the current weather like in Boston, MA today?', You should first say 'The user is asking about Boston, MA so I will call the function with 'Boston, MA' <now call the function>""",
        },
        {
            "role": "user",
            "content": "What's the current weather like in Boston, MA today? First think step by step as to which city and state to call, only then call the get_current_weather function. ",
        },
    ]

    response = chat_completion(
        messages=messages,
        model="gpt-4-turbo-2024-04-09",
        client=client,
        tools=tools,
        max_tokens=600,
        temperature=0.7,
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the animal in the image in one word."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(cat_image)}", "detail": "low"},
                },
            ],
        },
    ]

    response = chat_completion(
        messages=messages, model="gpt-4o-2024-05-13", max_tokens=200, temperature=0.5, client=client
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_length() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the animal in the image in one sentence."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(cat_image)}", "detail": "low"},
                },
            ],
        },
    ]
    response = chat_completion(messages=messages, model="gpt-4-turbo-2024-04-09", max_tokens=2, client=client)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_n() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the animal in the image in one word."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(cat_image)}", "detail": "low"},
                },
            ],
        },
    ]
    response = chat_completion(messages=messages, model="gpt-4-turbo-2024-04-09", max_tokens=200, client=client, n=2)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_seed() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Pick one random number that is written in the image. Just write the number."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(numbers_image)}", "detail": "low"},
                },
            ],
        },
    ]
    response_1 = chat_completion(
        messages=messages, model="gpt-4o-2024-05-13", max_tokens=200, client=client, temperature=2, seed=42
    )

    response_2 = chat_completion(
        messages=messages, model="gpt-4o-2024-05-13", max_tokens=200, client=client, temperature=2, seed=42
    )

    print(response_1)
    print(response_1["message"])
    print(response_2["message"])
    assert response_1["message"] == response_2["message"]


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_multiple_images() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are the animals in the images? Reply in one word for each animal."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(cat_image)}", "detail": "low"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(dog_image)}", "detail": "low"},
                },
            ],
        },
    ]

    response = chat_completion(messages=messages, model="gpt-4o-2024-05-13", max_tokens=200, client=client)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_multiple_messages() -> None:
    """Test with two image messages separated by an assistant message."""
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the animal in the image in one word."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(cat_image)}", "detail": "low"},
                },
            ],
        },
        {"role": "assistant", "content": "Cat"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What about this animal?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(dog_image)}", "detail": "low"},
                },
            ],
        },
    ]
    response = chat_completion(messages=messages, model="gpt-4o-2024-05-13", max_tokens=200, client=client)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_different_fidelity() -> None:
    """Test sending one image with low fidelity and another with high fidelity."""
    client = openai_client()
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
    response = chat_completion(messages=messages, model="gpt-4-turbo-2024-04-09", max_tokens=200, client=client)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_tool_call() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": """You are detecting if there is text (numbers or letters) in images. 
If you see any text, call the ocr tool. It takes no parameters.""",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(numbers_image)}", "detail": "low"},
                }
            ],
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ocr",
                "description": "Perform Optical Character Recognition (OCR) on an image",
                "parameters": {},
            },
        },
    ]
    response = chat_completion(messages=messages, model="gpt-4o-2024-05-13", client=client, tools=tools, max_tokens=200)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_json_mode() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": """You are performing optical character recognition (OCR) on an image. 
Return the recognized text in JSON format where the detected text is the value of the 'text' key.""",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(numbers_image)}", "detail": "low"},
                }
            ],
        },
    ]
    response = chat_completion(
        messages=messages, model="gpt-4o-2024-05-13", client=client, max_tokens=300, json_mode=True
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_many_features() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the animal in the image in one word."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(cat_image)}", "detail": "low"},
                },
            ],
        },
    ]
    response = chat_completion(
        messages=messages,
        model="gpt-4-turbo-2024-04-09",
        client=client,
        temperature=2,
        max_tokens=200,
        n=2,
        logprobs=(True, 2),
        seed=21,
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_complete_required_tool_call() -> None:
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
            "content": """You are a helpful assistant who listens to the user.""",
        },
        {
            "role": "user",
            "content": "Do not call any tools!",
        },
    ]

    response = chat_completion(
        messages=messages,
        model="gpt-4o-2024-05-13",
        client=client,
        tools=tools,
        tool_choice="required",
        max_tokens=400,
        temperature=0.5,
    )
    print(response)


if __name__ == "__main__":
    test_chat_completion()
