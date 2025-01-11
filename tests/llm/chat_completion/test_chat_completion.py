from collections.abc import Callable
import os
from pathlib import Path
from typing import Any

import pytest

from not_again_ai.llm.chat_completion import chat_completion
from not_again_ai.llm.chat_completion.providers.openai_api import openai_client
from not_again_ai.llm.chat_completion.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ImageContent,
    ImageDetail,
    ImageUrl,
    MessageT,
    SystemMessage,
    TextContent,
    UserMessage,
)
from not_again_ai.llm.openai_api.prompts import encode_image

image_dir = Path(__file__).parent.parent / "sample_images"
cat_image = image_dir / "cat.jpg"
dog_image = image_dir / "dog.jpg"
numbers_image = image_dir / "numbers.png"
sk_infographic = image_dir / "SKInfographic.png"
sk_diagram = image_dir / "SKDiagram.png"


@pytest.fixture(
    params=[
        {},
        {"api_type": "azure_openai", "aoai_api_version": "2024-10-01-preview"},
    ]
)
def client(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return openai_client(**request.param)


# region General Chat Completion
def test_chat_completion_simple(client: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            SystemMessage(content="Hello, world!"),
            UserMessage(content="What is the capital of France?"),
        ],
        max_completion_tokens=100,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_length(client: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-2024-11-20",
        messages=[
            SystemMessage(content="Hello, world!"),
            UserMessage(content="What is the capital of France and the capital of Germany?"),
        ],
        max_completion_tokens=2,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_json_mode(client: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        SystemMessage(
            content="""You are getting names of users and formatting them into json.
Example:
User: Jane Doe
Output: {"name": "Jane Doe"}"""
        ),
        UserMessage(content="John Doe"),
    ]
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        max_completion_tokens=300,
        temperature=0,
        json_mode=True,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_n(client: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello!"),
        ],
        max_completion_tokens=100,
        n=2,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_seed(client: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Generate a random number between 0 and 100."),
        ],
        max_completion_tokens=100,
        temperature=2,
        seed=42,
    )

    response_1 = chat_completion(request, "openai", client)
    response_2 = chat_completion(request, "openai", client)

    print(response_1.choices[0].message.content)
    print(response_2.choices[0].message.content)


def test_chat_completion_logprobs(client: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello!"),
        ],
        max_completion_tokens=100,
        logprobs=True,
        top_logprobs=None,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_toplogprobs(client: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello!"),
        ],
        max_completion_tokens=100,
        logprobs=True,
        top_logprobs=3,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_tool_simple(client: Callable[..., Any]) -> None:
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
        }
    ]

    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            UserMessage(
                content="What's the current weather like in Boston, MA today? Call the get_current_weather function."
            )
        ],
        tools=tools,
        max_completion_tokens=300,
        temperature=0,
    )

    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_tool_required_name(client: Callable[..., Any]) -> None:
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
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            UserMessage(content="What's the current weather like in Boston, MA today?"),
        ],
        tools=tools,
        tool_choice="get_current_weather",
        max_completion_tokens=300,
        temperature=0,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_required_tool_call(client: Callable[..., Any]) -> None:
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
    messages: list[MessageT] = [
        SystemMessage(
            content="You are a helpful assistant who always listens to the user and does not call any tools."
        ),
        UserMessage(content="Do not call any tools!"),
    ]

    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        tools=tools,
        tool_choice="required",
        max_completion_tokens=400,
        temperature=0.5,
    )

    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_multiple_tools(client: Callable[..., Any]) -> None:
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
    messages: list[MessageT] = [
        SystemMessage(content="Call the get_current_weather function once for each city that the user mentions."),
        UserMessage(content="What's the current weather like in Boston, MA and New York, NY today?"),
    ]
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        tools=tools,
        max_completion_tokens=400,
        temperature=0,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_tool_choice_none(client: Callable[..., Any]) -> None:
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
    messages: list[MessageT] = [
        SystemMessage(content="Call the get_current_weather function once for each city that the user mentions."),
        UserMessage(content="What's the current weather like in Boston, MA and New York, NY today?"),
    ]
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        tools=tools,
        tool_choice="none",
        max_completion_tokens=400,
        temperature=0,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_message_with_tools(client: Callable[..., Any]) -> None:
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
    messages: list[MessageT] = [
        SystemMessage(
            content="""You will be given a function called get_current_weather.
Before calling the function, first reason about which city the user is asking about. YOU MUST think step by step before calling the function.
For example, if the user asks 'What's the current weather like in Boston, MA today?', You should first say 'The user is asking about Boston, MA so I will call the function with 'Boston, MA' <now call the function>"""
        ),
        UserMessage(
            content="What's the current weather like in Boston, MA today? First think step by step as to which city and state to call, only then call the get_current_weather function. "
        ),
    ]

    request = ChatCompletionRequest(
        model="gpt-4o-2024-11-20",
        messages=messages,
        tools=tools,
        max_completion_tokens=600,
        temperature=0.7,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_structured_output(client: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        SystemMessage(content="You are a helpful assistant"),
        UserMessage(content="9.11 and 9.9 -- which is bigger?"),
    ]
    json_schema = {
        "name": "reasoning_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The reasoning steps leading to the final conclusion.",
                },
                "answer": {
                    "type": "string",
                    "description": "The final answer, taking into account the reasoning steps.",
                },
            },
            "required": ["reasoning_steps", "answer"],
            "additionalProperties": False,
        },
        "description": "A schema for structured output that includes reasoning steps and the final answer.",
    }

    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        max_completion_tokens=400,
        temperature=0.2,
        json_mode=False,
        structured_outputs=json_schema,
    )

    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_required_tools_none_called_structured(client: Callable[..., Any]) -> None:
    tools = [
        {
            "type": "function",
            "strict": True,
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
                    "additionalProperties": False,
                },
            },
        },
    ]
    messages: list[MessageT] = [
        SystemMessage(content="Do not call get_current_weather or any other tool under any circumstances."),
        UserMessage(content="What is 2+2?"),
    ]

    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        tools=tools,
        tool_choice="required",
        max_completion_tokens=300,
        temperature=0.5,
    )

    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision(client: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextContent(text="Describe the animal in the image in one word."),
                ImageContent(
                    image_url=ImageUrl(url=f"data:image/jpeg;base64,{encode_image(cat_image)}", detail=ImageDetail.LOW)
                ),
            ]
        ),
    ]

    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        max_completion_tokens=200,
        temperature=0.5,
    )

    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_n(client: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextContent(text="Describe the animal in the image in one word."),
                ImageContent(
                    image_url=ImageUrl(url=f"data:image/jpeg;base64,{encode_image(cat_image)}", detail=ImageDetail.LOW)
                ),
            ]
        ),
    ]
    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        max_completion_tokens=200,
        n=2,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_seed(client: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextContent(text="Pick one random number that is written in the image. Just write the number."),
                ImageContent(
                    image_url=ImageUrl(
                        url=f"data:image/png;base64,{encode_image(numbers_image)}", detail=ImageDetail.LOW
                    )
                ),
            ]
        ),
    ]
    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        max_completion_tokens=100,
        temperature=2,
        seed=42,
    )

    response_1 = chat_completion(request, "openai", client)
    response_2 = chat_completion(request, "openai", client)

    print(response_1.model_dump(mode="json", exclude_none=True))
    print(response_1.choices[0].message.content)
    print(response_2.choices[0].message.content)


def test_chat_completion_vision_multiple_images(client: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextContent(text="What are the animals in the images? Reply in one word for each animal."),
                ImageContent(
                    image_url=ImageUrl(url=f"data:image/jpeg;base64,{encode_image(cat_image)}", detail=ImageDetail.LOW)
                ),
                ImageContent(
                    image_url=ImageUrl(url=f"data:image/jpeg;base64,{encode_image(dog_image)}", detail=ImageDetail.LOW)
                ),
            ]
        ),
    ]
    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        max_completion_tokens=100,
    )
    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_multiple_messages(client: Callable[..., Any]) -> None:
    """Test with two image messages separated by an assistant message."""
    messages: list[MessageT] = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextContent(text="Describe the animal in the image in one word."),
                ImageContent(
                    image_url=ImageUrl(url=f"data:image/jpeg;base64,{encode_image(cat_image)}", detail=ImageDetail.LOW)
                ),
            ]
        ),
        AssistantMessage(content="Cat"),
        UserMessage(
            content=[
                TextContent(text="What about this animal?"),
                ImageContent(
                    image_url=ImageUrl(url=f"data:image/jpeg;base64,{encode_image(dog_image)}", detail=ImageDetail.LOW)
                ),
            ]
        ),
    ]

    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        max_completion_tokens=100,
    )

    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_different_fidelity(client: Callable[..., Any]) -> None:
    """Test sending one image with low fidelity and another with high fidelity."""
    messages: list[MessageT] = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextContent(
                    text="Based on these infographics, can you summarize how Semantic Kernel works in exactly one sentence?"
                ),
                ImageContent(
                    image_url=ImageUrl(
                        url=f"data:image/png;base64,{encode_image(sk_infographic)}", detail=ImageDetail.HIGH
                    )
                ),
                ImageContent(
                    image_url=ImageUrl(url=f"data:image/png;base64,{encode_image(sk_diagram)}", detail=ImageDetail.LOW)
                ),
            ]
        ),
    ]

    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        max_completion_tokens=200,
    )

    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_tool_call(client: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        SystemMessage(
            content="""You are detecting if there is text (numbers or letters) in images. 
If you see any text, call the ocr tool. It takes no parameters."""
        ),
        UserMessage(
            content=[
                ImageContent(
                    image_url=ImageUrl(
                        url=f"data:image/png;base64,{encode_image(numbers_image)}", detail=ImageDetail.LOW
                    )
                ),
            ]
        ),
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

    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        tools=tools,
        max_completion_tokens=200,
    )

    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_json_mode(client: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        SystemMessage(
            content="""You are performing optical character recognition (OCR) on an image. 
Return the recognized text in JSON format where the detected text is the value of the 'text' key."""
        ),
        UserMessage(
            content=[
                ImageContent(
                    image_url=ImageUrl(
                        url=f"data:image/png;base64,{encode_image(numbers_image)}", detail=ImageDetail.LOW
                    )
                ),
            ]
        ),
    ]

    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        max_completion_tokens=300,
        json_mode=True,
    )

    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_many_features(client: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        UserMessage(
            content=[
                TextContent(text="Describe the animal in the image in one word."),
                ImageContent(
                    image_url=ImageUrl(url=f"data:image/jpeg;base64,{encode_image(cat_image)}", detail=ImageDetail.LOW)
                ),
            ]
        ),
    ]

    request = ChatCompletionRequest(
        messages=messages,
        model="gpt-4o-2024-11-20",
        max_completion_tokens=100,
        temperature=2,
        n=2,
        logprobs=True,
        top_logprobs=2,
        seed=21,
    )

    response = chat_completion(request, "openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


# endregion


# region Azure OpenAI
def test_chat_completion_azure_key(client: Callable[..., Any]) -> None:
    client = openai_client(api_type="azure_openai", azure_key=os.environ["AZURE_OPENAI_KEY"])
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            SystemMessage(content="Hello, world!"),
            UserMessage(content="What is the capital of France?"),
        ],
        max_completion_tokens=100,
    )
    response = chat_completion(request, "azure_openai", client)
    print(response.model_dump(mode="json", exclude_none=True))


# endregion
