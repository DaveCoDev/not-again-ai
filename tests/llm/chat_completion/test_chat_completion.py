from collections.abc import Callable
import os
from pathlib import Path
from typing import Any

import pytest

from not_again_ai.llm.chat_completion import chat_completion
from not_again_ai.llm.chat_completion.providers.anthropic_api import anthropic_client
from not_again_ai.llm.chat_completion.providers.gemini_api import gemini_client
from not_again_ai.llm.chat_completion.providers.ollama_api import ollama_client
from not_again_ai.llm.chat_completion.providers.openai_api import openai_client
from not_again_ai.llm.chat_completion.types import (
    AssistantMessage,
    ChatCompletionRequest,
    Function,
    ImageContent,
    ImageDetail,
    ImageUrl,
    MessageT,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from not_again_ai.llm.prompting.compile_prompt import encode_image

image_dir = Path(__file__).parent.parent / "sample_images"
cat_image = image_dir / "cat.jpg"
dog_image = image_dir / "dog.jpg"
numbers_image = image_dir / "numbers.png"
sk_infographic = image_dir / "SKInfographic.png"
sk_diagram = image_dir / "SKDiagram.png"


# region OpenAI and Azure OpenAI Chat Completion
@pytest.fixture(
    params=[
        {},
        {"api_type": "azure_openai", "aoai_api_version": "2025-01-01-preview"},
    ]
)
def openai_aoai_client_fixture(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return openai_client(**request.param)


def test_chat_completion_simple(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            SystemMessage(content="Hello, world!"),
            UserMessage(content="What is the capital of France?"),
        ],
        max_completion_tokens=100,
    )
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_length(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4.1-2025-04-14",
        messages=[
            SystemMessage(content="Hello, world!"),
            UserMessage(content="What is the capital of France and the capital of Germany?"),
        ],
        max_completion_tokens=2,
    )
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_json_mode(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_n(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello!"),
        ],
        max_completion_tokens=100,
        n=2,
    )
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_seed(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response_1 = chat_completion(request, "openai", openai_aoai_client_fixture)
    response_2 = chat_completion(request, "openai", openai_aoai_client_fixture)

    print(response_1.choices[0].message.content)
    print(response_2.choices[0].message.content)


def test_chat_completion_logprobs(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_toplogprobs(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_tool_simple(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_tool_required_name(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_required_tool_call(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_multiple_tools(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_tool_choice_none(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_message_with_tools(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_tool_message(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        SystemMessage(content="You are a helpful assistant"),
        UserMessage(content="What is the weather in Boston, MA?"),
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="abc123",
                    function=Function(
                        name="get_current_weather",
                        arguments={"location": "Boston, MA"},
                    ),
                )
            ],
        ),
        ToolMessage(name="abc123", content="The weather in Boston, MA is 70 degrees Fahrenheit."),
    ]
    request = ChatCompletionRequest(
        model="gpt-4o-2024-11-20",
        messages=messages,
        max_completion_tokens=300,
        temperature=0.3,
    )
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_structured_output(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_required_tools_none_called_structured(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_n(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_seed(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response_1 = chat_completion(request, "openai", openai_aoai_client_fixture)
    response_2 = chat_completion(request, "openai", openai_aoai_client_fixture)

    print(response_1.model_dump(mode="json", exclude_none=True))
    print(response_1.choices[0].message.content)
    print(response_2.choices[0].message.content)


def test_chat_completion_vision_multiple_images(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_multiple_messages(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_different_fidelity(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_tool_call(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_json_mode(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_vision_many_features(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_invalid_params(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-2024-11-20",
        messages=[UserMessage(content="What is the capital of France?")],
        max_completion_tokens=100,
        context_window=1000,
        mirostat=1,
    )
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_max_tokens(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[UserMessage(content="What is the capital of France?")],
        max_tokens=100,
    )
    response = chat_completion(request, "openai", openai_aoai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


# region OpenAI
@pytest.fixture(
    params=[
        {},
    ]
)
def openai_client_fixture(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return openai_client(**request.param)


def test_chat_completion_o4_mini(openai_client_fixture: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        UserMessage(
            content="You only think for one sentence, and then write at most a one sentence response. What is the capital of France?"
        ),
    ]
    request = ChatCompletionRequest(
        model="o4-mini-2025-04-16",
        messages=messages,
        max_completion_tokens=500,
        reasoning_effort="low",
    )
    response = chat_completion(request, "openai", openai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


# endregion


# region Azure OpenAI
@pytest.fixture(
    params=[
        {"api_type": "azure_openai", "aoai_api_version": "2024-10-01-preview"},
    ]
)
def azure_openai_client_fixture(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return openai_client(**request.param)


def test_chat_completion_azure_key(azure_openai_client_fixture: Callable[..., Any]) -> None:
    azure_openai_client_fixture = openai_client(api_type="azure_openai", api_key=os.environ["AZURE_OPENAI_KEY"])
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            SystemMessage(content="Hello, world!"),
            UserMessage(content="What is the capital of France?"),
        ],
        max_completion_tokens=100,
    )
    response = chat_completion(request, "azure_openai", azure_openai_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


# endregion


# region Ollama


@pytest.fixture(
    params=[
        {},
    ]
)
def ollama_client_fixture(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return ollama_client(**request.param)


def test_chat_completion_ollama(ollama_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="llama3.2-vision:11b-instruct-q4_K_M",
        messages=[
            SystemMessage(content="Hello, world!"),
            UserMessage(content="What is the capital of France?"),
        ],
        max_completion_tokens=100,
        frequency_penalty=1.2,
        top_p=0.8,
        context_window=1000,
    )
    response = chat_completion(request, "ollama", ollama_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_ollama_structured_output(ollama_client_fixture: Callable[..., Any]) -> None:
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
        model="llama3.2-vision:11b-instruct-q4_K_M",
        max_completion_tokens=400,
        temperature=0.2,
        json_mode=False,
        structured_outputs=json_schema,
    )

    response = chat_completion(request, "ollama", ollama_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_ollama_multiple_tools(ollama_client_fixture: Callable[..., Any]) -> None:
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
        model="qwen2.5:14b",
        messages=messages,
        tools=tools,
        max_completion_tokens=400,
        temperature=0,
    )
    response = chat_completion(request, "ollama", ollama_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_ollama_tool_message(ollama_client_fixture: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        SystemMessage(content="You are a helpful assistant"),
        UserMessage(content="What is the weather in Boston, MA?"),
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="abc123",
                    function=Function(
                        name="get_current_weather",
                        arguments={"location": "Boston, MA"},
                    ),
                )
            ],
        ),
        ToolMessage(name="abc123", content="The weather in Boston, MA is 70 degrees Fahrenheit."),
    ]
    request = ChatCompletionRequest(
        model="qwen2.5:14b",
        messages=messages,
        max_completion_tokens=300,
        temperature=0.3,
    )
    response = chat_completion(request, "ollama", ollama_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_ollama_vision(ollama_client_fixture: Callable[..., Any]) -> None:
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
        model="llama3.2-vision:11b-instruct-q4_K_M",
        max_completion_tokens=200,
        temperature=0.5,
    )

    response = chat_completion(request, "ollama", ollama_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_ollama_vision_multiple_images(ollama_client_fixture: Callable[..., Any]) -> None:
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
        model="llama3.2-vision:11b-instruct-q4_K_M",
        max_completion_tokens=100,
    )
    response = chat_completion(request, "ollama", ollama_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_ollama_vision_multiple_messages(ollama_client_fixture: Callable[..., Any]) -> None:
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
        model="llama3.2-vision:11b-instruct-q4_K_M",
        max_completion_tokens=100,
    )

    response = chat_completion(request, "ollama", ollama_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_chat_completion_ollama_max_tokens(ollama_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="llama3.2-vision:11b-instruct-q4_K_M",
        messages=[UserMessage(content="What is the capital of France?")],
        max_tokens=100,
    )
    response = chat_completion(request, "ollama", ollama_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


# endregion


# region Anthropic


@pytest.fixture(
    params=[
        {},
    ]
)
def anthropic_client_fixture(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return anthropic_client(**request.param)


def test_anthropic_chat_completion_simple(anthropic_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="claude-3-7-sonnet-20250219",
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is the capital of France?"),
        ],
        max_completion_tokens=200,
    )
    response = chat_completion(request, "anthropic", anthropic_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_anthropic_chat_completion_multiple(anthropic_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="claude-3-7-sonnet-20250219",
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is the capital of France?"),
        ],
        max_completion_tokens=200,
        temperature=1,
        top_k=20,
        top_p=0.95,
    )
    response = chat_completion(request, "anthropic", anthropic_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_anthropic_chat_completion_tool(anthropic_client_fixture: Callable[..., Any]) -> None:
    stock_tool = {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."}
            },
            "required": ["ticker"],
        },
    }

    request = ChatCompletionRequest(
        model="claude-3-7-sonnet-20250219",
        messages=[
            UserMessage(content="What's MSFT stock at today?"),
        ],
        tools=[stock_tool],
        max_tokens=300,
    )
    response = chat_completion(request, "anthropic", anthropic_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_anthropic_chat_completion_multiple_tools(anthropic_client_fixture: Callable[..., Any]) -> None:
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                    },
                },
                "required": ["location"],
            },
        },
        {
            "name": "get_time",
            "description": "Get the current time in a given time zone",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "The IANA time zone name, e.g. America/Los_Angeles"}
                },
                "required": ["timezone"],
            },
        },
    ]

    request = ChatCompletionRequest(
        model="claude-3-7-sonnet-20250219",
        messages=[
            UserMessage(content="What is the weather like right now in New York? Also what time is it there?"),
        ],
        tools=tools,
        temperature=1,
        max_tokens=400,
    )
    response = chat_completion(request, "anthropic", anthropic_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_anthropic_chat_completion_tool_required_name(anthropic_client_fixture: Callable[..., Any]) -> None:
    tools = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "input_schema": {
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
        }
    ]
    request = ChatCompletionRequest(
        model="claude-3-7-sonnet-20250219",
        messages=[
            UserMessage(content="What's the current weather like in Boston, MA today?"),
        ],
        tools=tools,
        tool_choice="get_current_weather",
        max_completion_tokens=300,
        temperature=0,
    )
    response = chat_completion(request, "anthropic", anthropic_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_anthropic_chat_completion_tool_required(anthropic_client_fixture: Callable[..., Any]) -> None:
    tools = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "input_schema": {
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
        }
    ]
    request = ChatCompletionRequest(
        model="claude-3-7-sonnet-20250219",
        messages=[
            UserMessage(content="What's the current weather like in Boston, MA today?"),
        ],
        tools=tools,
        tool_choice="any",
        parallel_tool_calls=False,
        max_completion_tokens=300,
        temperature=0,
    )
    response = chat_completion(request, "anthropic", anthropic_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_anthropic_chat_completion_builtin_tool(anthropic_client_fixture: Callable[..., Any]) -> None:
    tools = [
        {"type": "text_editor_20250124", "name": "str_replace_editor"},
    ]
    request = ChatCompletionRequest(
        model="claude-3-7-sonnet-20250219",
        messages=[
            UserMessage(content="There's a syntax error in my primes.py file. Can you help me fix it?"),
        ],
        tools=tools,
        tool_choice="any",
        max_completion_tokens=500,
        temperature=0.2,
    )
    response = chat_completion(request, "anthropic", anthropic_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_anthropic_chat_completion_tool_message(anthropic_client_fixture: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        UserMessage(content="What is the weather in Boston, MA?"),
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="abc123",
                    function=Function(
                        name="get_current_weather",
                        arguments={"location": "Boston, MA"},
                    ),
                )
            ],
        ),
        ToolMessage(name="abc123", content="The weather in Boston, MA is 70 degrees Fahrenheit."),
    ]
    request = ChatCompletionRequest(
        model="claude-3-7-sonnet-20250219",
        messages=messages,
        max_completion_tokens=300,
        temperature=0.3,
    )
    response = chat_completion(request, "anthropic", anthropic_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


# endregion


# region Gemini


@pytest.fixture(
    params=[
        {},
    ]
)
def gemini_client_fixture(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return gemini_client(**request.param)


def test_gemini_chat_completion_simple(gemini_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gemini-2.5-pro-exp-03-25",
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is the capital of France?"),
        ],
        max_completion_tokens=200,
    )
    response = chat_completion(request, "gemini", gemini_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_gemini_chat_completion_multiple(gemini_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gemini-2.5-flash-preview-04-17",
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is the capital of France?"),
        ],
        max_completion_tokens=200,
        temperature=1,
        top_k=20,
        top_p=0.95,
    )
    response = chat_completion(request, "gemini", gemini_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_gemini_chat_completion_tool(gemini_client_fixture: Callable[..., Any]) -> None:
    stock_tool = {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."}
            },
            "required": ["ticker"],
        },
    }

    request = ChatCompletionRequest(
        model="gemini-2.5-flash-preview-04-17",
        messages=[
            UserMessage(content="What's MSFT stock at today?"),
        ],
        tools=[stock_tool],
        max_tokens=300,
    )
    response = chat_completion(request, "gemini", gemini_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_gemini_chat_completion_multiple_tools(gemini_client_fixture: Callable[..., Any]) -> None:
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                    },
                },
                "required": ["location"],
            },
        },
        {
            "name": "get_time",
            "description": "Get the current time in a given time zone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "The IANA time zone name, e.g. America/Los_Angeles"}
                },
                "required": ["timezone"],
            },
        },
    ]

    request = ChatCompletionRequest(
        model="gemini-2.5-flash-preview-04-17",
        messages=[
            UserMessage(content="What is the weather like right now in New York? Also what time is it there?"),
        ],
        tools=tools,
        temperature=1,
        max_tokens=400,
    )
    response = chat_completion(request, "gemini", gemini_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_gemini_chat_completion_tool_required_name(gemini_client_fixture: Callable[..., Any]) -> None:
    tools = [
        {
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
        }
    ]
    request = ChatCompletionRequest(
        model="gemini-2.5-flash-preview-04-17",
        messages=[
            UserMessage(content="What's the current weather like in Boston, MA today?"),
        ],
        tools=tools,
        tool_choice="get_current_weather",
        max_completion_tokens=300,
        temperature=0,
    )
    response = chat_completion(request, "gemini", gemini_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_gemini_chat_completion_tool_required(gemini_client_fixture: Callable[..., Any]) -> None:
    tools = [
        {
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
        }
    ]
    request = ChatCompletionRequest(
        model="gemini-2.5-flash-preview-04-17",
        messages=[
            UserMessage(content="What's the current weather like in Boston, MA today?"),
        ],
        tools=tools,
        tool_choice="any",
        parallel_tool_calls=False,
        max_completion_tokens=300,
        temperature=0,
    )
    response = chat_completion(request, "gemini", gemini_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_gemini_chat_completion_tool_message(gemini_client_fixture: Callable[..., Any]) -> None:
    messages: list[MessageT] = [
        UserMessage(content="What is the weather in Boston, MA?"),
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="abc123",
                    function=Function(
                        name="get_current_weather",
                        arguments={"location": "Boston, MA"},
                    ),
                )
            ],
        ),
        ToolMessage(name="abc123", content="70 F"),
    ]
    request = ChatCompletionRequest(
        model="gemini-2.5-flash-preview-04-17",
        messages=messages,
        max_completion_tokens=300,
        temperature=0.3,
    )
    response = chat_completion(request, "gemini", gemini_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_gemini_chat_completion_vision(gemini_client_fixture: Callable[..., Any]) -> None:
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
        model="gemini-2.5-flash-preview-04-17",
        max_completion_tokens=200,
        temperature=0.5,
    )

    response = chat_completion(request, "gemini", gemini_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_gemini_chat_completion_vision_tool_call(gemini_client_fixture: Callable[..., Any]) -> None:
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
            "name": "ocr",
            "description": "Perform Optical Character Recognition (OCR) on an image",
            "parameters": {},
        },
    ]

    request = ChatCompletionRequest(
        messages=messages,
        model="gemini-2.5-flash-preview-04-17",
        tools=tools,
        max_completion_tokens=200,
    )

    response = chat_completion(request, "gemini", gemini_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


def test_gemini_chat_completion_vision_multiple_images(gemini_client_fixture: Callable[..., Any]) -> None:
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
        model="gemini-2.5-flash-preview-04-17",
        max_completion_tokens=100,
    )
    response = chat_completion(request, "gemini", gemini_client_fixture)
    print(response.model_dump(mode="json", exclude_none=True))


# endregion
