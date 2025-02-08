from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from not_again_ai.llm.chat_completion import chat_completion_stream
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


# region Azure OpenAI


@pytest.fixture(
    params=[
        {"async_client": True},
        {"api_type": "azure_openai", "aoai_api_version": "2025-01-01-preview", "async_client": True},
    ]
)
def openai_aoai_client_fixture(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return openai_client(**request.param)


async def test_chat_completion_stream_simple(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            SystemMessage(content="Hello, world!"),
            UserMessage(content="What is the capital of France?"),
        ],
        max_completion_tokens=100,
    )
    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_json_mode(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
        max_completion_tokens=200,
        temperature=0,
        json_mode=True,
    )
    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_n(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello!"),
        ],
        max_completion_tokens=100,
        n=2,
    )
    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_toplogprobs(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_tool_simple(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_tool_required_name(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_multiple_tools(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_message_with_tools(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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
    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_structured_output(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_vision(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_vision_tool_call(openai_aoai_client_fixture: Callable[..., Any]) -> None:
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

    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_invalid_params(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini-2024-07-18",
        messages=[UserMessage(content="What is the capital of France?")],
        max_completion_tokens=100,
        context_window=1000,
        mirostat=1,
    )
    async for chunk in chat_completion_stream(request, "openai", openai_aoai_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


# endregion


# region Ollama
@pytest.fixture(
    params=[
        {"async_client": True},
    ]
)
def ollama_client_fixture(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return ollama_client(**request.param)


async def test_chat_completion_stream_ollama(ollama_client_fixture: Callable[..., Any]) -> None:
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
    async for chunk in chat_completion_stream(request, "ollama", ollama_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_ollama_structured_output(ollama_client_fixture: Callable[..., Any]) -> None:
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

    async for chunk in chat_completion_stream(request, "ollama", ollama_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_ollama_multiple_tools(ollama_client_fixture: Callable[..., Any]) -> None:
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
    async for chunk in chat_completion_stream(request, "ollama", ollama_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_ollama_tool_message(ollama_client_fixture: Callable[..., Any]) -> None:
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
    async for chunk in chat_completion_stream(request, "ollama", ollama_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


async def test_chat_completion_stream_ollama_vision_multiple_images(ollama_client_fixture: Callable[..., Any]) -> None:
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
    async for chunk in chat_completion_stream(request, "ollama", ollama_client_fixture):
        print(chunk.model_dump(mode="json", exclude_none=True))


# endregion
