from collections.abc import AsyncGenerator, Callable
import json
import os
import re
import time
from typing import Any, Literal, cast

from loguru import logger
from ollama import AsyncClient, ChatResponse, Client, ResponseError

from not_again_ai.llm.chat_completion.types import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChoiceStream,
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Function,
    PartialFunction,
    PartialToolCall,
    Role,
    ToolCall,
)

OLLAMA_PARAMETER_MAP = {
    "frequency_penalty": "repeat_penalty",
    "max_completion_tokens": "num_predict",
    "context_window": "num_ctx",
    "n": None,
    "tool_choice": None,
    "reasoning_effort": None,
    "parallel_tool_calls": None,
    "logit_bias": None,
    "top_logprobs": None,
    "presence_penalty": None,
    "max_tokens": "num_predict",
}


def validate(request: ChatCompletionRequest) -> None:
    if request.json_mode and request.structured_outputs is not None:
        raise ValueError("json_schema and json_mode cannot be used together.")

    # Check if any of the parameters set to OLLAMA_PARAMETER_MAP are not None
    for key, value in OLLAMA_PARAMETER_MAP.items():
        if value is None and getattr(request, key) is not None:
            logger.warning(f"Parameter {key} is not supported by Ollama and will be ignored.")

    # If "stop" is not None, check if it is just a string
    if isinstance(request.stop, list):
        logger.warning("Parameter 'stop' needs to be a string and not a list. It will be ignored.")
        request.stop = None

    # Raise an error if both "max_tokens" and "max_completion_tokens" are provided
    if request.max_tokens is not None and request.max_completion_tokens is not None:
        raise ValueError("`max_tokens` and `max_completion_tokens` cannot both be provided.")


def format_kwargs(request: ChatCompletionRequest) -> dict[str, Any]:
    kwargs = request.model_dump(mode="json", exclude_none=True)
    # For each key in OLLAMA_PARAMETER_MAP
    # If it is not None, set the key in kwargs to the value of the corresponding value in OLLAMA_PARAMETER_MAP
    # If it is None, remove that key from kwargs
    for key, value in OLLAMA_PARAMETER_MAP.items():
        if value is not None and key in kwargs:
            kwargs[value] = kwargs.pop(key)
        elif value is None and key in kwargs:
            del kwargs[key]

    # If json_mode is True, set the format to json
    json_mode = kwargs.get("json_mode", None)
    if json_mode:
        kwargs["format"] = "json"
        kwargs.pop("json_mode")
    elif json_mode is not None and not json_mode:
        kwargs.pop("json_mode")

    # If structured_outputs is not None, set the format to structured_outputs
    if kwargs.get("structured_outputs", None):
        # Check if the schema is in the OpenAI and pull out the schema
        if "schema" in kwargs["structured_outputs"]:
            kwargs["format"] = kwargs["structured_outputs"]["schema"]
            kwargs.pop("structured_outputs")
        else:
            kwargs["format"] = kwargs.pop("structured_outputs")

    option_fields = [
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "num_ctx",
        "repeat_last_n",
        "repeat_penalty",
        "temperature",
        "seed",
        "stop",
        "tfs_z",
        "num_predict",
        "top_k",
        "top_p",
        "min_p",
    ]
    # For each field in option_fields, if it is in kwargs, make it under an options dictionary
    options = {}
    for field in option_fields:
        if field in kwargs:
            options[field] = kwargs.pop(field)
    kwargs["options"] = options

    for message in kwargs["messages"]:
        role = message.get("role", None)
        # For each ToolMessage, remove the name field
        if role is not None and role == "tool":
            message.pop("name")

        # For each AssistantMessage with tool calls, remove the id field
        if role is not None and role == "assistant" and message.get("tool_calls", None):
            for tool_call in message["tool_calls"]:
                tool_call.pop("id")

        # Content and images need to be separated
        images = []
        content = ""
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "image_url":
                    image_url = item["image_url"]["url"]
                    # Remove the data URL prefix if present
                    if image_url.startswith("data:"):
                        image_url = image_url.split("base64,", 1)[1]
                    images.append(image_url)
                else:
                    content += item["text"]
        else:
            content = message["content"]

        message["content"] = content
        if len(images) > 1:
            images = images[:1]
            logger.warning("Ollama model only supports a single image per message. Using only the first images.")
        message["images"] = images

    return kwargs


def ollama_chat_completion(
    request: ChatCompletionRequest,
    client: Callable[..., Any],
) -> ChatCompletionResponse:
    validate(request)
    kwargs = format_kwargs(request)

    try:
        start_time = time.time()
        response: ChatResponse = client(**kwargs)
        end_time = time.time()
        response_duration = round(end_time - start_time, 4)
    except ResponseError as e:
        # If the error says "model 'model' not found" use regex then raise a more specific error
        expected_pattern = f"model '{request.model}' not found"
        if re.search(expected_pattern, e.error):
            raise ResponseError(f"Model '{request.model}' not found.") from e
        else:
            raise ResponseError(e.error) from e

    errors = ""

    # Handle tool calls
    tool_calls: list[ToolCall] | None = None
    if response.message.tool_calls:
        parsed_tool_calls: list[ToolCall] = []
        for tool_call in response.message.tool_calls:
            tool_name = tool_call.function.name
            if request.tools and tool_name not in [tool["function"]["name"] for tool in request.tools]:
                errors += f"Tool call {tool_call} has an invalid tool name: {tool_name}\n"
            tool_args = dict(tool_call.function.arguments)
            parsed_tool_calls.append(
                ToolCall(
                    id="",
                    function=Function(
                        name=tool_name,
                        arguments=tool_args,
                    ),
                )
            )
        tool_calls = parsed_tool_calls

    json_message = None
    if (request.json_mode or (request.structured_outputs is not None)) and response.message.content:
        try:
            json_message = json.loads(response.message.content)
        except json.JSONDecodeError:
            errors += "Message failed to parse into JSON\n"

    finish_reason = cast(
        Literal["stop", "length", "tool_calls", "content_filter"],
        "stop" if response.done_reason is None else response.done_reason or "stop",
    )

    choice = ChatCompletionChoice(
        message=AssistantMessage(
            content=response.message.content or "",
            tool_calls=tool_calls,
        ),
        finish_reason=finish_reason,
        json_message=json_message,
    )

    return ChatCompletionResponse(
        choices=[choice],
        errors=errors.strip(),
        completion_tokens=response.get("eval_count", -1),
        prompt_tokens=response.get("prompt_eval_count", -1),
        response_duration=response_duration,
    )


async def ollama_chat_completion_stream(
    request: ChatCompletionRequest,
    client: Callable[..., Any],
) -> AsyncGenerator[ChatCompletionChunk, None]:
    validate(request)
    kwargs = format_kwargs(request)

    start_time = time.time()
    stream = await client(**kwargs)

    async for chunk in stream:
        errors = ""
        # Handle tool calls
        tool_calls: list[PartialToolCall] | None = None
        if chunk.message.tool_calls:
            parsed_tool_calls: list[PartialToolCall] = []
            for tool_call in chunk.message.tool_calls:
                tool_name = tool_call.function.name
                if request.tools and tool_name not in [tool["function"]["name"] for tool in request.tools]:
                    errors += f"Tool call {tool_call} has an invalid tool name: {tool_name}\n"
                tool_args = tool_call.function.arguments

                parsed_tool_calls.append(
                    PartialToolCall(
                        id="",
                        function=PartialFunction(
                            name=tool_name,
                            arguments=tool_args,
                        ),
                    )
                )
            tool_calls = parsed_tool_calls

        current_time = time.time()
        response_duration = round(current_time - start_time, 4)

        delta = ChatCompletionDelta(
            content=chunk.message.content or "",
            role=Role.ASSISTANT,
            tool_calls=tool_calls,
        )
        choice_obj = ChatCompletionChoiceStream(
            delta=delta,
            finish_reason=chunk.done_reason,
            index=0,
        )
        chunk_obj = ChatCompletionChunk(
            choices=[choice_obj],
            errors=errors.strip(),
            completion_tokens=chunk.get("eval_count", None),
            prompt_tokens=chunk.get("prompt_eval_count", None),
            response_duration=response_duration,
        )
        yield chunk_obj


def ollama_client(
    host: str | None = None, timeout: float | None = None, async_client: bool = False
) -> Callable[..., Any]:
    """Create an Ollama client instance based on the specified host or will read from the OLLAMA_HOST environment variable.

    Args:
        host (str, optional): The host URL of the Ollama server.
        timeout (float, optional): The timeout for requests

    Returns:
        Client: An instance of the Ollama client.

    Examples:
        >>> client = client(host="http://localhost:11434")
    """
    if host is None:
        host = os.getenv("OLLAMA_HOST")
        if host is None:
            logger.warning("OLLAMA_HOST environment variable not set, using default host: http://localhost:11434")
            host = "http://localhost:11434"

    def client_callable(**kwargs: Any) -> Any:
        client = AsyncClient(host=host, timeout=timeout) if async_client else Client(host=host, timeout=timeout)
        return client.chat(**kwargs)

    return client_callable
