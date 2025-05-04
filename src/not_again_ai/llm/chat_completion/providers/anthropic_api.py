from collections.abc import Callable
import os
import time
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message

from not_again_ai.llm.chat_completion.types import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Function,
    ToolCall,
)

ANTHROPIC_PARAMETER_MAP = {
    "max_completion_tokens": "max_tokens",
}


def anthropic_chat_completion(request: ChatCompletionRequest, client: Callable[..., Any]) -> ChatCompletionResponse:
    """Anthropic chat completion function.

    TODO
    - Image messages
    - Thinking
    - Citations
    - Stop sequences
    - Documents
    """
    kwargs = request.model_dump(mode="json", exclude_none=True)

    # For each key in ANTHROPIC_PARAMETER_MAP
    # If it is not None, set the key in kwargs to the value of the corresponding value in ANTHROPIC_PARAMETER_MAP
    # If it is None, remove that key from kwargs
    for key, value in ANTHROPIC_PARAMETER_MAP.items():
        if value is not None and key in kwargs:
            kwargs[value] = kwargs.pop(key)
        elif value is None and key in kwargs:
            del kwargs[key]

    # Handle messages
    # Any system messages need to be removed from messages and concatenated into a single string (in order)
    # Any tool messages need to be converted to a special user message
    # Any assistant messages with tool calls need to be converted.
    system = ""
    new_messages = []
    for message in kwargs["messages"]:
        if message["role"] == "system":
            system += message["content"] + "\n"
        elif message["role"] == "tool":
            new_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["name"],
                            "content": message["content"],
                        }
                    ],
                }
            )
        elif message["role"] == "assistant":
            content = []
            if message.get("content", None):
                content.append(
                    {
                        "type": "text",
                        "content": message["content"],
                    }
                )
            for tool_call in message.get("tool_calls", []):
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": tool_call["function"]["arguments"],
                    }
                )
            new_messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
        else:
            new_messages.append(message)
    kwargs["messages"] = new_messages
    system = system.strip()
    if system:
        kwargs["system"] = system

    # Handle tool choice and parallel tool calls
    if kwargs.get("tool_choice") is not None:
        tool_choice_value = kwargs.pop("tool_choice")
        tool_choice = {}
        if tool_choice_value == "none":
            tool_choice["type"] = "none"
        elif tool_choice_value in ["auto", "any"]:
            tool_choice["type"] = "auto"
            if kwargs.get("parallel_tool_calls") is not None:
                tool_choice["disable_parallel_tool_use"] = not kwargs["parallel_tool_calls"]  # type: ignore
        else:
            tool_choice["name"] = tool_choice_value
            tool_choice["type"] = "tool"
            if kwargs.get("parallel_tool_calls") is not None:
                tool_choice["disable_parallel_tool_use"] = not kwargs["parallel_tool_calls"]  # type: ignore
        kwargs["tool_choice"] = tool_choice
    kwargs.pop("parallel_tool_calls", None)

    start_time = time.time()
    response: Message = client(**kwargs)
    end_time = time.time()
    response_duration = round(end_time - start_time, 4)

    tool_calls: list[ToolCall] = []
    assistant_message = ""
    for block in response.content:
        if block.type == "text":
            assistant_message += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    function=Function(
                        name=block.name,
                        arguments=block.input,  # type: ignore
                    ),
                )
            )

    choice = ChatCompletionChoice(
        message=AssistantMessage(
            content=assistant_message,
            tool_calls=tool_calls,
        ),
        finish_reason=response.stop_reason or "stop",
    )

    chat_completion_response = ChatCompletionResponse(
        choices=[choice],
        errors="",
        completion_tokens=response.usage.output_tokens,
        prompt_tokens=response.usage.input_tokens,
        cache_read_input_tokens=response.usage.cache_read_input_tokens,
        cache_creation_input_tokens=response.usage.cache_creation_input_tokens,
        response_duration=response_duration,
    )
    return chat_completion_response


def create_client_callable(client_class: type[Anthropic], **client_args: Any) -> Callable[..., Any]:
    """Creates a callable that instantiates and uses an Anthropic client.

    Args:
        client_class: The Anthropic client class to instantiate
        **client_args: Arguments to pass to the client constructor

    Returns:
        A callable that creates a client and returns completion results
    """
    filtered_args = {k: v for k, v in client_args.items() if v is not None}

    def client_callable(**kwargs: Any) -> Any:
        client = client_class(**filtered_args)
        completion = client.beta.messages.create(**kwargs)
        return completion

    return client_callable


def anthropic_client(api_key: str | None = None) -> Callable[..., Any]:
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    client_callable = create_client_callable(Anthropic, api_key=api_key)
    return client_callable
