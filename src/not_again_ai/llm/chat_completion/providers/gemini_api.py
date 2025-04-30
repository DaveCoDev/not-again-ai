from collections.abc import Callable
import os
import time
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import FunctionCall, GenerateContentResponse

from not_again_ai.llm.chat_completion.types import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Function,
    ToolCall,
)

# This should be all of the options we want to support in types.GenerateContentConfig, that are not handled otherwise
GEMINI_PARAMETER_MAP = {
    "max_completion_tokens": "max_output_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
}

GEMINI_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "safety",
    "RECITATION": "recitation",
    "LANGUAGE": "language",
    "OTHER": "other",
    "BLOCKLIST": "blocklist",
    "PROHIBITED_CONTENT": "prohibited_content",
    "SPII": "spii",
    "MALFORMED_FUNCTION_CALL": "malformed_function_call",
    "IMAGE_SAFETY": "image_safety",
}


def gemini_chat_completion(request: ChatCompletionRequest, client: Callable[..., Any]) -> ChatCompletionResponse:
    """Gemini chat completion function."""
    # Handle messages
    # Any system messages need to be removed from messages and concatenated into a single string (in order)
    system = ""
    contents = []
    for message in request.messages:
        if message.role == "system":
            system += message.content + "\n"
        elif message.role == "tool":
            function_response_part = types.Part.from_function_response(
                name=message.name,
                response={"result": message.content},
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[function_response_part],
                )
            )
        elif message.role == "assistant":
            if message.content:
                contents.append(types.Content(role="model", parts=[types.Part(text=message.content)]))
            function_parts = []
            for tool_call in message.tool_calls or []:
                function_call_part = types.Part(
                    function_call=FunctionCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        args=tool_call.function.arguments,
                    )
                )
                function_parts.append(function_call_part)
            if function_parts:
                contents.append(types.Content(role="model", parts=function_parts))
        elif message.role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=message.content)]))

    kwargs = {}
    kwargs["contents"] = contents
    kwargs["model"] = request.model
    config = {}
    config["system_instruction"] = system.rstrip()
    config["automatic_function_calling"] = {"disable": True}

    # Handle the possible tool choice options
    if request.tool_choice:
        tool_choice = request.tool_choice
        if tool_choice == "auto":
            config["tool_config"] = types.FunctionCallingConfig(mode="AUTO")
        elif tool_choice == "any":
            config["tool_config"] = types.FunctionCallingConfig(mode="ANY")
        elif tool_choice == "none":
            config["tool_config"] = types.FunctionCallingConfig(mode="NONE")
        elif isinstance(tool_choice, list):
            config["tool_config"] = types.FunctionCallingConfig(mode="ANY", allowed_function_names=tool_choice)
        elif tool_choice not in (None, "auto", "any", "none"):
            config["tool_config"] = types.FunctionCallingConfig(mode="ANY", allowed_function_names=[tool_choice])

    # Handle tools
    tools = []
    for tool in request.tools or []:
        tools.append(types.Tool(function_declarations=[tool]))
    if tools:
        config["tools"] = tools

    # Everything else defined in GEMINI_PARAMETER_MAP goes into kwargs["config"]
    request_kwargs = request.model_dump(mode="json", exclude_none=True)
    for key, value in GEMINI_PARAMETER_MAP.items():
        if value is not None and key in request_kwargs:
            config[value] = request_kwargs.pop(key)

    kwargs["config"] = types.GenerateContentConfig(**config)

    start_time = time.time()
    response: GenerateContentResponse = client(**kwargs)
    end_time = time.time()
    response_duration = round(end_time - start_time, 4)

    finish_reason = response.candidates[0].finish_reason
    finish_reason = GEMINI_FINISH_REASON_MAP.get(finish_reason, "other")

    tool_calls: list[ToolCall] = []
    tool_call_objs = response.function_calls
    if tool_call_objs:
        for tool_call in tool_call_objs:
            tool_call_id = tool_call.id
            if not tool_call_id:
                tool_call_id = ""
            tool_calls.append(
                ToolCall(
                    id=tool_call_id,
                    function=Function(
                        name=tool_call.name,
                        arguments=tool_call.args,
                    ),
                )
            )

    assistant_message = response.candidates[0].content.parts[0].text
    if not assistant_message:
        assistant_message = ""

    choice = ChatCompletionChoice(
        message=AssistantMessage(
            role="assistant",
            content=assistant_message,
            tool_calls=tool_calls,
        ),
        finish_reason=finish_reason,
    )

    completion_tokens = 0
    if response.usage_metadata.thoughts_token_count:
        completion_tokens = response.usage_metadata.thoughts_token_count
    completion_tokens += response.usage_metadata.candidates_token_count

    chat_completion_response = ChatCompletionResponse(
        choices=[choice],
        completion_tokens=completion_tokens,
        prompt_tokens=response.usage_metadata.prompt_token_count,
        response_duration=response_duration,
    )
    return chat_completion_response


def create_client_callable(client_class: type[genai.Client], **client_args: Any) -> Callable[..., Any]:
    """Creates a callable that instantiates and uses a Google genai client.

    Args:
        client_class: The Google genai client class to instantiate
        **client_args: Arguments to pass to the client constructor

    Returns:
        A callable that creates a client and returns completion results
    """
    filtered_args = {k: v for k, v in client_args.items() if v is not None}

    def client_callable(**kwargs: Any) -> Any:
        client = client_class(**filtered_args)
        completion = client.models.generate_content(**kwargs)
        return completion

    return client_callable


def gemini_client(api_key: str | None = None) -> Callable[..., Any]:
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    client_callable = create_client_callable(genai.Client, api_key=api_key)
    return client_callable
