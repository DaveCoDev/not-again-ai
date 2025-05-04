import base64
from collections.abc import Callable
import os
import time
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import FunctionCall, FunctionCallingConfigMode, GenerateContentResponse

from not_again_ai.llm.chat_completion.types import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Function,
    ImageContent,
    Role,
    TextContent,
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
    """Experimental Gemini chat completion function."""
    # Handle messages
    # Any system messages need to be removed from messages and concatenated into a single string (in order)
    system = ""
    contents = []
    for message in request.messages:
        if message.role == "system":
            # Handle both string content and structured content
            if isinstance(message.content, str):
                system += message.content + "\n"
            else:
                # If it's a list of content parts, extract text content
                for part in message.content:
                    if hasattr(part, "text"):
                        system += part.text + "\n"
        elif message.role == "tool":
            tool_name = message.name if message.name is not None else ""
            function_response_part = types.Part.from_function_response(
                name=tool_name,
                response={"result": message.content},
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[function_response_part],
                )
            )
        elif message.role == "assistant":
            if message.content and isinstance(message.content, str):
                contents.append(types.Content(role="model", parts=[types.Part(text=message.content)]))
            function_parts = []
            if isinstance(message, AssistantMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
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
            if isinstance(message.content, str):
                contents.append(types.Content(role="user", parts=[types.Part(text=message.content)]))
            elif isinstance(message.content, list):
                parts = []
                for part in message.content:
                    if isinstance(part, TextContent):
                        parts.append(types.Part(text=part.text))
                    elif isinstance(part, ImageContent):
                        # Extract MIME type and data from data URI
                        uri_parts = part.image_url.url.split(",", 1)
                        if len(uri_parts) == 2:
                            mime_type = uri_parts[0].split(":")[1].split(";")[0]
                            base64_data = uri_parts[1]
                            image_data = base64.b64decode(base64_data)
                            parts.append(types.Part.from_bytes(mime_type=mime_type, data=image_data))
                contents.append(types.Content(role="user", parts=parts))

    kwargs: dict[str, Any] = {}
    kwargs["contents"] = contents
    kwargs["model"] = request.model
    config: dict[str, Any] = {}
    config["system_instruction"] = system.rstrip()
    config["automatic_function_calling"] = {"disable": True}

    # Handle the possible tool choice options
    if request.tool_choice:
        tool_choice = request.tool_choice
        if tool_choice == "auto":
            config["tool_config"] = types.FunctionCallingConfig(mode=FunctionCallingConfigMode.AUTO)
        elif tool_choice == "any":
            config["tool_config"] = types.FunctionCallingConfig(mode=FunctionCallingConfigMode.ANY)
        elif tool_choice == "none":
            config["tool_config"] = types.FunctionCallingConfig(mode=FunctionCallingConfigMode.NONE)
        elif isinstance(tool_choice, list):
            config["tool_config"] = types.FunctionCallingConfig(
                mode=FunctionCallingConfigMode.ANY, allowed_function_names=tool_choice
            )
        elif tool_choice not in (None, "auto", "any", "none"):
            config["tool_config"] = types.FunctionCallingConfig(
                mode=FunctionCallingConfigMode.ANY, allowed_function_names=[tool_choice]
            )

    # Handle tools
    tools = []
    for tool in request.tools or []:
        tools.append(types.Tool(function_declarations=[tool]))  # type: ignore
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

    finish_reason = "other"
    if response.candidates and response.candidates[0].finish_reason:
        finish_reason_str = str(response.candidates[0].finish_reason)
        finish_reason = GEMINI_FINISH_REASON_MAP.get(finish_reason_str, "other")

    tool_calls: list[ToolCall] = []
    tool_call_objs = response.function_calls
    if tool_call_objs:
        for tool_call_obj in tool_call_objs:
            tool_call_id = tool_call_obj.id if tool_call_obj.id else ""
            tool_calls.append(
                ToolCall(
                    id=tool_call_id,
                    function=Function(
                        name=tool_call_obj.name if tool_call_obj.name is not None else "",
                        arguments=tool_call_obj.args if tool_call_obj.args is not None else {},
                    ),
                )
            )

    assistant_message = ""
    if (
        response.candidates
        and response.candidates[0].content
        and response.candidates[0].content.parts
        and response.candidates[0].content.parts[0].text
    ):
        assistant_message = response.candidates[0].content.parts[0].text

    choice = ChatCompletionChoice(
        message=AssistantMessage(
            role=Role.ASSISTANT,
            content=assistant_message,
            tool_calls=tool_calls,
        ),
        finish_reason=finish_reason,
    )

    completion_tokens = 0
    # Add null check for usage_metadata
    if response.usage_metadata is not None:
        if response.usage_metadata.thoughts_token_count:
            completion_tokens = response.usage_metadata.thoughts_token_count
        if response.usage_metadata.candidates_token_count:
            completion_tokens += response.usage_metadata.candidates_token_count

    # Set safe default for prompt_tokens
    prompt_tokens = 0
    if response.usage_metadata is not None and response.usage_metadata.prompt_token_count:
        prompt_tokens = response.usage_metadata.prompt_token_count

    chat_completion_response = ChatCompletionResponse(
        choices=[choice],
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
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
