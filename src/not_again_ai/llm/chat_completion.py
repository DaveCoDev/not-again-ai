import contextlib
import json
from typing import Any

from openai import OpenAI


def chat_completion(
    messages: list[dict[str, str]],
    model: str,
    client: OpenAI,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    max_tokens: int | None = None,
    temperature: float = 0.7,
    json_mode: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get an OpenAI chat completion response: https://platform.openai.com/docs/api-reference/chat/create

    Args:
        messages (list): A list of messages comprising the conversation so far.
        model (str): ID of the model to use. See the model endpoint compatibility table:
            https://platform.openai.com/docs/models/model-endpoint-compatibility
            for details on which models work with the Chat API.
        client (OpenAI): An instance of the OpenAI client.
        tools (list[dict[str, Any]], optional): A list of tools the model may generate JSON inputs for.
            Defaults to None.
        tool_choice (str, optional): The tool choice to use. Can be "auto", "none", or a specific function name.
            Defaults to "auto".
        max_tokens (int, optional): The maximum number of tokens to generate in the chat completion.
            Defaults to None, which automatically limits to the model's maximum context length.
        temperature (float, optional): What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic. Defaults to 0.7.
        json_mode (bool, optional): When JSON mode is enabled, the model is constrained to only
            generate strings that parse into valid JSON object and will return a dictionary.
            See https://platform.openai.com/docs/guides/text-generation/json-mode
        **kwargs: Additional keyword arguments to pass to the OpenAI client chat completion.

    Returns:
        dict: A dictionary containing the following keys:
            - "finish_reason" (str): The reason the model stopped generating further tokens.
                Can be "stop", "length", or "tool_calls".
            - "tool_names" (list[str], optional): The names of the tools called by the model.
            - "tool_args_list" (list[dict], optional): The arguments of the tools called by the model.
            - "message" (str | dict): The content of the generated assistant message.
                If json_mode is True, this will be a dictionary.
            - "completion_tokens" (int): The number of tokens used by the model to generate the completion.
            - "prompt_tokens" (int): The number of tokens in the generated response.
    """
    response_format = {"type": "json_object"} if json_mode else None

    kwargs.update(
        {
            "messages": messages,
            "model": model,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format,
            "n": 1,
        }
    )

    if tools is not None:
        if tool_choice not in ["none", "auto"]:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        else:
            kwargs["tool_choice"] = tool_choice

    # Call the function with the set parameters
    response = client.chat.completions.create(**kwargs)

    response_data = {}
    finish_reason = response.choices[0].finish_reason
    response_data["finish_reason"] = finish_reason

    # Not checking finish_reason=="tool_calls" here because when a user providea function name as tool_choice,
    # the finish reason is "stop", not "tool_calls"
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        tool_names = []
        tool_args_list = []
        for tool_call in tool_calls:
            tool_names.append(tool_call.function.name)
            tool_args_list.append(json.loads(tool_call.function.arguments))
        response_data["tool_names"] = tool_names
        response_data["tool_args_list"] = tool_args_list
    elif finish_reason == "stop" or finish_reason == "length":
        message = response.choices[0].message.content
        if json_mode:
            with contextlib.suppress(json.JSONDecodeError):
                message = json.loads(message)
        response_data["message"] = message

    usage = response.usage
    if usage is not None:
        response_data["completion_tokens"] = usage.completion_tokens
        response_data["prompt_tokens"] = usage.prompt_tokens

    return response_data
