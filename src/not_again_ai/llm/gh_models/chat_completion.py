import contextlib
import json
import time
from typing import Any

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import ChatCompletionsToolDefinition, ChatRequestMessage


def chat_completion(
    messages: list[ChatRequestMessage],
    model: str,
    client: ChatCompletionsClient,
    tools: list[ChatCompletionsToolDefinition] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Gets a response from GitHub Models using the Azure AI Inference SDK.
    See the available models at https://github.com/marketplace/models
    Full documentation of the SDK is at: https://learn.microsoft.com/en-us/azure/ai-studio/reference/reference-model-inference-chat-completions
    And samples at: https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/ai/azure-ai-inference/samples

    Returns:
        dict[str, Any]: A dictionary with the following keys
            message (str | dict): The content of the generated assistant message.
                If json_mode is True, this will be a dictionary.
            tool_names (list[str], optional): The names of the tools called by the model.
                If the model does not support tools, a ResponseError is raised.
            tool_args_list (list[dict], optional): The arguments of the tools called by the model.
            prompt_tokens (int): The number of tokens in the messages sent to the model.
            completion_tokens (int): The number of tokens used by the model to generate the completion.
            response_duration (float): The time, in seconds, taken to generate the response by using the model.
            system_fingerprint (str, optional): If seed is set, a unique identifier for the model used to generate the response.
    """
    response_format = {"type": "json_object"} if json_mode else None
    start_time = time.time()
    response = client.complete(  # type: ignore
        messages=messages,
        model=model,
        response_format=response_format,  # type: ignore
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        seed=seed,
    )
    end_time = time.time()
    response_duration = end_time - start_time

    response_data = {}
    finish_reason = response.choices[0].finish_reason
    response_data["finish_reason"] = finish_reason.value  # type: ignore

    message = response.choices[0].message.content
    if message and json_mode:
        with contextlib.suppress(json.JSONDecodeError):
            message = json.loads(message)
    response_data["message"] = message

    # Check for tool calls because even if the finish_reason is stop, the model may have called a tool
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        tool_names = []
        tool_args_list = []
        for tool_call in tool_calls:
            tool_names.append(tool_call.function.name)
            tool_args_list.append(json.loads(tool_call.function.arguments))
        response_data["tool_names"] = tool_names
        response_data["tool_args_list"] = tool_args_list

    if seed is not None and hasattr(response, "system_fingerprint"):
        response_data["system_fingerprint"] = response.system_fingerprint

    usage = response.usage
    if usage is not None:
        response_data["completion_tokens"] = usage.completion_tokens
        response_data["prompt_tokens"] = usage.prompt_tokens
    response_data["response_duration"] = round(response_duration, 4)

    return response_data
