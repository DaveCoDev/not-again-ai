import json
from typing import Any

from openai import OpenAI


def chat_completion(
    messages: list[dict[str, str]],
    model: str,
    client: OpenAI,
    functions: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get an OpenAI chat completion response: https://platform.openai.com/docs/api-reference/chat/create

    Args:
        messages (list): A list of messages comprising the conversation so far.
        model (str): ID of the model to use. See the model endpoint compatibility table:
            https://platform.openai.com/docs/models/model-endpoint-compatibility
            for details on which models work with the Chat API.
        client (OpenAI): An instance of the OpenAI client.
        functions (list, optional): A list of functions the model may generate JSON inputs for. Defaults to None.
        max_tokens (int, optional): The maximum number of tokens to generate in the chat completion.
            Defaults to limited to the model's context length.
        temperature (float, optional): What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic. Defaults to 0.7.
        **kwargs: Additional keyword arguments to pass to the OpenAI client chat completion.

    Returns:
        dict: A dictionary containing the following keys:
            - "finish_reason" (str): The reason the model stopped generating further tokens. Can be "stop" or "function_call".
            - "function_name" (str, optional): The name of the function called by the model, present only if "finish_reason" is "function_call".
            - "function_args" (dict, optional): The arguments of the function called by the model, present only if "finish_reason" is "function_call".
            - "message" (str, optional): The content of the generated assistant message, present only if "finish_reason" is "stop".
            - "completion_tokens" (int): The number of tokens used by the model to generate the completion.
            - "prompt_tokens" (int): The number of tokens in the generated response.
    """
    if functions is None:
        response = client.chat.completions.create(
            messages=messages,  # type: ignore
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            **kwargs,
        )
    else:
        response = client.chat.completions.create(  # type: ignore
            messages=messages,
            model=model,
            functions=functions,
            function_call="auto",
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            **kwargs,
        )

    response_data = {}
    finish_reason = response.choices[0].finish_reason
    response_data["finish_reason"] = finish_reason
    if finish_reason == "function_call":
        function_call = response.choices[0].message.function_call
        if function_call is not None:
            response_data["function_name"] = function_call.name  # type: ignore
            response_data["function_args"] = json.loads(function_call.arguments)
    elif finish_reason == "stop" or finish_reason == "length":
        message = response.choices[0].message
        response_data["message"] = message.content  # type: ignore
    usage = response.usage
    if usage is not None:
        response_data["completion_tokens"] = usage.completion_tokens  # type: ignore
        response_data["prompt_tokens"] = usage.prompt_tokens  # type: ignore
    return response_data
