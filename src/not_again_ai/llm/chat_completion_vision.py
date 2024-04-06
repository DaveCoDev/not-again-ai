from typing import Any

from openai import OpenAI


def chat_completion_vision(
    messages: list[dict[str, Any]],
    model: str,
    client: OpenAI,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    seed: int | None = None,
    n: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get an OpenAI chat completion response for vision models only: https://platform.openai.com/docs/guides/vision

    Args:
        messages (list): A list of messages comprising the conversation so far.
            See https://platform.openai.com/docs/api-reference/chat/create for details on the format
        model (str): ID of the model to use for generating chat completions. Refer to OpenAI's documentation
            for details on available models.
        client (OpenAI): An instance of the OpenAI client, used to make requests to the API.
        max_tokens (int | None, optional): The maximum number of tokens to generate in the chat completion.
            If None, defaults to the model's maximum context length. Defaults to None.
        temperature (float, optional): Controls the randomness of the output. A higher temperature produces
            more varied results, whereas a lower temperature results in more deterministic and predictable text.
            Must be between 0 and 2. Defaults to 0.7.
        seed (int | None, optional): A seed used for deterministic generation. Providing a seed ensures that
            the same input will produce the same output across different runs. Defaults to None.
        n (int, optional): The number of chat completion choices to generate for each input message.
            Defaults to 1.
        **kwargs (Any): Additional keyword arguments to pass to the OpenAI client chat completion method.

    Returns:
        dict[str, Any]: A dictionary containing the generated responses and metadata. Key components include:
            'finish_reason' (str): The reason the model stopped generating further tokens.
                Can be 'stop' or 'length'
            'tool_names' (list[str], optional): The names of the tools called by the model.
            'tool_args_list' (list[dict], optional): The arguments of the tools called by the model.
            'message' (str | dict): The content of the generated assistant message.
            'choices' (list[dict], optional): A list of chat completion choices if n > 1 where each dict contains the above fields.
            'completion_tokens' (int): The number of tokens used by the model to generate the completion.
                NOTE: If n > 1 this is the sum of all completions and thus will be same value in each dict.
            'prompt_tokens' (int): The number of tokens in the generated response.
                NOTE: If n > 1 this is the sum of all completions and thus will be same value in each dict.
            'system_fingerprint' (str, optional): If seed is set, a unique identifier for the model used to generate the response.
    """
    kwargs.update(
        {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
        }
    )

    if seed is not None:
        kwargs["seed"] = seed

    response = client.chat.completions.create(**kwargs)

    response_data: dict[str, Any] = {"choices": []}
    for response_choice in response.choices:
        response_data_curr = {}
        finish_reason = response_choice.finish_reason
        response_data_curr["finish_reason"] = finish_reason

        if finish_reason == "stop" or finish_reason == "length":
            message = response_choice.message.content
            response_data_curr["message"] = message

        response_data["choices"].append(response_data_curr)

    usage = response.usage
    if usage is not None:
        response_data["completion_tokens"] = usage.completion_tokens
        response_data["prompt_tokens"] = usage.prompt_tokens

    if seed is not None and response.system_fingerprint is not None:
        response_data["system_fingerprint"] = response.system_fingerprint

    if len(response_data["choices"]) == 1:
        response_data.update(response_data["choices"][0])
        del response_data["choices"]

    return response_data
