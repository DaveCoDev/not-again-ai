import contextlib
import json
import re
from typing import Any

from ollama import Client, ResponseError


def _convert_duration(nanoseconds: int) -> float:
    seconds = nanoseconds / 1_000_000_000
    return round(seconds, 5)


def chat_completion(
    messages: list[dict[str, Any]],
    model: str,
    client: Client,
    max_tokens: int | None = None,
    context_window: int | None = None,
    temperature: float = 0.8,
    json_mode: bool = False,
    seed: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Gets a Ollama chat completion response, see https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    For a full list of valid parameters: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

    Args:
        messages (list[dict[str, Any]]): A list of messages to send to the model.
        model (str): The model to use.
        client (Client): The Ollama client.
        max_tokens (int, optional): The maximum number of tokens to generate. Ollama calls this `num_predict`.
        context_window (int, optional): The number of tokens to consider as context. Ollama calls this `num_ctx`.
        temperature (float, optional): The temperature of the model. Increasing the temperature will make the model answer more creatively.
        json_mode (bool, optional): This will structure the response as a valid JSON object.
            It is important to instruct the model to use JSON in the prompt. Otherwise, the model may generate large amounts whitespace.
        seed (int, optional): The seed to use for the model for reproducible outputs. Defaults to None.

    Returns:
        dict[str, Any]: A dictionary with the following keys
            message (str | dict): The content of the generated assistant message.
                If json_mode is True, this will be a dictionary.
            completion_tokens (int): The number of tokens used by the model to generate the completion.
            response_duration (float): The time taken to generate the response in seconds.
    """

    options = {
        "num_predict": max_tokens,
        "num_ctx": context_window,
        "temperature": temperature,
    }
    if seed is not None:
        options["seed"] = seed
    options.update(kwargs)

    all_args = {
        "model": model,
        "messages": messages,
        "options": options,
    }
    if json_mode:
        all_args["format"] = "json"

    try:
        response = client.chat(**all_args)
    except ResponseError as e:
        # If the error says "model 'model' not found" use regex then raise a more specific error
        expected_pattern = f"model '{model}' not found"
        if re.search(expected_pattern, e.error):
            raise ResponseError(
                f"Model '{model}' not found. Please use not_again_ai.llm.ollama.service.pull() first."
            ) from e
        else:
            raise ResponseError(e.message) from e

    response_data: dict[str, Any] = {}

    # Handle getting the message returned by the model
    message = response["message"].get("content", None)
    if message and json_mode:
        with contextlib.suppress(json.JSONDecodeError):
            message = json.loads(message)
    if message:
        response_data["message"] = message

    # Get the number of tokens generated
    response_data["completion_tokens"] = response.get("eval_count", None)

    # Get the latency of the response
    if response.get("total_duration", None):
        response_data["response_duration"] = _convert_duration(response["total_duration"])
    else:
        response_data["response_duration"] = None

    return response_data
