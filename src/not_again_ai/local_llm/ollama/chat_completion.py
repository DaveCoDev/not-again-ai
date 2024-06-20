import contextlib
import json
import re
import time
from typing import Any

from ollama import Client, ResponseError

from not_again_ai.local_llm.ollama.tokens import load_tokenizer, num_tokens_from_messages, num_tokens_in_string


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
            prompt_tokens (int): The number of tokens in the messages sent to the model.
            completion_tokens (int): The number of tokens used by the model to generate the completion.
            response_duration (float): The time, in seconds, taken to generate the response by using the model.
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
        start_time = time.time()
        response = client.chat(**all_args)  # type: ignore
        end_time = time.time()
        response_duration = end_time - start_time
    except ResponseError as e:
        # If the error says "model 'model' not found" use regex then raise a more specific error
        expected_pattern = f"model '{model}' not found"
        if re.search(expected_pattern, e.error):
            raise ResponseError(
                f"Model '{model}' not found. Please use not_again_ai.llm.ollama.service.pull() first."
            ) from e
        else:
            raise ResponseError(e.error) from e

    response_data: dict[str, Any] = {}

    # Handle getting the message returned by the model
    message = response["message"].get("content", None)  # type: ignore
    if message and json_mode:
        with contextlib.suppress(json.JSONDecodeError):
            message = json.loads(message)
    if message:
        response_data["message"] = message

    tokenizer = load_tokenizer(model)
    prompt_tokens = num_tokens_from_messages(messages, tokenizer)
    response_data["prompt_tokens"] = prompt_tokens

    # Get the number of tokens generated
    response_data["completion_tokens"] = response.get("eval_count", None)  # type: ignore
    if response_data["completion_tokens"] is None:
        response_data["completion_tokens"] = num_tokens_in_string(str(response_data["message"]), tokenizer)

    # Get the latency of the response
    response_data["response_duration"] = response_duration

    return response_data
