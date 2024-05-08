from typing import Any

from ollama import Client
from openai import OpenAI

from not_again_ai.llm.ollama import chat_completion as chat_completion_ollama
from not_again_ai.llm.openai_api import chat_completion as chat_completion_openai


def chat_completion(
    messages: list[dict[str, Any]],
    model: str,
    client: OpenAI | Client,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    json_mode: bool = False,
    seed: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Creates a common wrapper around chat completion models from different providers.
    Currently supports the OpenAI API and Ollama local models.
    All input parameters are supported by all providers in similar ways and the output is standardized.

    Args:
        messages (list[dict[str, Any]]): A list of messages to send to the model.
        model (str): The model name to use.
        client (OpenAI | Client): The client object to use for chat completion.
        max_tokens (int, optional): The maximum number of tokens to generate.
        temperature (float, optional): The temperature of the model. Increasing the temperature will make the model answer more creatively.
        json_mode (bool, optional): This will structure the response as a valid JSON object.
        seed (int, optional): The seed to use for the model for reproducible outputs.

    Returns:
        dict[str, Any]: A dictionary with the following keys
            message (str | dict): The content of the generated assistant message.
                If json_mode is True, this will be a dictionary.
            completion_tokens (int): The number of tokens used by the model to generate the completion.
            extras (dict): This will contain any additional fields returned by corresponding provider.
    """
    # Determine which chat_completion function to call based on the client type
    if isinstance(client, OpenAI):
        response = chat_completion_openai.chat_completion(
            messages=messages,
            model=model,
            client=client,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            seed=seed,
            **kwargs,
        )
    elif isinstance(client, Client):
        response = chat_completion_ollama.chat_completion(
            messages=messages,
            model=model,
            client=client,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            seed=seed,
            **kwargs,
        )
    else:
        raise ValueError("Invalid client type")

    # Parse the responses to be consistent
    response_data = {}
    response_data["message"] = response.get("message", None)
    response_data["completion_tokens"] = response.get("completion_tokens", None)

    # Return any additional fields from the response in an "extras" dictionary
    extras = {k: v for k, v in response.items() if k not in response_data}
    if extras:
        response_data["extras"] = extras

    return response_data
