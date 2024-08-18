from typing import Any

from azure.ai.inference import ChatCompletionsClient
from ollama import Client
from openai import AzureOpenAI, OpenAI

from not_again_ai.llm.gh_models import chat_completion as chat_completion_gh_models
from not_again_ai.llm.openai_api import chat_completion as chat_completion_openai
from not_again_ai.local_llm.ollama import chat_completion as chat_completion_ollama


def chat_completion(
    messages: list[dict[str, Any]],
    model: str,
    client: OpenAI | AzureOpenAI | Client | ChatCompletionsClient,
    tools: list[dict[str, Any]] | None = None,
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
        client (OpenAI | AzureOpenAI | Client | ChatCompletionsClient): The client object to use for chat completion.
        tools (list[dict[str, Any]], optional):A list of tools the model may call.
            Use this to provide a list of functions the model may generate JSON inputs for. Defaults to None.
        max_tokens (int, optional): The maximum number of tokens to generate.
        temperature (float, optional): The temperature of the model. Increasing the temperature will make the model answer more creatively.
        json_mode (bool, optional): This will structure the response as a valid JSON object.
        seed (int, optional): The seed to use for the model for reproducible outputs.

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
            extras (dict): This will contain any additional fields returned by corresponding provider.
    """
    # Determine which chat_completion function to call based on the client type
    if isinstance(client, OpenAI | AzureOpenAI):
        response = chat_completion_openai.chat_completion(
            messages=messages,
            model=model,
            client=client,
            tools=tools,
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
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            seed=seed,
            **kwargs,
        )
    elif isinstance(client, ChatCompletionsClient):
        response = chat_completion_gh_models.chat_completion(
            messages=messages,  # type: ignore
            model=model,
            client=client,
            tools=tools,  # type: ignore
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
    response_data["message"] = response.get("message")
    if response.get("tool_names") and response.get("tool_args_list"):
        response_data["tool_names"] = response.get("tool_names")
        response_data["tool_args_list"] = response.get("tool_args_list")
    response_data["completion_tokens"] = response.get("completion_tokens")
    response_data["prompt_tokens"] = response.get("prompt_tokens")
    response_data["response_duration"] = response.get("response_duration")

    # Return any additional fields from the response in an "extras" dictionary
    extras = {k: v for k, v in response.items() if k not in response_data}
    if extras:
        response_data["extras"] = extras

    return response_data
