import contextlib
import json
from typing import Any

from openai import OpenAI


def chat_completion(
    messages: list[dict[str, Any]],
    model: str,
    client: OpenAI,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    max_tokens: int | None = None,
    temperature: float = 0.7,
    json_mode: bool = False,
    seed: int | None = None,
    logprobs: tuple[bool, int | None] | None = None,
    n: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get an OpenAI chat completion response: https://platform.openai.com/docs/api-reference/chat/create

    NOTE: Depending on the model, certain parameters may not be supported,
    particularly for older vision-enabled models like gpt-4-1106-vision-preview.
    Be sure to check the documentation: https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4

    Args:
        messages (list): A list of messages comprising the conversation so far.
        model (str): ID of the model to use. See the model endpoint compatibility table:
            https://platform.openai.com/docs/models/model-endpoint-compatibility
            for details on which models work with the Chat API.
        client (OpenAI): An instance of the OpenAI client.
        tools (list[dict[str, Any]], optional): A list of tools the model may generate JSON inputs for.
            Defaults to None.
        tool_choice (str, optional): The tool choice to use. Can be "auto", "required", "none", or a specific function name.
            Note the function name cannot be any of "auto", "required", or "none". Defaults to "auto".
        max_tokens (int, optional): The maximum number of tokens to generate in the chat completion.
            Defaults to None, which automatically limits to the model's maximum context length.
        temperature (float, optional): What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic. Defaults to 0.7.
        json_mode (bool, optional): When JSON mode is enabled, the model is constrained to only
            generate strings that parse into valid JSON object and will return a dictionary.
            See https://platform.openai.com/docs/guides/text-generation/json-mode
        seed (int, optional): If specified, OpenAI will make a best effort to sample deterministically,
            such that repeated requests with the same `seed` and parameters should return the same result.
            Determinism is not guaranteed, and you should refer to the `system_fingerprint` response
            parameter to monitor changes in the backend.
        logprobs (tuple[bool, int], optional): Whether to return log probabilities of the output tokens or not.
            If `logprobs[0]` is true, returns the log probabilities of each output token returned in the content of message.
            `logprobs[1]` is an integer between 0 and 5 specifying the number of most likely tokens to return at each token position,
            each with an associated log probability. `logprobs[0]` must be set to true if this parameter is used.
        n (int, optional): How many chat completion choices to generate for each input message.
            Defaults to 1.
        **kwargs: Additional keyword arguments to pass to the OpenAI client chat completion.

    Returns:
        dict[str, Any]: A dictionary with the following keys:
            'finish_reason' (str): The reason the model stopped generating further tokens.
                Can be 'stop', 'length', or 'tool_calls'.
            'tool_names' (list[str], optional): The names of the tools called by the model.
            'tool_args_list' (list[dict], optional): The arguments of the tools called by the model.
            'message' (str | dict): The content of the generated assistant message.
                If json_mode is True, this will be a dictionary.
            'logprobs' (list[dict[str, Any] | list[dict[str, Any]]]): If logprobs[1] is between 1 and 5, each element in the list
                will be a list of dictionaries containing the token, logprob, and bytes for the top `logprobs[1]` logprobs. Otherwise,
                this will be a list of dictionaries containing the token, logprob, and bytes for each token in the message.
            'choices' (list[dict], optional): A list of chat completion choices if n > 1 where each dict contains the above fields.
            'completion_tokens' (int): The number of tokens used by the model to generate the completion.
                NOTE: If n > 1 this is the sum of all completions.
            'prompt_tokens' (int): The number of tokens in the messages sent to the model.
            'system_fingerprint' (str, optional): If seed is set, a unique identifier for the model used to generate the response.
    """
    response_format = {"type": "json_object"} if json_mode else None

    kwargs.update(
        {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format,
            "n": n,
        }
    )

    if tools is not None:
        kwargs["tools"] = tools
        if tool_choice not in ["none", "auto", "required"]:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        else:
            kwargs["tool_choice"] = tool_choice

    if seed is not None:
        kwargs["seed"] = seed

    if logprobs is not None:
        kwargs["logprobs"] = logprobs[0]
        if logprobs[0] and logprobs[1] is not None:
            kwargs["top_logprobs"] = logprobs[1]

    response = client.chat.completions.create(**kwargs)

    response_data: dict[str, Any] = {"choices": []}
    for response_choice in response.choices:
        response_data_curr = {}
        finish_reason = response_choice.finish_reason
        response_data_curr["finish_reason"] = finish_reason

        # We first check for tool calls because even if the finish_reason is stop, the model may have called a tool
        tool_calls = response_choice.message.tool_calls
        if tool_calls:
            tool_names = []
            tool_args_list = []
            for tool_call in tool_calls:
                tool_names.append(tool_call.function.name)
                tool_args_list.append(json.loads(tool_call.function.arguments))
            response_data_curr["message"] = response_choice.message.content
            response_data_curr["tool_names"] = tool_names
            response_data_curr["tool_args_list"] = tool_args_list
        elif finish_reason == "stop" or finish_reason == "length":
            message = response_choice.message.content
            if json_mode:
                with contextlib.suppress(json.JSONDecodeError):
                    message = json.loads(message)
            response_data_curr["message"] = message

        if response_choice.logprobs and response_choice.logprobs.content is not None:
            logprobs_list: list[dict[str, Any] | list[dict[str, Any]]] = []
            for logprob in response_choice.logprobs.content:
                if logprob.top_logprobs:
                    curr_logprob_infos = []
                    for top_logprob in logprob.top_logprobs:
                        curr_logprob_infos.append(
                            {
                                "token": top_logprob.token,
                                "logprob": top_logprob.logprob,
                                "bytes": top_logprob.bytes,
                            }
                        )
                    logprobs_list.append(curr_logprob_infos)
                else:
                    logprobs_list.append(
                        {
                            "token": logprob.token,
                            "logprob": logprob.logprob,
                            "bytes": logprob.bytes,
                        }
                    )

            response_data_curr["logprobs"] = logprobs_list
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
