from collections.abc import Generator
import contextlib
import json
import time
from typing import Any

from openai import AzureOpenAI, OpenAI


def chat_completion(
    messages: list[dict[str, Any]],
    model: str,
    client: OpenAI | AzureOpenAI | Any,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    max_tokens: int | None = None,
    temperature: float = 0.7,
    json_mode: bool = False,
    json_schema: dict[str, Any] | None = None,
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
        client (OpenAI | AzureOpenAI | Any): An instance of the OpenAI or AzureOpenAI client.
            If anything else is provided, we assume that it follows the OpenAI spec and call it by passing kwargs directly.
            For example you can provide something like:
            ```
            def custom_client(**kwargs):
                client = openai_client()
                completion = client.chat.completions.create(**kwargs)
                return completion.to_dict()
            ```
        tools (list[dict[str, Any]], optional):A list of tools the model may call.
            Use this to provide a list of functions the model may generate JSON inputs for. Defaults to None.
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
        json_schema (dict, optional): Enables Structured Outputs which ensures the model will
            always generate responses that adhere to your supplied JSON Schema.
            See https://platform.openai.com/docs/guides/structured-outputs/structured-outputs
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
            finish_reason (str): The reason the model stopped generating further tokens.
                Can be 'stop', 'length', or 'tool_calls'.
            tool_names (list[str], optional): The names of the tools called by the model.
            tool_args_list (list[dict], optional): The arguments of the tools called by the model.
            message (str | dict): The content of the generated assistant message.
                If json_mode is True, this will be a dictionary.
            logprobs (list[dict[str, Any] | list[dict[str, Any]]]): If logprobs[1] is between 1 and 5, each element in the list
                will be a list of dictionaries containing the token, logprob, and bytes for the top `logprobs[1]` logprobs. Otherwise,
                this will be a list of dictionaries containing the token, logprob, and bytes for each token in the message.
            choices (list[dict], optional): A list of chat completion choices if n > 1 where each dict contains the above fields.
            completion_tokens (int): The number of tokens used by the model to generate the completion.
                NOTE: If n > 1 this is the sum of all completions.
            prompt_tokens (int): The number of tokens in the messages sent to the model.
            system_fingerprint (str, optional): If seed is set, a unique identifier for the model used to generate the response.
            response_duration (float): The time, in seconds, taken to generate the response from the API.
    """

    if json_mode and json_schema is not None:
        raise ValueError("json_schema and json_mode cannot be used together.")

    if json_mode:
        response_format: dict[str, Any] = {"type": "json_object"}
    elif json_schema is not None:
        if isinstance(json_schema, dict):
            response_format = {"type": "json_schema", "json_schema": json_schema}
    else:
        response_format = {"type": "text"}

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

    start_time = time.time()
    if isinstance(client, OpenAI | AzureOpenAI):
        response = client.chat.completions.create(**kwargs)
        response = response.to_dict()
    else:
        response = client(**kwargs)
    end_time = time.time()
    response_duration = end_time - start_time

    response_data: dict[str, Any] = {"choices": []}
    for response_choice in response["choices"]:
        response_data_curr = {}
        finish_reason = response_choice["finish_reason"]
        response_data_curr["finish_reason"] = finish_reason

        # We first check for tool calls because even if the finish_reason is stop, the model may have called a tool
        tool_calls = response_choice["message"].get("tool_calls", None)
        if tool_calls:
            tool_names = []
            tool_args_list = []
            for tool_call in tool_calls:
                tool_names.append(tool_call["function"]["name"])
                tool_args_list.append(json.loads(tool_call["function"]["arguments"]))
            response_data_curr["message"] = response_choice["message"]["content"]
            response_data_curr["tool_names"] = tool_names
            response_data_curr["tool_args_list"] = tool_args_list
        elif finish_reason == "stop" or finish_reason == "length":
            message = response_choice["message"]["content"]
            if json_mode or json_schema is not None:
                with contextlib.suppress(json.JSONDecodeError):
                    message = json.loads(message)
            response_data_curr["message"] = message

        if response_choice["logprobs"] and response_choice["logprobs"]["content"] is not None:
            logprobs_list: list[dict[str, Any] | list[dict[str, Any]]] = []
            for logprob in response_choice["logprobs"]["content"]:
                if logprob["top_logprobs"]:
                    curr_logprob_infos = []
                    for top_logprob in logprob["top_logprobs"]:
                        curr_logprob_infos.append(
                            {
                                "token": top_logprob["token"],
                                "logprob": top_logprob["logprob"],
                                "bytes": top_logprob["bytes"],
                            }
                        )
                    logprobs_list.append(curr_logprob_infos)
                else:
                    logprobs_list.append(
                        {
                            "token": logprob["token"],
                            "logprob": logprob["logprob"],
                            "bytes": logprob["bytes"],
                        }
                    )

            response_data_curr["logprobs"] = logprobs_list
        response_data["choices"].append(response_data_curr)

    usage = response["usage"]
    if usage is not None:
        response_data["completion_tokens"] = usage["completion_tokens"]
        response_data["prompt_tokens"] = usage["prompt_tokens"]

    if seed is not None and response["system_fingerprint"] is not None:
        response_data["system_fingerprint"] = response["system_fingerprint"]

    response_data["response_duration"] = round(response_duration, 4)

    if len(response_data["choices"]) == 1:
        response_data.update(response_data["choices"][0])
        del response_data["choices"]

    return response_data


def chat_completion_stream(
    messages: list[dict[str, Any]],
    model: str,
    client: OpenAI | AzureOpenAI | Any,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    max_tokens: int | None = None,
    temperature: float = 0.7,
    seed: int | None = None,
    **kwargs: Any,
) -> Generator[dict[str, Any], None, None]:
    """Stream a chat completion from the OpenAI API.

    Args:
        messages (list[dict[str, Any]]): The messages to send to the model.
        model (str): The model to use for the chat completion.
        client (OpenAI | AzureOpenAI | Any): The client to use to send the request.
            If anything else is provided, we assume that it follows the OpenAI spec and call it by passing kwargs directly.
            For example you can provide something like:
            ```
            def custom_client(**kwargs) -> Generator[dict[str, Any], None, None]:  # type: ignore
                client = openai_client()
                completion = client.chat.completions.create(**kwargs)
                for chunk in completion:
                    yield chunk.to_dict()
            ```
        tools (list[dict[str, Any]], optional):A list of tools the model may call.
            Use this to provide a list of functions the model may generate JSON inputs for. Defaults to None.
        tool_choice (str, optional): The tool choice to use. Can be "auto", "required", "none", or a specific function name.
            Note the function name cannot be any of "auto", "required", or "none". Defaults to "auto".
        max_tokens (int | None): The maximum number of tokens to generate.
        temperature (float): The temperature to use for the chat completion.
        seed (int, optional): If specified, OpenAI will make a best effort to sample deterministically,
            such that repeated requests with the same `seed` and parameters should return the same result.
            Does not currently return `system_fingerprint`.

    Returns:
        Generator[dict[str, Any], None, None]: A generator of chunks of the chat completion.
        Each chunk is a dictionary with the following keys:
            role (str): The role of the chunk. Can be "assistant", "tool", or "usage".
            content (str): The content of the chunk.
            tool_name (str | None): The name of the tool called by the model.
            tool_call_id (str | None): The ID of the tool call.
            completion_tokens (int | None): The number of tokens used by the model to generate the completion.
            prompt_tokens (int | None): The number of tokens in the messages sent to the model.
    """

    class ChatCompletionStreamParser:
        def __init__(self) -> None:
            # Remembers if we are currently streaming an assistant message or tool call
            self.last_type: str = ""
            self.last_tool_name: str | None = None
            self.last_tool_call_id: str | None = None

        def process_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
            """Convert the current chunk into a more digestible format
            {
                "role": Literal["assistant", "tool", "usage"],
                "content": str,
                "tool_name": str | None,
                "tool_call_id": str | None,
                "completion_tokens": int | None,
                "prompt_tokens": int | None,
            }
            """
            processed_chunk: dict[str, Any] = {}
            if chunk["choices"]:
                choice = chunk["choices"][0]
                # This checks if its just a regular message currently being streamed
                if choice["delta"].get("role", "") and choice["delta"].get("tool_calls", None) is None:
                    if choice["delta"]["role"] != self.last_type:
                        self.last_type = choice["delta"]["role"]
                        processed_chunk["role"] = self.last_type
                        if not choice["delta"]["content"]:
                            processed_chunk["content"] = ""
                        else:
                            processed_chunk["content"] = choice["delta"]["content"]
                    else:
                        processed_chunk["role"] = self.last_type
                elif choice["delta"].get("tool_calls", None):
                    # tool_calls will always be present if the model is calling a tool
                    tool_call = choice["delta"]["tool_calls"][0]
                    if tool_call["function"].get("name"):
                        self.last_type = "tool"
                        self.last_tool_name = tool_call["function"]["name"]
                        self.last_tool_call_id = tool_call["id"]
                    processed_chunk["role"] = "tool"
                    processed_chunk["content"] = tool_call["function"]["arguments"]
                    processed_chunk["tool_name"] = self.last_tool_name
                    processed_chunk["tool_call_id"] = self.last_tool_call_id
                elif choice["delta"].get("content", ""):
                    # This is the case after the first regular assistant message
                    processed_chunk["role"] = self.last_type
                    processed_chunk["content"] = choice["delta"]["content"]
            else:
                if chunk.get("usage"):
                    processed_chunk["role"] = "usage"
                    processed_chunk["completion_tokens"] = chunk["usage"]["completion_tokens"]
                    processed_chunk["prompt_tokens"] = chunk["usage"]["prompt_tokens"]
                else:
                    return None
            return processed_chunk

    kwargs.update(
        {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
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

    if isinstance(client, OpenAI | AzureOpenAI):
        response = client.chat.completions.create(**kwargs)
    else:
        response = client(**kwargs)

    parser = ChatCompletionStreamParser()
    for chunk in response:
        if isinstance(client, OpenAI | AzureOpenAI):
            chunk = chunk.to_dict()
        processed_chunk = parser.process_chunk(chunk)
        if processed_chunk:
            yield processed_chunk
