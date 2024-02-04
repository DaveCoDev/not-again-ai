from not_again_ai.llm.chat_completion import chat_completion
from not_again_ai.llm.openai_client import openai_client


def test_chat_completion() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(messages=messages, model="gpt-3.5-turbo-0125", max_tokens=100, client=client)
    print(response)


def test_chat_completion_length() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(messages=messages, model="gpt-3.5-turbo-0125", max_tokens=2, client=client)
    print(response)


def test_chat_completion_expected_function() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the current weather like in Boston, MA today? Call the get_current_weather function.",
        }
    ]
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-3.5-turbo-0125",
        client=client,
        tools=tools,
        max_tokens=300,
        temperature=0,
    )
    print(response)


def test_chat_completion_tool_choice() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the current weather like in Boston, MA today?",
        }
    ]
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-3.5-turbo-0125",
        client=client,
        tools=tools,
        tool_choice="get_current_weather",
        max_tokens=300,
        temperature=0,
    )
    print(response)


def test_chat_completion_multiple_functions() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "system",
            "content": "Call the get_current_weather function once for each city that the user mentions.",
        },
        {
            "role": "user",
            "content": "What's the current weather like in Boston, MA and New York, NY today?",
        },
    ]
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-3.5-turbo-0125",
        client=client,
        tools=tools,
        max_tokens=400,
        temperature=0,
    )
    print(response)


def test_dont_call_function() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "system",
            "content": "Call the get_current_weather function once for each city that the user mentions.",
        },
        {
            "role": "user",
            "content": "What's the current weather like in Boston, MA and New York, NY today?",
        },
    ]
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-3.5-turbo-0125",
        client=client,
        tools=tools,
        tool_choice="none",
        max_tokens=200,
        temperature=0,
    )
    print(response)


def test_json_mode() -> None:
    messages = [
        {
            "role": "system",
            "content": """You are getting names of users and formatting them into json.
Example:
User: Jane Doe
Output: {"name": "Jane Doe"}""",
        },
        {
            "role": "user",
            "content": "John Doe",
        },
    ]
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-3.5-turbo-0125",
        client=client,
        max_tokens=300,
        temperature=0,
        json_mode=True,
    )
    print(response)


def test_chat_completion_n() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(messages=messages, model="gpt-3.5-turbo-0125", max_tokens=100, client=client, n=2)
    print(response)


def test_chat_completion_seed() -> None:
    client = openai_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate a random number between 0 and 100."},
    ]
    response_1 = chat_completion(
        messages=messages, model="gpt-3.5-turbo-0125", max_tokens=100, client=client, temperature=2, seed=42
    )

    response_2 = chat_completion(
        messages=messages, model="gpt-3.5-turbo-0125", max_tokens=100, client=client, temperature=2, seed=42
    )

    print(response_1)
    print(response_1["message"])
    print(response_2["message"])

    assert "system_fingerprint" in response_1

    # Add this assertion when outputs become guaranteed to be the same. It currently fails roughly 1 in 5 times.
    # if response_1["system_fingerprint"] == response_2["system_fingerprint"]:
    #   assert response_1["message"] == response_2["message"]


def test_chat_completion_logprobs() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(
        messages=messages, model="gpt-3.5-turbo-0125", max_tokens=100, client=client, logprobs=(True, None)
    )
    print(response)


def test_chat_completion_toplogprobs() -> None:
    client = openai_client()
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(
        messages=messages, model="gpt-3.5-turbo-0125", max_tokens=100, client=client, logprobs=(True, 3)
    )
    print(response)


def test_chat_completion_misc_1() -> None:
    client = openai_client()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the current weather like in Boston, MA today? Call the get_current_weather function.",
        }
    ]
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-3.5-turbo-0125",
        client=client,
        tools=tools,
        max_tokens=300,
        temperature=0,
        logprobs=(True, 2),
        seed=42,
        n=2,
    )
    # NOTE: When a function is called, logprobs are not returned.
    print(response)


def test_chat_completion_misc_2() -> None:
    messages = [
        {
            "role": "system",
            "content": """You are getting names of users and formatting them into json.
Example:
User: Jane Doe
Output: {"name": "Jane Doe"}""",
        },
        {
            "role": "user",
            "content": "John Doe",
        },
    ]
    client = openai_client()
    response = chat_completion(
        messages=messages,
        model="gpt-4-0125-preview",
        client=client,
        max_tokens=200,
        temperature=0,
        json_mode=True,
        logprobs=(True, None),
        seed=-5,
        n=2,
    )
    print(response)


test_chat_completion_misc_1()
