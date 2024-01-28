# LLM (Large Language Model)

## Client
### openai_client.openai_client
This is required for making any call to the OpenAI API. This creates an OpenAI client object as described in this v1.0.0 beta [documentation](https://github.com/openai/openai-python/discussions/631).
Azure OpenAI is not currently supported by this function, but it is if you use the OpenAI package directly (via the AzureOpenAI client object).

## Model Responses
### chat_completion.chat_completion
Use this to perform a standard chat completion. Takes in messages, a model name, and client object. Also exposes some community used optional parameters like max_tokens and tools for tool calling.
```python
client = openai_client()
messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
response = chat_completion(messages=messages, model="gpt-3.5-turbo", max_tokens=100, client=client)
```

### chat_completion.chat_prompt
Injects variables into the chat completion prompt. sa
In the `messages_unformatted` argument, the "content" field can be a [Python Liquid](https://jg-rp.github.io/liquid/introduction/getting-started) template string to allow for more dynamic prompts.

## Token Management
### tokens.truncate_str
Truncates a string to a certain length.

### tokens.num_tokens_from_messages
Return the number of tokens used by a list of messages. 
NOTE: Does not support counting tokens used by function calling.

## Context Management
Implements several methods to make injecting context into the chat prompt easier in relation to the max context window.

### context_management.priority_truncatation
Formats messages_unformatted and injects variables into the messages in the order of priority, truncating the messages to fit the token limit.