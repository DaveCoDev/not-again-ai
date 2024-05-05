# LLM (Large Language Model)

## Client
### openai_client.openai_client
This is required for making any call to the OpenAI API. This creates an OpenAI client object as described in this v1.0.0 beta [documentation](https://github.com/openai/openai-python/discussions/631).
Azure OpenAI is not currently supported by this function, but it is if you use the OpenAI package directly (via the AzureOpenAI client object).

## Ollama
### Installation
1. Follow the instructions to install ollama for your system: https://github.com/ollama/ollama
1. [Add Ollama as a startup service (recommended)](https://github.com/ollama/ollama/blob/main/docs/linux.md#adding-ollama-as-a-startup-service-recommended)
1. If you'd like to make the ollama service accessible on your local network and it is hosted on Linux, add the following to the `/etc/systemd/system/ollama.service` file:
    ```bash
    [Service]
    ...
    Environment="OLLAMA_HOST=0.0.0.0"
    ```
    Now ollama will be available at `http://<local_address>:11434`


## Chat Completion
### `chat_completion.chat_completion`
Use this to perform a standard chat completion. Takes in messages, a model name, and client object. Also exposes some community used optional parameters like max_tokens and tools for tool calling.
```python
client = openai_client()
messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
response = chat_completion(messages=messages, model="gpt-3.5-turbo", max_tokens=100, client=client)["message"]
>>> "Hello! How can I help you today?"
```

### prompts.chat_prompt
Injects variables into the chat completion prompt. sa
In the `messages_unformatted` argument, the "content" field can be a [Python Liquid](https://jg-rp.github.io/liquid/introduction/getting-started) template string to allow for more dynamic prompts.

### chat_completion_vision.chat_completion_vision
[Example Notebook](https://github.com/DaveCoDev/not-again-ai/blob/main/notebooks/llm/gpt-4-v.ipynb)

Use this to perform a chat completion with vision. Takes in messages, a model name, and client object. Also exposes some community used optional parameters.

### prompts.chat_prompt_vision
Formats a list of messages for OpenAI's chat completion API, for vision models only, using Liquid templating. 
Allows for easy injection of text variables and images provided as either paths or URLs.

```python
vision_prompt = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": ["Describe the animal in the image in one word.", {"image": cat_image_path, "detail": "low"}],
    },
    {"role": "assistant", "content": "{{ answer }}"},
    {
        "role": "user",
        "content": ["What about this animal?", {"image": dog_image_path, "detail": "low"}],
    },
]
```

### prompts.create_image_url
Takes in any Path object and returns the correctly formatted image URL for OpenAI's chat completion API as long as a valid MIME type is used.


## Text Embeddings
### embeddings.embed_text
Generates an embedding vector for a given text using OpenAI's API. It supports multiple text inputs and can return embeddings in either a list of floats or base64 encoded string.

Supported models include `text-embedding-3-small`, `text-embedding-3-large`, and `text-embedding-ada-002`.
Note that `text-embedding-ada-002` does not support specifying dimensions. The function handles this case and will raise an error if dimensions are specified for this model.


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