from collections.abc import Callable
from typing import Any

from not_again_ai.llm.chat_completion.providers.ollama_api import ollama_chat_completion
from not_again_ai.llm.chat_completion.providers.openai_api import openai_chat_completion
from not_again_ai.llm.chat_completion.types import ChatCompletionRequest, ChatCompletionResponse


def chat_completion(
    request: ChatCompletionRequest,
    provider: str,
    client: Callable[..., Any],
) -> ChatCompletionResponse:
    """Get a chat completion response from the given provider. Currently supported providers:
    - `openai` - OpenAI
    - `azure_openai` - Azure OpenAI
    - `ollama` - Ollama

    Args:
        request: Request parameter object
        provider: The supported provider name
        client: Client information, see the provider's implementation for what can be provided

    Returns:
        ChatCompletionResponse: The chat completion response.
    """
    if provider == "openai" or provider == "azure_openai":
        return openai_chat_completion(request, client)
    elif provider == "ollama":
        return ollama_chat_completion(request, client)
    else:
        raise ValueError(f"Provider {provider} not supported")
