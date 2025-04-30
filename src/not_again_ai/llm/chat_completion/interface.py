from collections.abc import AsyncGenerator, Callable
from typing import Any

from not_again_ai.llm.chat_completion.providers.anthropic_api import anthropic_chat_completion
from not_again_ai.llm.chat_completion.providers.gemini_api import gemini_chat_completion
from not_again_ai.llm.chat_completion.providers.ollama_api import ollama_chat_completion, ollama_chat_completion_stream
from not_again_ai.llm.chat_completion.providers.openai_api import openai_chat_completion, openai_chat_completion_stream
from not_again_ai.llm.chat_completion.types import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse


def chat_completion(
    request: ChatCompletionRequest,
    provider: str,
    client: Callable[..., Any],
) -> ChatCompletionResponse:
    """Get a chat completion response from the given provider. Currently supported providers:
    - `openai` - OpenAI
    - `azure_openai` - Azure OpenAI
    - `ollama` - Ollama
    - `anthropic` - Anthropic
    - `gemini` - Gemini

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
    elif provider == "anthropic":
        return anthropic_chat_completion(request, client)
    elif provider == "gemini":
        return gemini_chat_completion(request, client)
    else:
        raise ValueError(f"Provider {provider} not supported")


async def chat_completion_stream(
    request: ChatCompletionRequest,
    provider: str,
    client: Callable[..., Any],
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Stream a chat completion response from the given provider. Currently supported providers:
    - `openai` - OpenAI
    - `azure_openai` - Azure OpenAI

    Args:
        request: Request parameter object
        provider: The supported provider name
        client: Client information, see the provider's implementation for what can be provided

    Returns:
        AsyncGenerator[ChatCompletionChunk, None]
    """
    request.stream = True
    if provider == "openai" or provider == "azure_openai":
        async for chunk in openai_chat_completion_stream(request, client):
            yield chunk
    elif provider == "ollama":
        async for chunk in ollama_chat_completion_stream(request, client):
            yield chunk
    else:
        raise ValueError(f"Provider {provider} not supported")
