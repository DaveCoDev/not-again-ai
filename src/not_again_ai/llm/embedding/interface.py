from collections.abc import Callable
from typing import Any

from not_again_ai.llm.embedding.providers.ollama_api import ollama_create_embeddings
from not_again_ai.llm.embedding.providers.openai_api import openai_create_embeddings
from not_again_ai.llm.embedding.types import EmbeddingRequest, EmbeddingResponse


def create_embeddings(request: EmbeddingRequest, provider: str, client: Callable[..., Any]) -> EmbeddingResponse:
    """Get a embedding response from the given provider. Currently supported providers:
    - `openai` - OpenAI
    - `azure_openai` - Azure OpenAI
    - `ollama` - Ollama

    Args:
        request: Request parameter object
        provider: The supported provider name
        client: Client information, see the provider's implementation for what can be provided

    Returns:
        EmbeddingResponse: The embedding response.
    """
    if provider == "openai" or provider == "azure_openai":
        return openai_create_embeddings(request, client)
    elif provider == "ollama":
        return ollama_create_embeddings(request, client)
    else:
        raise ValueError(f"Provider {provider} not supported")
