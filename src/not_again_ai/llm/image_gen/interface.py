from collections.abc import Callable
from typing import Any

from not_again_ai.llm.image_gen.providers.openai_api import openai_create_image
from not_again_ai.llm.image_gen.types import ImageGenRequest, ImageGenResponse


def create_image(request: ImageGenRequest, provider: str, client: Callable[..., Any]) -> ImageGenResponse:
    """Get a image response from the given provider. Currently supported providers:
    - `openai` - OpenAI
    - `azure_openai` - Azure OpenAI

    Args:
        request: Request parameter object
        provider: The supported provider name
        client: Client information, see the provider's implementation for what can be provided

    Returns:
        ImageGenResponse: The image generation response.
    """
    if provider == "openai" or provider == "azure_openai":
        return openai_create_image(request, client)
    else:
        raise ValueError(f"Provider {provider} not supported")
