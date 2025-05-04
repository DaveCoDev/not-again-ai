import base64
from collections.abc import Callable
import time
from typing import Any, Literal

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI, OpenAI
from openai.types.images_response import ImagesResponse

from not_again_ai.llm.image_gen.types import ImageGenRequest, ImageGenResponse


def openai_create_image(request: ImageGenRequest, client: Callable[..., Any]) -> ImageGenResponse:
    """Create an image using OpenAI API.

    Args:
        request (ImageGenRequest): The request object containing parameters for image generation.
        client (Callable[..., Any]): The OpenAI client callable.

    Returns:
        ImageGenResponse: The response object containing the generated image and metadata.
    """
    kwargs = request.model_dump(exclude_none=True)
    if kwargs.get("images"):
        kwargs["image"] = kwargs.pop("images", None)

    start_time = time.time()
    response: ImagesResponse = client(**kwargs)
    end_time = time.time()
    response_duration = round(end_time - start_time, 4)

    images: list[bytes] = []
    if response.data:
        for data in response.data:
            images.append(base64.b64decode(data.b64_json or ""))

    input_tokens = response.usage.input_tokens if response.usage else -1
    output_tokens = response.usage.output_tokens if response.usage else -1
    input_tokens_details = response.usage.input_tokens_details.to_dict() if response.usage else {}
    image_gen_response = ImageGenResponse(
        images=images,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=input_tokens_details,
        response_duration=response_duration,
    )
    return image_gen_response


def create_client_callable(client_class: type[OpenAI | AzureOpenAI], **client_args: Any) -> Callable[..., Any]:
    """
    Creates the correct callable depending on the parameters provided.
    """
    filtered_args = {k: v for k, v in client_args.items() if v is not None}

    def client_callable(**kwargs: Any) -> Any:
        client = client_class(**filtered_args)
        # If mask or image is not none, use client.images.edit instead of client.images.generate
        if kwargs.get("mask") or kwargs.get("image"):
            completion = client.images.edit(**kwargs)
        else:
            completion = client.images.generate(**kwargs)
        return completion

    return client_callable


class InvalidOAIAPITypeError(Exception):
    """Raised when an invalid OAIAPIType string is provided."""


def openai_client(
    api_type: Literal["openai", "azure_openai"] = "openai",
    api_key: str | None = None,
    organization: str | None = None,
    aoai_api_version: str = "2024-06-01",
    azure_endpoint: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> Callable[..., Any]:
    """Create an OpenAI or Azure OpenAI client instance based on the specified API type and other provided parameters.

    It is preferred to use RBAC authentication for Azure OpenAI. You must be signed in with the Azure CLI and have correct role assigned.
    See https://techcommunity.microsoft.com/t5/microsoft-developer-community/using-keyless-authentication-with-azure-openai/ba-p/4111521

    Args:
        api_type (str, optional): Type of the API to be used. Accepted values are 'openai' or 'azure_openai'.
            Defaults to 'openai'.
        api_key (str, optional): The API key to authenticate the client. If not provided,
            OpenAI automatically uses `OPENAI_API_KEY` from the environment.
            If provided for Azure OpenAI, it will be used for authentication instead of the Azure AD token provider.
        organization (str, optional): The ID of the organization. If not provided,
            OpenAI automotically uses `OPENAI_ORG_ID` from the environment.
        aoai_api_version (str, optional): Only applicable if using Azure OpenAI https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
        azure_endpoint (str, optional): The endpoint to use for Azure OpenAI.
        timeout (float, optional): By default requests time out after 10 minutes.
        max_retries (int, optional): Certain errors are automatically retried 2 times by default,
            with a short exponential backoff. Connection errors (for example, due to a network connectivity problem),
            408 Request Timeout, 409 Conflict, 429 Rate Limit, and >=500 Internal errors are all retried by default.

    Returns:
        Callable[..., Any]: A callable that creates a client and returns completion results


    Raises:
        InvalidOAIAPITypeError: If an invalid API type string is provided.
        NotImplementedError: If the specified API type is recognized but not yet supported (e.g., 'azure_openai').
    """
    if api_type not in ["openai", "azure_openai"]:
        raise InvalidOAIAPITypeError(f"Invalid OAIAPIType: {api_type}. Must be 'openai' or 'azure_openai'.")

    if api_type == "openai":
        return create_client_callable(
            OpenAI,
            api_key=api_key,
            organization=organization,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif api_type == "azure_openai":
        if api_key:
            return create_client_callable(
                AzureOpenAI,
                api_version=aoai_api_version,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            azure_credential = DefaultAzureCredential()
            ad_token_provider = get_bearer_token_provider(
                azure_credential, "https://cognitiveservices.azure.com/.default"
            )
            return create_client_callable(
                AzureOpenAI,
                api_version=aoai_api_version,
                azure_endpoint=azure_endpoint,
                azure_ad_token_provider=ad_token_provider,
                timeout=timeout,
                max_retries=max_retries,
            )
    else:
        raise NotImplementedError(f"API type '{api_type}' is invalid.")
