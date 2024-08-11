from typing import Literal

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI, OpenAI


class InvalidOAIAPITypeError(Exception):
    """Raised when an invalid OAIAPIType string is provided."""

    pass


def openai_client(
    api_type: Literal["openai", "azure_openai"] = "openai",
    api_key: str | None = None,
    organization: str | None = None,
    aoai_api_version: str = "2024-06-01",
    azure_endpoint: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> OpenAI | AzureOpenAI:
    """Create an OpenAI or Azure OpenAI client instance based on the specified API type and other provided parameters.

    Azure OpenAI requires RBAC authentication. You must be signed in with the Azure CLI and have correct role assigned.
    See https://techcommunity.microsoft.com/t5/microsoft-developer-community/using-keyless-authentication-with-azure-openai/ba-p/4111521

    Args:
        api_type (str, optional): Type of the API to be used. Accepted values are 'openai' or 'azure_openai'.
            Defaults to 'openai'.
        api_key (str, optional): The API key to authenticate the client. If not provided,
            OpenAI automatically uses `OPENAI_API_KEY` from the environment.
        organization (str, optional): The ID of the organization. If not provided,
            OpenAI automotically uses `OPENAI_ORG_ID` from the environment.
        aoai_api_version (str, optional): Only applicable if using Azure OpenAI https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
        azure_endpoint (str, optional): The endpoint to use for Azure OpenAI.
            If not provided, will be read from the `AZURE_OPENAI_ENDPOINT` environment variable.
        timeout (float, optional): By default requests time out after 10 minutes.
        max_retries (int, optional): Certain errors are automatically retried 2 times by default,
            with a short exponential backoff. Connection errors (for example, due to a network connectivity problem),
            408 Request Timeout, 409 Conflict, 429 Rate Limit, and >=500 Internal errors are all retried by default.

    Returns:
        OpenAI: An instance of the OpenAI client.

    Raises:
        InvalidOAIAPITypeError: If an invalid API type string is provided.
        NotImplementedError: If the specified API type is recognized but not yet supported (e.g., 'azure_openai').

    Examples:
        >>> client = openai_client(api_type="openai", api_key="YOUR_API_KEY")
    """
    if api_type not in ["openai", "azure_openai"]:
        raise InvalidOAIAPITypeError(f"Invalid OAIAPIType: {api_type}. Must be 'openai' or 'azure_openai'.")

    if api_type == "openai":
        args = {
            "api_key": api_key,
            "organization": organization,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        # Remove any None values in order to use the default values.
        filtered_args = {k: v for k, v in args.items() if v is not None}
        return OpenAI(**filtered_args)  # type: ignore
    elif api_type == "azure_openai":
        azure_credential = DefaultAzureCredential()
        ad_token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")
        args = {
            "api_version": aoai_api_version,
            "azure_endpoint": azure_endpoint,
            "azure_ad_token_provider": ad_token_provider,  # type: ignore
            "timeout": timeout,
            "max_retries": max_retries,
        }
        filtered_args = {k: v for k, v in args.items() if v is not None}
        return AzureOpenAI(**filtered_args)  # type: ignore
    else:
        raise NotImplementedError(f"API type '{api_type}' is invalid.")
