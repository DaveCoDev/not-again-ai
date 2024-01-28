from openai import OpenAI


class InvalidOAIAPITypeError(Exception):
    """Raised when an invalid OAIAPIType string is provided."""

    pass


def openai_client(
    api_type: str = "openai",
    api_key: str | None = None,
    organization: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> OpenAI:
    """Create an OpenAI client instance based on the specified API type and other provided parameters.

    Args:
        api_type (str, optional): Type of the API to be used. Accepted values are 'openai' or 'azure_openai'.
            Defaults to 'openai'.
        api_key (str, optional): The API key to authenticate the client. If not provided,
            OpenAI automatically uses `OPENAI_API_KEY` from the environment.
        organization (str, optional): The ID of the organization. If not provided,
            OpenAI automotically uses `OPENAI_ORG_ID` from the environment.
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
        raise NotImplementedError("AzureOpenAI is not yet supported by not-again-ai.")
    else:
        raise NotImplementedError("This should never happen.")
