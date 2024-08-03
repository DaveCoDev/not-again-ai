import os

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential


def azure_ai_client(
    token: str | None = None,
    endpoint: str = "https://models.inference.ai.azure.com",
) -> ChatCompletionsClient:
    if not token:
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("Token must be provided or GITHUB_TOKEN environment variable must be set")

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )
    return client
