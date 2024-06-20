import os

from ollama import Client


def ollama_client(host: str | None = None, timeout: float | None = None) -> Client:
    """Create an Ollama client instance based on the specified host or will read from the OLLAMA_HOST environment variable.

    Args:
        host (str, optional): The host URL of the Ollama server.
        timeout (float, optional): The timeout for requests

    Returns:
        Client: An instance of the Ollama client.

    Examples:
        >>> client = client(host="http://localhost:11434")
    """
    if host is None:
        host = os.getenv("OLLAMA_HOST")
        if host is None:
            raise ValueError("Host must be provided or OLLAMA_HOST environment variable must be set.")

    return Client(host=host, timeout=timeout)
