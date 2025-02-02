from collections.abc import Callable
import os
import re
import time
from typing import Any

from loguru import logger
from ollama import Client, EmbedResponse, ResponseError

from not_again_ai.llm.embedding.types import EmbeddingObject, EmbeddingRequest, EmbeddingResponse

OLLAMA_PARAMETER_MAP = {
    "dimensions": None,
}


def validate(request: EmbeddingRequest) -> None:
    # Check if any of the parameters set to OLLAMA_PARAMETER_MAP are not None
    for key, value in OLLAMA_PARAMETER_MAP.items():
        if value is None and getattr(request, key) is not None:
            logger.warning(f"Parameter {key} is not supported by Ollama and will be ignored.")


def ollama_create_embeddings(request: EmbeddingRequest, client: Callable[..., Any]) -> EmbeddingResponse:
    validate(request)
    kwargs = request.model_dump(mode="json", exclude_none=True)

    # For each key in OLLAMA_PARAMETER_MAP
    # If it is not None, set the key in kwargs to the value of the corresponding value in OLLAMA_PARAMETER_MAP
    # If it is None, remove that key from kwargs
    for key, value in OLLAMA_PARAMETER_MAP.items():
        if value is not None and key in kwargs:
            kwargs[value] = kwargs.pop(key)
        elif value is None and key in kwargs:
            del kwargs[key]

    # Explicitly set truncate to True (it is the default)
    kwargs["truncate"] = True

    try:
        start_time = time.time()
        response: EmbedResponse = client(**kwargs)
        end_time = time.time()
        response_duration = round(end_time - start_time, 4)
    except ResponseError as e:
        # If the error says "model 'model' not found" use regex then raise a more specific error
        expected_pattern = f"model '{request.model}' not found"
        if re.search(expected_pattern, e.error):
            raise ResponseError(f"Model '{request.model}' not found.") from e
        else:
            raise ResponseError(e.error) from e

    embeddings: list[EmbeddingObject] = []
    for index, embedding in enumerate(response.embeddings):
        embeddings.append(EmbeddingObject(embedding=list(embedding), index=index))

    return EmbeddingResponse(
        embeddings=embeddings,
        response_duration=response_duration,
        total_tokens=response.prompt_eval_count,
    )


def ollama_client(host: str | None = None, timeout: float | None = None) -> Callable[..., Any]:
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
            logger.warning("OLLAMA_HOST environment variable not set, using default host: http://localhost:11434")
            host = "http://localhost:11434"

    def client_callable(**kwargs: Any) -> Any:
        client = Client(host=host, timeout=timeout)
        return client.embed(**kwargs)

    return client_callable
