from collections.abc import Callable
from typing import Any

import pytest

from not_again_ai.llm.embedding import EmbeddingRequest, create_embeddings
from not_again_ai.llm.embedding.providers.ollama_api import ollama_client
from not_again_ai.llm.embedding.providers.openai_api import openai_client
from not_again_ai.llm.embedding.types import EmbeddingResponse


def print_embedding_response(embedding_response: EmbeddingResponse, max_elements: int = 5) -> None:
    """Print an EmbeddingResponse, truncating each embedding vector after `max_elements` values.

    Args:
        embedding_response (EmbeddingResponse): The response to print.
        max_elements (int): Maximum number of elements from each embedding vector to display.
                            Defaults to 5.
    """
    print("EmbeddingResponse:")
    print(f"  Response Duration: {embedding_response.response_duration}")
    print(f"  Total Tokens: {embedding_response.total_tokens}")
    print(f"  Extras: {embedding_response.extras}")
    print("  Embeddings:")
    for obj in embedding_response.embeddings:
        vector = obj.embedding
        if len(vector) > max_elements:
            # Format the first max_elements, then append an ellipsis.
            vector_str = ", ".join(f"{value:.4f}" for value in vector[:max_elements])
            vector_str += ", ..."
        else:
            vector_str = ", ".join(f"{value:.4f}" for value in vector)
        print(f"    Index {obj.index}: [{vector_str}]")


# region OpenAI and Azure OpenAI Embedding
@pytest.fixture(
    params=[
        {},
        {"api_type": "azure_openai", "aoai_api_version": "2025-01-01-preview"},
    ]
)
def openai_aoai_client_fixture(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return openai_client(**request.param)


def test_create_embeddings(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = EmbeddingRequest(input="Hello, world!", model="text-embedding-3-small")
    response = create_embeddings(request, "openai", openai_aoai_client_fixture)
    print_embedding_response(response)


def test_create_embeddings_multiple(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = EmbeddingRequest(input=["Hello, world!", "Hello, world 2!"], model="text-embedding-3-small")
    response = create_embeddings(request, "openai", openai_aoai_client_fixture)
    print_embedding_response(response)


def test_create_embeddings_dimensions(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = EmbeddingRequest(
        input="This is a test of the dimensions parameter",
        model="text-embedding-3-large",
        dimensions=3,
    )
    response = create_embeddings(request, "openai", openai_aoai_client_fixture)
    print_embedding_response(response)


# endregion
# region Ollama
@pytest.fixture(
    params=[
        {},
    ]
)
def ollama_client_fixture(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return ollama_client(**request.param)


def test_create_embeddings_ollama(ollama_client_fixture: Callable[..., Any]) -> None:
    request = EmbeddingRequest(input="Hello, world!", model="snowflake-arctic-embed2")
    response = create_embeddings(request, "ollama", ollama_client_fixture)
    print_embedding_response(response)


def test_create_embeddings_ollama_missing_param(ollama_client_fixture: Callable[..., Any]) -> None:
    request = EmbeddingRequest(input="Hello, world!", model="snowflake-arctic-embed2", dimensions=3)
    response = create_embeddings(request, "ollama", ollama_client_fixture)
    print_embedding_response(response)


def test_create_embeddings_ollama_multiple(ollama_client_fixture: Callable[..., Any]) -> None:
    request = EmbeddingRequest(input=["Hello, world!", "Hello, world 2!"], model="snowflake-arctic-embed2")
    response = create_embeddings(request, "ollama", ollama_client_fixture)
    print_embedding_response(response)


# endregion
