from typing import Any

from openai import OpenAI


def embed_text(
    text: str | list[str],
    client: OpenAI,
    model: str = "text-embedding-3-large",
    dimensions: int | None = None,
    encoding_format: str = "float",
    **kwargs: Any,
) -> list[float] | str | list[list[float]] | list[str]:
    """Generates an embedding vector for a given text using OpenAI's API.

    Args:
        text (str | list[str]): The input text to be embedded. Each text should not exceed 8191 tokens, which is the max for V2 and V3 models
        client (OpenAI): The OpenAI client used to interact with the API.
        model (str, optional): The ID of the model to use for embedding.
            Defaults to "text-embedding-3-large".
            Choose from text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002.
            See https://platform.openai.com/docs/models/embeddings for more details.
        dimensions (int | None, optional): The number of dimensions for the output embeddings.
            This is only supported in "text-embedding-3" and later models. Defaults to None.
        encoding_format (str, optional): The format for the returned embeddings. Can be either "float" or "base64".
            Defaults to "float".

    Returns:
        list[float] | str | list[list[float]] | list[str]: The embedding vector represented as a list of floats or base64 encoded string.
            If multiple text inputs are provided, a list of embedding vectors is returned.
            The length and format of the vector depend on the model, encoding_format, and dimensions.

    Raises:
        ValueError: If 'text-embedding-ada-002' model is used and dimensions are specified,
            as this model does not support specifying dimensions.

    Example:
        client = OpenAI()
        embedding = embed_text("Example text", client, model="text-embedding-ada-002")
    """
    if model == "text-embedding-ada-002" and dimensions:
        # text-embedding-ada-002 does not support dimensions
        raise ValueError("text-embedding-ada-002 does not support dimensions")

    kwargs = {
        "model": model,
        "input": text,
        "encoding_format": encoding_format,
    }
    if dimensions:
        kwargs["dimensions"] = dimensions

    response = client.embeddings.create(**kwargs)

    responses = []
    for embedding in response.data:
        responses.append(embedding.embedding)

    if len(responses) == 1:
        return responses[0]

    return responses
