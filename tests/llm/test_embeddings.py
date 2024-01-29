from not_again_ai.llm.embeddings import embed_text
from not_again_ai.llm.openai_client import openai_client


def test_embeddings_basic() -> None:
    client = openai_client()
    text = "Hello, world!"
    response = embed_text(text=text, client=client, model="text-embedding-3-large")
    assert isinstance(response, list) and all(isinstance(elem, float) for elem in response)
    print(response)


def test_embeddings_base64() -> None:
    client = openai_client()
    text = "Hello, world!"
    response = embed_text(text=text, client=client, model="text-embedding-3-small", encoding_format="base64")
    assert isinstance(response, str)
    print(response)


def test_embeddings_dimensions() -> None:
    client = openai_client()
    text = "Hello, world!"
    response = embed_text(text=text, client=client, model="text-embedding-3-small", dimensions=3)
    assert isinstance(response, list) and all(isinstance(elem, float) for elem in response)
    print(response)


def test_embeddings_ada002() -> None:
    client = openai_client()
    text = "Hello, world!"
    response = embed_text(text=text, client=client, model="text-embedding-ada-002")
    assert isinstance(response, list) and all(isinstance(elem, float) for elem in response)
    print(response)


def test_multiple_embeddings() -> None:
    client = openai_client()
    text = ["Hello, world!", "Hello, world!"]
    response = embed_text(text=text, client=client, model="text-embedding-3-small", dimensions=3)
    assert isinstance(response, list) and all(
        isinstance(inner_list, list) and all(isinstance(elem, float) for elem in inner_list) for inner_list in response
    )
    print(response)
