from pathlib import Path
from typing import Any

import pytest

from not_again_ai.llm.chat_completion_vision import chat_completion_vision
from not_again_ai.llm.openai_client import openai_client
from not_again_ai.llm.prompts import encode_image

cat_image = Path(__file__).parent / "sample_images" / "cat.jpg"
dog_image = Path(__file__).parent / "sample_images" / "dog.jpg"
numbers_image = Path(__file__).parent / "sample_images" / "numbers.png"
sk_infographic = Path(__file__).parent / "sample_images" / "SKInfographic.png"
sk_diagram = Path(__file__).parent / "sample_images" / "SKDiagram.png"


def test_chat_completion_vision() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the animal in the image in one word."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(cat_image)}", "detail": "low"},
                },
            ],
        },
    ]

    response = chat_completion_vision(
        messages=messages, model="gpt-4-vision-preview", max_tokens=200, temperature=0.5, client=client
    )
    print(response)


def test_chat_completion_vision_length() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the animal in the image in one sentence."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(cat_image)}", "detail": "low"},
                },
            ],
        },
    ]
    response = chat_completion_vision(messages=messages, model="gpt-4-1106-vision-preview", max_tokens=2, client=client)
    print(response)


def test_chat_completion_vision_n() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the animal in the image in one word."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(cat_image)}", "detail": "low"},
                },
            ],
        },
    ]
    response = chat_completion_vision(
        messages=messages, model="gpt-4-1106-vision-preview", max_tokens=200, client=client, n=2
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_seed() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Pick one random number that is written in the image. Just write the number."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(numbers_image)}", "detail": "low"},
                },
            ],
        },
    ]
    response_1 = chat_completion_vision(
        messages=messages, model="gpt-4-1106-vision-preview", max_tokens=200, client=client, temperature=2, seed=42
    )

    response_2 = chat_completion_vision(
        messages=messages, model="gpt-4-1106-vision-preview", max_tokens=200, client=client, temperature=2, seed=42
    )

    print(response_1)
    print(response_1["message"])
    print(response_2["message"])

    # There currently seems to a bug where the system fingerprint is None even when seed is set
    # assert "system_fingerprint" in response_1


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_multiple_images() -> None:
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are the animals in the images? Reply in one word for each animal."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(cat_image)}", "detail": "low"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(dog_image)}", "detail": "low"},
                },
            ],
        },
    ]

    response = chat_completion_vision(
        messages=messages, model="gpt-4-1106-vision-preview", max_tokens=200, client=client
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_multiple_messages() -> None:
    """Test with two image messages separated by an assistant message."""
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the animal in the image in one word."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(cat_image)}", "detail": "low"},
                },
            ],
        },
        {"role": "assistant", "content": "Cat"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What about this animal?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(dog_image)}", "detail": "low"},
                },
            ],
        },
    ]
    response = chat_completion_vision(
        messages=messages, model="gpt-4-1106-vision-preview", max_tokens=200, client=client
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_different_fidelity() -> None:
    """Test sending one image with low fidelity and another with high fidelity."""
    client = openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Based on these infographics, can you summarize how Semantic Kernel works in exactly one sentence?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(sk_infographic)}", "detail": "high"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(sk_diagram)}", "detail": "low"},
                },
            ],
        },
    ]
    response = chat_completion_vision(
        messages=messages, model="gpt-4-1106-vision-preview", max_tokens=200, client=client
    )
    print(response)
