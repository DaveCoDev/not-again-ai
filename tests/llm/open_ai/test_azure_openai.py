from pathlib import Path
from typing import Any

from openai import AzureOpenAI

from not_again_ai.llm.openai_api.chat_completion import chat_completion
from not_again_ai.llm.openai_api.openai_client import openai_client
from not_again_ai.llm.openai_api.prompts import encode_image

image_dir = Path(__file__).parent.parent / "sample_images"
cat_image = image_dir / "cat.jpg"
dog_image = image_dir / "dog.jpg"
numbers_image = image_dir / "numbers.png"
sk_infographic = image_dir / "SKInfographic.png"
sk_diagram = image_dir / "SKDiagram.png"


def test_azure_openai() -> None:
    client = openai_client(api_type="azure_openai")
    assert isinstance(client, AzureOpenAI)


def test_aoai_chat_completion() -> None:
    client = openai_client(api_type="azure_openai")
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    response = chat_completion(messages=messages, model="gpt-4o-mini-2024-07-18", max_tokens=15, client=client)
    print(response)


def test_chat_completion_vision_different_fidelity() -> None:
    """Test sending one image with low fidelity and another with high fidelity."""
    client = openai_client(api_type="azure_openai")
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
    response = chat_completion(messages=messages, model="gpt-4o-2024-05-13", max_tokens=200, client=client)
    print(response)


def test_chat_completion_misc_1() -> None:
    client = openai_client(api_type="azure_openai")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        },
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the current weather like in Boston, MA today? Call the get_current_weather function.",
        }
    ]
    client = openai_client(api_type="azure_openai")
    response = chat_completion(
        messages=messages,
        model="gpt-4o-mini-2024-07-18",
        client=client,
        tools=tools,
        max_tokens=300,
        temperature=0,
        logprobs=(True, 2),
        seed=42,
        n=2,
    )
    # NOTE: When a function is called, logprobs are not returned.
    print(response)


if __name__ == "__main__":
    test_chat_completion_misc_1()
    test_chat_completion_vision_different_fidelity()
    test_aoai_chat_completion()
    test_azure_openai()
