from pathlib import Path

from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletionsFunctionToolDefinition,
    FunctionDefinition,
    ImageContentItem,
    ImageDetailLevel,
    ImageUrl,
    SystemMessage,
    TextContentItem,
    UserMessage,
)
import pytest

from not_again_ai.llm.gh_models.azure_ai_client import azure_ai_client
from not_again_ai.llm.gh_models.chat_completion import chat_completion

MODELS = ["meta-llama-3.1-8b-instruct", "gpt-4o-mini", "gpt-4o"]

image_dir = Path(__file__).parent.parent / "sample_images"
cat_image = image_dir / "cat.jpg"
dog_image = image_dir / "dog.jpg"
numbers_image = image_dir / "numbers.png"
sk_infographic = image_dir / "SKInfographic.png"
sk_diagram = image_dir / "SKDiagram.png"


@pytest.fixture(params=MODELS)
def model(request):  # type: ignore
    return request.param


@pytest.mark.skip("API Cost")
def test_chat_completion(model: str) -> None:
    client = azure_ai_client()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the capital of France?"),
    ]

    response = chat_completion(messages=messages, model=model, max_tokens=300, client=client)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_length(model: str) -> None:
    client = azure_ai_client()
    messages = [SystemMessage(content="You are a helpful assistant."), UserMessage(content="Hello!")]
    response = chat_completion(messages=messages, model=model, max_tokens=2, client=client)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_expected_function(model: str) -> None:
    tools = [
        ChatCompletionsFunctionToolDefinition(
            function=FunctionDefinition(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
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
            )
        )
    ]
    messages = [
        UserMessage(
            content="What's the current weather like in Boston, MA today? Call the get_current_weather function."
        )
    ]
    client = azure_ai_client()
    response = chat_completion(
        messages=messages,  # type: ignore
        model=model,
        client=client,
        tools=tools,  # type: ignore
        max_tokens=300,
        temperature=0,
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_multiple_functions(model: str) -> None:
    tools = [
        ChatCompletionsFunctionToolDefinition(
            function=FunctionDefinition(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
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
            )
        )
    ]
    messages = [
        SystemMessage(content="Call the get_current_weather function once for each city that the user mentions."),
        UserMessage(content="What's the current weather like in Boston, MA and New York, NY today?"),
    ]
    client = azure_ai_client()
    response = chat_completion(
        messages=messages,
        model=model,
        client=client,
        tools=tools,  # type: ignore
        max_tokens=400,
        temperature=0,
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_json_mode(model: str) -> None:
    messages = [
        SystemMessage(
            content="""You are getting names of users and formatting them into json.
Example:
User: Jane Doe
Output: {"name": "Jane Doe"}"""
        ),
        UserMessage(content="John Doe"),
    ]
    client = azure_ai_client()
    response = chat_completion(
        messages=messages,
        model=model,
        client=client,
        max_tokens=300,
        temperature=0,
        json_mode=True,
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_seed(model: str) -> None:
    client = azure_ai_client()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Generate a random number between 0 and 100."),
    ]
    response_1 = chat_completion(messages=messages, model=model, max_tokens=100, client=client, temperature=1, seed=42)
    response_2 = chat_completion(messages=messages, model=model, max_tokens=100, client=client, temperature=1, seed=42)

    print(response_1["message"])
    print(response_2["message"])


@pytest.mark.skip("API Cost")
def test_chat_completion_misc_1(model: str) -> None:
    client = azure_ai_client()
    tools = [
        ChatCompletionsFunctionToolDefinition(
            function=FunctionDefinition(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
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
            )
        )
    ]
    messages = [
        UserMessage(
            content="What's the current weather like in Boston, MA today? Call the get_current_weather function."
        )
    ]
    client = azure_ai_client()
    response = chat_completion(
        messages=messages,  # type: ignore
        model=model,
        client=client,
        tools=tools,  # type: ignore
        max_tokens=300,
        temperature=0,
        seed=42,
    )
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision(model: str) -> None:
    client = azure_ai_client()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextContentItem(text="Describe the animal in the image in one word."),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file=cat_image.absolute().as_posix(),
                        image_format="jpeg",
                        detail=ImageDetailLevel.LOW,
                    )
                ),
            ]
        ),
    ]

    response = chat_completion(messages=messages, model=model, max_tokens=200, temperature=0.5, client=client)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_multiple_images(model: str) -> None:
    client = azure_ai_client()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextContentItem(text="What are the animals in the images? Reply in one word for each animal."),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file=cat_image.absolute().as_posix(),
                        image_format="jpeg",
                        detail=ImageDetailLevel.LOW,
                    )
                ),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file=dog_image.absolute().as_posix(),
                        image_format="jpeg",
                        detail=ImageDetailLevel.LOW,
                    )
                ),
            ]
        ),
    ]
    response = chat_completion(messages=messages, model=model, max_tokens=200, client=client)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_multiple_messages(model: str) -> None:
    """Test with two image messages separated by an assistant message."""
    client = azure_ai_client()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextContentItem(text="Describe the animal in the image in one word."),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file=cat_image.absolute().as_posix(),
                        image_format="jpeg",
                        detail=ImageDetailLevel.LOW,
                    )
                ),
            ]
        ),
        AssistantMessage(content="Cat"),
        UserMessage(
            content=[
                TextContentItem(text="What about this animal?"),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file=dog_image.absolute().as_posix(),
                        image_format="jpeg",
                        detail=ImageDetailLevel.LOW,
                    )
                ),
            ]
        ),
    ]
    response = chat_completion(messages=messages, model=model, max_tokens=200, client=client)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_different_fidelity(model: str) -> None:
    """Test sending one image with low fidelity and another with high fidelity."""
    client = azure_ai_client()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextContentItem(
                    text="Based on these infographics, can you summarize how Semantic Kernel works in exactly one sentence?"
                ),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file=sk_infographic.absolute().as_posix(),
                        image_format="png",
                        detail=ImageDetailLevel.HIGH,
                    )
                ),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file=sk_diagram.absolute().as_posix(),
                        image_format="png",
                        detail=ImageDetailLevel.LOW,
                    )
                ),
            ]
        ),
    ]
    response = chat_completion(messages=messages, model=model, max_tokens=200, client=client)
    print(response)


@pytest.mark.skip("API Cost")
def test_chat_completion_vision_tool_call(model: str) -> None:
    client = azure_ai_client()
    messages = [
        SystemMessage(
            content="""You are detecting if there is text (numbers or letters) in images.
If you see any text, call the ocr tool. It takes no parameters."""
        ),
        UserMessage(
            content=[
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file=numbers_image.absolute().as_posix(),
                        image_format="png",
                        detail=ImageDetailLevel.LOW,
                    )
                )
            ]
        ),
    ]
    tools = [
        ChatCompletionsFunctionToolDefinition(
            function=FunctionDefinition(
                name="ocr",
                description="Perform Optical Character Recognition (OCR) on an image",
                parameters={},
            )
        )
    ]
    response = chat_completion(
        messages=messages,
        model=model,
        client=client,
        tools=tools,  # type: ignore
        max_tokens=200,
    )
    print(response)


if __name__ == "__main__":
    test_chat_completion_vision_tool_call(model=MODELS[2])
    test_chat_completion_vision_different_fidelity(model=MODELS[2])
    test_chat_completion_vision_multiple_messages(model=MODELS[2])
    test_chat_completion_vision_multiple_images(model=MODELS[2])
    test_chat_completion_vision(model=MODELS[2])
    test_chat_completion_misc_1(model=MODELS[1])
    test_chat_completion_seed(model=MODELS[0])
    test_json_mode(model=MODELS[0])
    test_chat_completion_multiple_functions(model=MODELS[1])
    test_chat_completion_expected_function(model=MODELS[1])
    test_chat_completion_length(model=MODELS[0])
    test_chat_completion(model=MODELS[0])
