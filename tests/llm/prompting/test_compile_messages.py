from pathlib import Path

from pydantic import BaseModel

from not_again_ai.llm.chat_completion.types import (
    AssistantMessage,
    ImageContent,
    ImageDetail,
    ImageUrl,
    MessageT,
    SystemMessage,
    TextContent,
    UserMessage,
)
from not_again_ai.llm.prompting.compile_messages import compile_messages, create_image_url, pydantic_to_json_schema

image_dir = Path(__file__).parent.parent / "sample_images"
sk_infographic = image_dir / "SKInfographic.png"
sk_diagram = image_dir / "SKDiagram.png"
cat_image = image_dir / "cat.jpg"
dog_image = image_dir / "dog.jpg"


def test_chat_prompt_no_vars() -> None:
    no_vars_prompt: list[MessageT] = [
        SystemMessage(
            content="This is a system message with no variables",
        ),
        UserMessage(
            content="This is a user message with no variables",
        ),
    ]

    messages_formatted = compile_messages(no_vars_prompt, {})
    messages_expected = [
        SystemMessage(
            content="This is a system message with no variables",
        ),
        UserMessage(
            content="This is a user message with no variables",
        ),
    ]
    assert messages_formatted == messages_expected


def test_compile_messages_place_extraction() -> None:
    place_extraction_prompt: list[MessageT] = [
        SystemMessage(
            content="""- You are a helpful assistant trying to extract places that occur in a given text.
- You must identify all the places in the text and return them in a list like this: ["place1", "place2", "place3"].""",
        ),
        UserMessage(
            content="""Here is the text I want you to extract places from:
{%- # The user's input text goes below %}
{{text}}""",
        ),
    ]
    variables = {
        "text": "I went to Paris and Berlin.",
    }
    messages_formatted = compile_messages(place_extraction_prompt, variables)
    messages_expected = [
        SystemMessage(
            content="""- You are a helpful assistant trying to extract places that occur in a given text.
- You must identify all the places in the text and return them in a list like this: ["place1", "place2", "place3"].""",
        ),
        UserMessage(
            content="""Here is the text I want you to extract places from:
I went to Paris and Berlin.""",
        ),
    ]
    assert messages_formatted == messages_expected


def test_compile_messages_vision() -> None:
    vision_prompt: list[MessageT] = [
        SystemMessage(
            content="You are a helpful assistant.",
        ),
        UserMessage(
            content=[
                TextContent(text="Describe the animal in the image in one word."),
                ImageContent(image_url=ImageUrl(url=create_image_url(cat_image), detail=ImageDetail.LOW)),
            ],
        ),
        AssistantMessage(
            content="{{ answer }}",
        ),
        UserMessage(
            content=[
                TextContent(text="What about this animal?"),
                ImageContent(image_url=ImageUrl(url=create_image_url(dog_image), detail=ImageDetail.LOW)),
            ],
        ),
    ]

    messages_formatted = compile_messages(vision_prompt, {"answer": "Cat"})
    messages_expected = [
        SystemMessage(
            content="You are a helpful assistant.",
        ),
        UserMessage(
            content=[
                TextContent(text="Describe the animal in the image in one word."),
                ImageContent(image_url=ImageUrl(url=create_image_url(cat_image), detail=ImageDetail.LOW)),
            ],
        ),
        AssistantMessage(
            content="Cat",
        ),
        UserMessage(
            content=[
                TextContent(text="What about this animal?"),
                ImageContent(image_url=ImageUrl(url=create_image_url(dog_image), detail=ImageDetail.LOW)),
            ],
        ),
    ]

    assert messages_formatted == messages_expected


def test_pydantic_to_json_schema() -> None:
    class Step(BaseModel):
        explanation: str
        output: str

    class MathResponse(BaseModel):
        steps: list[Step]
        final_answer: str

    json_schema = pydantic_to_json_schema(MathResponse, schema_name="math_response", description="A math response")

    expected_schema = {
        "$defs": {
            "Step": {
                "properties": {
                    "explanation": {"title": "Explanation", "type": "string"},
                    "output": {"title": "Output", "type": "string"},
                },
                "required": ["explanation", "output"],
                "title": "Step",
                "type": "object",
                "additionalProperties": False,
            }
        },
        "properties": {
            "steps": {"items": {"$ref": "#/$defs/Step"}, "title": "Steps", "type": "array"},
            "final_answer": {"title": "Final Answer", "type": "string"},
        },
        "required": ["steps", "final_answer"],
        "title": "MathResponse",
        "type": "object",
        "additionalProperties": False,
    }

    assert json_schema["strict"] is True
    assert json_schema["name"] == "math_response"
    assert json_schema["description"] == "A math response"
    assert json_schema["schema"] == expected_schema
