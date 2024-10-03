from pathlib import Path
from typing import Any

from pydantic import BaseModel

from not_again_ai.llm.openai_api.prompts import chat_prompt, encode_image, pydantic_to_json_schema

image_dir = Path(__file__).parent.parent / "sample_images"
sk_infographic = image_dir / "SKInfographic.png"
sk_diagram = image_dir / "SKDiagram.png"
cat_image = image_dir / "cat.jpg"
dog_image = image_dir / "dog.jpg"


def test_chat_prompt_no_vars() -> None:
    no_vars_prompt = [
        {
            "role": "system",
            "content": """This is a system message with no variables""",
        },
        {
            "role": "user",
            "content": """This is a user message with no variables""",
        },
    ]

    messages_formatted = chat_prompt(no_vars_prompt, {})
    messages_expected = [
        {
            "role": "system",
            "content": """This is a system message with no variables""",
        },
        {
            "role": "user",
            "content": """This is a user message with no variables""",
        },
    ]
    assert messages_formatted == messages_expected


def test_chat_prompt_place_extraction() -> None:
    place_extraction_prompt = [
        {
            "role": "system",
            "content": """- You are a helpful assistant trying to extract places that occur in a given text.
- You must identify all the places in the text and return them in a list like this: ["place1", "place2", "place3"].""",
        },
        {
            "role": "user",
            "content": """Here is the text I want you to extract places from:
{%- # The user's input text goes below %}
{{text}}""",
        },
    ]
    variables = {
        "text": "I went to Paris and Berlin.",
    }
    messages_formatted = chat_prompt(place_extraction_prompt, variables)
    messages_expected = [
        {
            "role": "system",
            "content": """- You are a helpful assistant trying to extract places that occur in a given text.
- You must identify all the places in the text and return them in a list like this: ["place1", "place2", "place3"].""",
        },
        {
            "role": "user",
            "content": """Here is the text I want you to extract places from:
I went to Paris and Berlin.""",
        },
    ]
    assert messages_formatted == messages_expected


def test_vision_prompt_1() -> None:
    vision_prompt_1: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful {{ persona }}."},
        {
            "role": "user",
            "content": [
                "Based on these infographics, can you summarize how {{ library }} works in exactly one sentence?",
                {"image": sk_infographic, "detail": "high"},
                {"image": sk_diagram, "detail": "low"},
            ],
        },
    ]
    messages_formatted = chat_prompt(vision_prompt_1, {"persona": "assistant", "library": "Semantic Kernel"})
    messages_expected = [
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

    assert messages_formatted == messages_expected


def test_vision_prompt_2() -> None:
    vision_prompt_2: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": ["Describe the animal in the image in one word.", {"image": cat_image, "detail": "low"}],
        },
        {"role": "assistant", "content": "{{ answer }}"},
        {
            "role": "user",
            "content": ["What about this animal?", {"image": dog_image, "detail": "low"}],
        },
    ]
    messages_formatted = chat_prompt(vision_prompt_2, {"answer": "Cat"})
    messages_expected = [
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
