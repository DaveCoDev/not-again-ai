import base64
from copy import deepcopy
import mimetypes
from pathlib import Path
from typing import Any

from liquid import Template
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

from not_again_ai.llm.chat_completion.types import MessageT, TextContent


def compile_messages(messages: list[MessageT], variables: dict[str, str]) -> list[MessageT]:
    """Compiles messages using Liquid templating and the provided variables.
    Calls Template(content_part).render(**variables) on each text content part.

    Args:
        messages: List of MessageT where content can contain Liquid templates.
        variables: The variables to inject into the templates.

    Returns:
        The same list of messages with the content parts injected with the variables.
    """
    messages_formatted = deepcopy(messages)
    for message in messages_formatted:
        if isinstance(message.content, str):
            # For simple string content, apply template directly
            message.content = Template(message.content).render(**variables)
        elif isinstance(message.content, list):
            # For UserMessage with content parts
            for content_part in message.content:
                if isinstance(content_part, TextContent):
                    content_part.text = Template(content_part.text).render(**variables)
                # ImageContent parts are left unchanged
    return messages_formatted


def encode_image(image_path: Path) -> str:
    """Encodes an image file at the given Path to base64.

    Args:
        image_path: The path to the image file to encode.

    Returns:
        The base64 encoded image as a string.
    """
    with Path.open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_image_url(image_path: Path) -> str:
    """Creates a data URL for an image file at the given Path.

    Args:
        image_path: The path to the image file to encode.

    Returns:
        The data URL for the image.
    """
    image_data = encode_image(image_path)

    valid_mime_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]

    # Get the MIME type from the image file extension
    mime_type = mimetypes.guess_type(image_path)[0]

    # Check if the MIME type is valid
    # List of valid types is here: https://platform.openai.com/docs/guides/vision/what-type-of-files-can-i-upload
    if mime_type not in valid_mime_types:
        raise ValueError(f"Invalid MIME type for image: {mime_type}")

    return f"data:{mime_type};base64,{image_data}"


def pydantic_to_json_schema(
    pydantic_model: type[BaseModel], schema_name: str, description: str | None = None
) -> dict[str, Any]:
    """Converts a Pydantic model to a JSON schema expected by Structured Outputs.
    Must adhere to the supported schemas: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas

    Args:
        pydantic_model: The Pydantic model to convert.
        schema_name: The name of the schema.
        description: An optional description of the schema.

    Returns:
        A JSON schema dictionary representing the Pydantic model.
    """
    converted_pydantic = to_strict_json_schema(pydantic_model)
    schema = {
        "name": schema_name,
        "strict": True,
        "schema": converted_pydantic,
    }
    if description:
        schema["description"] = description
    return schema
