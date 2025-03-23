import base64
from collections.abc import Sequence
from copy import deepcopy
import mimetypes
from pathlib import Path
from typing import Any

from liquid import render
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

from not_again_ai.llm.chat_completion.types import MessageT


def _apply_templates(value: Any, variables: dict[str, str]) -> Any:
    """Recursively applies Liquid templating to all string fields within the given value."""
    if isinstance(value, str):
        return render(value, **variables)
    elif isinstance(value, list):
        return [_apply_templates(item, variables) for item in value]
    elif isinstance(value, dict):
        return {key: _apply_templates(val, variables) for key, val in value.items()}
    elif isinstance(value, BaseModel):
        # Process each field in the BaseModel by converting it to a dict,
        # applying templating to its values, and then re-instantiating the model.
        processed_data = {key: _apply_templates(val, variables) for key, val in value.model_dump().items()}
        return value.__class__(**processed_data)
    else:
        return value


def compile_messages(messages: Sequence[MessageT], variables: dict[str, str]) -> Sequence[MessageT]:
    """Compiles messages using Liquid templating and the provided variables.
    Calls render(content_part, **variables) on each text content part.

    Args:
        messages: List of MessageT where content can contain Liquid templates.
        variables: The variables to inject into the templates.

    Returns:
        The same list of messages with the content parts injected with the variables.
    """
    messages_formatted = deepcopy(messages)
    messages_formatted = [_apply_templates(message, variables) for message in messages_formatted]
    return messages_formatted


def compile_tools(tools: Sequence[dict[str, Any]], variables: dict[str, str]) -> Sequence[dict[str, Any]]:
    """Compiles a list of tool argument dictionaries using Liquid templating and provided variables.

    Each dictionary in the list is deep copied and processed recursively to substitute any Liquid
    templates present in its data structure.

    Args:
        tools: A list of dictionaries representing tool arguments, where values can include Liquid templates.
        variables: A dictionary of variables to substitute into the Liquid templates.

    Returns:
        A new list of dictionaries with the Liquid templates replaced by their corresponding variable values.
    """
    tools_formatted = deepcopy(tools)
    tools_formatted = [_apply_templates(tool, variables) for tool in tools_formatted]
    return tools_formatted


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
