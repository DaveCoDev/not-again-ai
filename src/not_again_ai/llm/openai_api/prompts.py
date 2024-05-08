import base64
from copy import deepcopy
import mimetypes
from pathlib import Path
from typing import Any

from liquid import Template


def _validate_message_vision(message: dict[str, list[dict[str, Path | str]] | str]) -> bool:
    """Validates that a message for a vision model is valid"""
    valid_fields = ["role", "content", "name", "tool_call_id", "tool_calls"]
    if not all(key in valid_fields for key in message):
        raise ValueError(f"Message contains invalid fields: {message.keys()}")

    valid_roles = ["system", "user", "assistant", "tool"]
    if message["role"] not in valid_roles:
        raise ValueError(f"Message contains invalid role: {message['role']}")

    if not isinstance(message["content"], list) and not isinstance(message["content"], str):
        raise ValueError(f"content must be a list of dictionaries or a string: {message['content']}")

    if isinstance(message["content"], list):
        for content_part in message["content"]:
            if isinstance(content_part, dict):
                if "image" not in content_part:
                    raise ValueError(f"Dictionary content part must contain 'image' key: {content_part}")
                if "detail" in content_part and content_part["detail"] not in ["low", "high"]:
                    raise ValueError(f"Optional 'detail' key must be 'low' or 'high': {content_part['detail']}")
            elif not isinstance(content_part, str):
                raise ValueError(f"content_part must be a dictionary or a string: {content_part}")

    return True


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


def chat_prompt(messages_unformatted: list[dict[str, Any]], variables: dict[str, str]) -> list[dict[str, Any]]:
    """Formats a list of messages for OpenAI's chat completion API,
    including special syntax for vision models, using Liquid templating.

    Args:
        messages_unformatted (list[dict[str, list[dict[str, Path | str]] | str]]):
            A list of dictionaries where each dictionary represents a message.
            `content` can be a Liquid template string or a list of dictionaries where each dictionary
            represents a content part. Each content part can be a string or a dictionary with 'image' and 'detail' keys.
            The 'image' key must be a Path or a string representing a URL. The 'detail' key is optional and must be 'low' or 'high'.
        variables: A dictionary where each key-value pair represents a variable
            name and its value for template rendering.

    Returns:
        A list which represents messages in the format that OpenAI expects for its chat completions API.
        See here for details: https://platform.openai.com/docs/api-reference/chat/create

    Examples:
        >>> # Assume cat_image and dog_image are Path objects to image files
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {
        ...         "role": "user",
        ...          "content": ["Describe the animal in the image in one word.", {"image": cat_image, "detail": "low"}],
        ...     }
        ...     {"role": "assistant", "content": "{{ answer }}"},
        ...     {
        ...         "role": "user",
        ...         "content": ["What about this animal?", {"image": dog_image, "detail": "high"}],
        ...     }
        ... ]
        >>> vars = {"answer": "Cat"}
        >>> chat_prompt(messages, vars)
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the animal in the image in one word."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,<encoding>", "detail": "low"},
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
                        "image_url": {"url": f"data:image/jpeg;base64,<encoding>", "detail": "high"},
                    },
                ],
            },
        ]
    """
    messages_formatted = deepcopy(messages_unformatted)
    for message in messages_formatted:
        if not _validate_message_vision(message):
            raise ValueError()

        if isinstance(message["content"], list):
            for i in range(len(message["content"])):
                content_part = message["content"][i]
                if isinstance(content_part, dict):
                    image_path = content_part["image"]
                    if isinstance(image_path, Path):
                        temp_content_part: dict[str, Any] = {
                            "type": "image_url",
                            "image_url": {
                                "url": create_image_url(image_path),
                            },
                        }
                        if "detail" in content_part:
                            temp_content_part["image_url"]["detail"] = content_part["detail"]
                    elif isinstance(image_path, str):
                        # Assume its a valid URL
                        pass
                    else:
                        raise ValueError(f"Image path must be a Path or str: {image_path}")
                    message["content"][i] = temp_content_part
                elif isinstance(content_part, str):
                    message["content"][i] = {
                        "type": "text",
                        "text": Template(content_part).render(**variables),
                    }
        elif isinstance(message["content"], str):
            message["content"] = Template(message["content"]).render(**variables)

    return messages_formatted
