from liquid import Template


def _validate_message(message: dict[str, str]) -> bool:
    valid_fields = ["role", "content", "name", "function_call"]
    # Check if the only keys in the message are in valid_fields
    if not all(key in valid_fields for key in message):
        return False

    # Check if the each of the values are strings except function_call
    if not all(isinstance(value, str) for key, value in message.items() if key != "function_call"):
        return False

    valid_roles = ["system", "user", "assistant", "function"]
    if message["role"] not in valid_roles:
        return False

    return True


def chat_prompt(messages_unformatted: list[dict[str, str]], variables: dict[str, str]) -> list[dict[str, str]]:
    """
    Formats a list of messages for OpenAI's chat completion API using Liquid templating.

    Args:
        messages_unformatted: A list of dictionaries where each dictionary
            represents a message. Each message must have 'role' and 'content'
            keys with string values, where content is a Liquid template.
        variables: A dictionary where each key-value pair represents a variable
            name and its value for template rendering.

    Returns:
        A list of dictionaries with the same structure as `messages_unformatted`,
        but with the 'content' of each message with the provided `variables`.

    Examples:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Help me {{task}}"}
        ... ]
        >>> vars = {"task": "write Python code for the fibonnaci sequence"}
        >>> chat_prompt(messages, vars)
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Help me write Python code for the fibonnaci sequence"}
        ]
    """

    messages_formatted = messages_unformatted.copy()
    for message in messages_formatted:
        if not _validate_message(message):
            raise ValueError(f"Invalid message: {message}")

        liquid_template = Template(message["content"])
        message["content"] = liquid_template.render(**variables)

    return messages_formatted
