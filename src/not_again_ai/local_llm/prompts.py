from copy import deepcopy

from liquid import Template


def chat_prompt(messages_unformatted: list[dict[str, str]], variables: dict[str, str]) -> list[dict[str, str]]:
    """Formats a list of messages for chat completion models using Liquid templating.

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

    messages_formatted = deepcopy(messages_unformatted)
    for message in messages_formatted:
        liquid_template = Template(message["content"])
        message["content"] = liquid_template.render(**variables)

    return messages_formatted
