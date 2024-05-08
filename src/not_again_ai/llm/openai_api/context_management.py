import copy

from not_again_ai.llm.openai_api.tokens import num_tokens_from_messages, truncate_str


def _inject_variable(
    messages_unformatted: list[dict[str, str]], variable_name: str, variable_text: str
) -> list[dict[str, str]]:
    """Injects variables into the messages using Python string formatting."""
    messages_formatted = copy.deepcopy(messages_unformatted)
    for message in messages_formatted:
        message["content"] = message["content"].replace("{{" + variable_name + "}}", variable_text)
    return messages_formatted


def priority_truncation(
    messages_unformatted: list[dict[str, str]],
    variables: dict[str, str],
    priority: list[str],
    token_limit: int,
    model: str = "gpt-3.5-turbo-0125",
) -> list[dict[str, str]]:
    """Formats messages_unformatted and injects variables into the messages in the order of priority, truncating the messages to fit the token limit.

    Algorithm:
        0. Checks if all variables in the priority list are in the variables dict. If not, adds the missing variables into priority in any order.
        1. Iterating over priority:
            a. Count the current number of tokens in messages_formatted and compute how many tokens remain.
            b. Count the number of times the variable occurs in messages_formatted.
            c. Truncate the variable to fit the remaining token budget taking into account the number of times it occurs in the messages.
            d. Inject the variable text into messages_formatted.

    Args:
        messages_unformatted: A list of dictionaries where each dictionary
            represents a message. Each message must have 'role' and 'content'
            keys with string values, where content is a string with any number of occurances of {{variable_name}}.
        variables: A dictionary where each key-value pair represents a variable name and its value to inject.
        priority: A list of variable names in their order of priority.
        token_limit: The maximum number of tokens allowed in the messages.
        model: The model to use for tokenization. Defaults to "gpt-3.5-turbo-0125".
    """

    # Check if all variables in the priority list are in the variables dict.
    # If not, add the missing variables into priority in any order.
    for var in variables:
        if var not in priority:
            priority.append(var)

    messages_formatted = copy.deepcopy(messages_unformatted)
    for var in priority:
        # Count the current number of tokens in messages_formatted and compute a remaining token budget.
        num_tokens = num_tokens_from_messages(messages_formatted, model=model)
        remaining_tokens = token_limit - num_tokens
        if remaining_tokens <= 0:
            break

        # Count the number of times the variable occurs in messages_formatted (including within the same message).
        num_var_occurrences = 0
        for message in messages_formatted:
            num_var_occurrences += message["content"].count("{{" + var + "}}")

        # Truncate the variable to fit the remaining token budget taking into account the number of times it occurs in the messages.
        truncated_var = truncate_str(variables[var], remaining_tokens // num_var_occurrences, model=model)

        # Inject the variable text into messages_formatted.
        messages_formatted = _inject_variable(messages_formatted, var, truncated_var)

    return messages_formatted
