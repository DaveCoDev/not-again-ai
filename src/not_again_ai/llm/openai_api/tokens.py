import tiktoken


def load_tokenizer(model: str) -> tiktoken.Encoding:
    """Load the tokenizer for the given model

    Args:
        model (str): The name of the language model to load the tokenizer for

    Returns:
        A tiktoken encoding object
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding


def truncate_str(text: str, max_len: int, tokenizer: tiktoken.Encoding) -> str:
    """Truncates a string to a maximum token length.

    Args:
        text (str): The string to truncate.
        max_len (int): The maximum number of tokens to keep.
        tokenizer (tiktoken.Encoding): A tiktoken encoding object

    Returns:
        str: The truncated string.
    """
    tokens = tokenizer.encode(text)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        # Decode the tokens back to a string
        truncated_text = tokenizer.decode(tokens)
        return truncated_text
    else:
        return text


def num_tokens_in_string(text: str, tokenizer: tiktoken.Encoding) -> int:
    """Return the number of tokens in a string.

    Args:
        text (str): The string to count the tokens.
        tokenizer (tiktoken.Encoding): A tiktoken encoding object

    Returns:
        int: The number of tokens in the string.
    """
    return len(tokenizer.encode(text))


def num_tokens_from_messages(
    messages: list[dict[str, str]], tokenizer: tiktoken.Encoding, model: str = "gpt-3.5-turbo-0125"
) -> int:
    """Return the number of tokens used by a list of messages.
    NOTE: Does not support counting tokens used by function calling or prompts with images.
    Reference: # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    and https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    Args:
        messages (list[dict[str, str]]): A list of messages to count the tokens
            should ideally be the result after calling llm.prompts.chat_prompt.
        tokenizer (tiktoken.Encoding): A tiktoken encoding object
        model (str): The model to use for tokenization. Defaults to "gpt-3.5-turbo-0125".
            See https://platform.openai.com/docs/models for a list of OpenAI models.

    Returns:
        int: The number of tokens used by the messages.
    """
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-1106-preview",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
    }:
        tokens_per_message = 3  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = 1  # if there's a name, the role is omitted
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    # Approximate catch-all. Assumes future versions of 3.5 and 4 will have the same token counts as the 0613 versions.
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, tokenizer=tokenizer, model="gpt-3.5-turbo-0613")
    elif "gpt-4o" in model:
        return num_tokens_from_messages(messages, tokenizer=tokenizer, model="gpt-4o-2024-05-13")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, tokenizer=tokenizer, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(tokenizer.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
