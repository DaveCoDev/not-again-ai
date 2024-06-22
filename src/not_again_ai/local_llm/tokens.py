import tiktoken
from transformers import AutoTokenizer

from not_again_ai.llm.openai_api.tokens import load_tokenizer as openai_load_tokenizer
from not_again_ai.llm.openai_api.tokens import num_tokens_from_messages as openai_num_tokens_from_messages
from not_again_ai.llm.openai_api.tokens import num_tokens_in_string as openai_num_tokens_in_string
from not_again_ai.llm.openai_api.tokens import truncate_str as openai_truncate_str
from not_again_ai.local_llm.ollama.tokens import load_tokenizer as ollama_load_tokenizer
from not_again_ai.local_llm.ollama.tokens import num_tokens_from_messages as ollama_num_tokens_from_messages
from not_again_ai.local_llm.ollama.tokens import num_tokens_in_string as ollama_num_tokens_in_string
from not_again_ai.local_llm.ollama.tokens import truncate_str as ollama_truncate_str


def load_tokenizer(model: str, provider: str) -> AutoTokenizer | tiktoken.Encoding:
    """Load the tokenizer for the given model and providers

    Args:
        model (str): The name of the language model to load the tokenizer for
        provider (str): Either "openai_api" or "ollama"

    Returns:
        Either a HuggingFace tokenizer or a tiktoken encoding object
    """
    if provider == "openai_api":
        return openai_load_tokenizer(model)
    elif provider == "ollama":
        return ollama_load_tokenizer(model)
    else:
        raise ValueError(f"Unknown tokenizer provider {provider}")


def truncate_str(text: str, max_len: int, tokenizer: AutoTokenizer | tiktoken.Encoding, provider: str) -> str:
    """Truncates a string to a maximum token length.

    Args:
        text: The string to truncate.
        max_len: The maximum number of tokens to keep.
        tokenizer: Either a HuggingFace tokenizer or a tiktoken encoding object
        provider (str): Either "openai_api" or "ollama"

    Returns:
        str: The truncated string.
    """
    if provider == "openai_api":
        return openai_truncate_str(text, max_len, tokenizer)
    elif provider == "ollama":
        return ollama_truncate_str(text, max_len, tokenizer)
    else:
        raise ValueError(f'Unknown tokenizer provider "{provider}"')


def num_tokens_in_string(text: str, tokenizer: AutoTokenizer | tiktoken.Encoding, provider: str) -> int:
    """Return the number of tokens in a string.

    Args:
        text: The string to count the tokens.
        tokenizer: Either a HuggingFace tokenizer or a tiktoken encoding object
        provider (str): Either "openai_api" or "ollama"

    Returns:
        int: The number of tokens in the string.
    """
    if provider == "openai_api":
        return openai_num_tokens_in_string(text, tokenizer)
    elif provider == "ollama":
        return ollama_num_tokens_in_string(text, tokenizer)
    else:
        raise ValueError(f'Unknown tokenizer provider "{provider}"')


def num_tokens_from_messages(
    messages: list[dict[str, str]], tokenizer: AutoTokenizer | tiktoken.Encoding, provider: str
) -> int:
    """Return the number of tokens used by a list of messages.

    Args:
        messages: A list of messages to count the tokens
            should ideally be the result after calling llm.prompts.chat_prompt.
        tokenizer: Either a HuggingFace tokenizer or a tiktoken encoding object
        provider (str): Either "openai_api" or "ollama"

    Returns:
        int: The number of tokens used by the messages.
    """
    if provider == "openai_api":
        return openai_num_tokens_from_messages(messages, tokenizer)
    elif provider == "ollama":
        return ollama_num_tokens_from_messages(messages, tokenizer)
    else:
        raise ValueError(f'Unknown tokenizer provider "{provider}"')
