"""By default use the associated huggingface transformer tokenizer.
If it does not exist in the mapping, default to tiktoken with some buffer (const + percentage)"""

from loguru import logger
import tiktoken
from transformers import AutoTokenizer

from not_again_ai.llm.openai_api.tokens import num_tokens_from_messages as openai_num_tokens_from_messages
from not_again_ai.local_llm.ollama.model_mapping import OLLAMA_MODEL_MAPPING

TIKTOKEN_NUM_TOKENS_BUFFER = 10
TIKTOKEN_PERCENT_TOKENS_BUFFER = 1.1


def load_tokenizer(model: str) -> AutoTokenizer | tiktoken.Encoding:
    """Use the model mapping to load the appropriate tokenizer

    Args:
        model: The name of the language model to load the tokenizer for

    Returns:
        Either a HuggingFace tokenizer or a tiktoken encoding object
    """

    # Loop over the keys in the model mapping checking if the model starts with the key
    for key in OLLAMA_MODEL_MAPPING:
        if model.startswith(key):
            return AutoTokenizer.from_pretrained(OLLAMA_MODEL_MAPPING[key], use_fast=True)

    # If the model does not start with any key in the model mapping, default to tiktoken
    logger.warning(
        f'Model "{model}" not found in OLLAMA_MODEL_MAPPING. Using tiktoken - token counts will have an added buffer of \
{TIKTOKEN_PERCENT_TOKENS_BUFFER * 100}% plus {TIKTOKEN_NUM_TOKENS_BUFFER} tokens.'
    )
    tokenizer = tiktoken.get_encoding("o200k_base")
    return tokenizer


def truncate_str(text: str, max_len: int, tokenizer: AutoTokenizer | tiktoken.Encoding) -> str:
    """Truncates a string to a maximum token length.

    Args:
        text: The string to truncate.
        max_len: The maximum number of tokens to keep.
        tokenizer: Either a HuggingFace tokenizer or a tiktoken encoding object

    Returns:
        str: The truncated string.
    """
    if isinstance(tokenizer, tiktoken.Encoding):
        tokens = tokenizer.encode(text)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            truncated_text = tokenizer.decode(tokens)
            return truncated_text
    else:
        tokens = tokenizer(text, return_tensors=None)["input_ids"]
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            truncated_text = tokenizer.decode(tokens)
            return truncated_text

    return text


def num_tokens_in_string(text: str, tokenizer: AutoTokenizer | tiktoken.Encoding) -> int:
    """Return the number of tokens in a string.

    Args:
        text: The string to count the tokens.
        tokenizer: Either a HuggingFace tokenizer or a tiktoken encoding object

    Returns:
        int: The number of tokens in the string.
    """
    if isinstance(tokenizer, tiktoken.Encoding):
        num_tokens = (len(tokenizer.encode(text)) * TIKTOKEN_PERCENT_TOKENS_BUFFER) + TIKTOKEN_NUM_TOKENS_BUFFER
        return int(num_tokens)
    else:
        tokens = tokenizer(text, return_tensors=None)["input_ids"]
        return len(tokens)


def num_tokens_from_messages(messages: list[dict[str, str]], tokenizer: AutoTokenizer | tiktoken.Encoding) -> int:
    """Return the number of tokens used by a list of messages.
    For models with HuggingFace tokenizers, uses

    Args:
        messages: A list of messages to count the tokens
            should ideally be the result after calling llm.prompts.chat_prompt.
        tokenizer: Either a HuggingFace tokenizer or a tiktoken encoding object

    Returns:
        int: The number of tokens used by the messages.
    """
    if isinstance(tokenizer, tiktoken.Encoding):
        num_tokens = (
            openai_num_tokens_from_messages(messages, tokenizer=tokenizer, model="gpt-4o")
            * TIKTOKEN_PERCENT_TOKENS_BUFFER
        ) + TIKTOKEN_NUM_TOKENS_BUFFER
        return int(num_tokens)
    else:
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors=None)
        return len(tokens)
