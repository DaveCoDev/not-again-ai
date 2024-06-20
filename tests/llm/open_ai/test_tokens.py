import pytest

from not_again_ai.llm.openai_api.tokens import (
    load_tokenizer,
    num_tokens_from_messages,
    num_tokens_in_string,
    truncate_str,
)


# Tests for truncate_str function
def test_truncate_normal() -> None:
    text = "This is a test sentence for the function."
    max_len = 3  # Assuming 'This is a' is within 3 tokens
    tokenizer = load_tokenizer("gpt-3.5-turbo-0125")
    assert truncate_str(text, max_len, tokenizer=tokenizer) == "This is a"


def test_truncate_no_truncation() -> None:
    text = "Short text"
    max_len = 20
    tokenizer = load_tokenizer("gpt-3.5-turbo-0125")
    assert truncate_str(text, max_len, tokenizer=tokenizer) == text


def test_truncate_empty_string() -> None:
    text = ""
    max_len = 5
    tokenizer = load_tokenizer("gpt-3.5-turbo-0125")
    assert truncate_str(text, max_len, tokenizer=tokenizer) == ""


def test_truncate_model_not_found() -> None:
    text = "This will use the base encoding."
    max_len = 3
    tokenizer = load_tokenizer("unknown-model")
    assert truncate_str(text, max_len, tokenizer=tokenizer) == "This will use"


def test_num_tokens_in_string() -> None:
    text = "This is a test sentence for the function."
    tokenizer = load_tokenizer("gpt-3.5-turbo-0125")
    assert num_tokens_in_string(text, tokenizer=tokenizer) == 9


def test_num_tokens_in_string_wrong_model() -> None:
    text = "This is a test sentence for the function."
    tokenizer = load_tokenizer("gpt-3.5-turbo-0125")
    assert num_tokens_in_string(text, tokenizer=tokenizer) == 9


# Tests for num_tokens_from_messages function
def test_num_tokens_single_message() -> None:
    messages = [{"role": "user", "content": "Hello!"}]
    tokenizer = load_tokenizer("gpt-3.5-turbo-0125")
    assert num_tokens_from_messages(messages, tokenizer=tokenizer, model="gpt-3.5-turbo-0125") == 9


def test_num_tokens_multiple_messages() -> None:
    messages = [{"role": "system", "content": "System message."}, {"role": "user", "content": "User message."}]
    tokenizer = load_tokenizer("gpt-3.5-turbo-0125")
    assert num_tokens_from_messages(messages, tokenizer=tokenizer, model="gpt-3.5-turbo-0125") == 17


def test_num_tokens_with_names() -> None:
    messages = [
        {"role": "user", "name": "Alice", "content": "Hi!"},
        {"role": "user", "name": "Bob", "content": "Hello!"},
    ]
    tokenizer = load_tokenizer("gpt-3.5-turbo-0125")
    assert num_tokens_from_messages(messages, tokenizer=tokenizer, model="gpt-3.5-turbo-0125") == 19


def test_num_tokens_unknown_model() -> None:
    messages = [{"role": "user", "content": "Hello!"}]
    tokenizer = load_tokenizer("unknown-model")
    with pytest.raises(NotImplementedError):
        num_tokens_from_messages(messages, tokenizer=tokenizer, model="unknown-model")
