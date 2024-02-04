import pytest

from not_again_ai.llm.tokens import num_tokens_from_messages, num_tokens_in_string, truncate_str


# Tests for truncate_str function
def test_truncate_normal() -> None:
    text = "This is a test sentence for the function."
    max_len = 3  # Assuming 'This is a' is within 3 tokens
    assert truncate_str(text, max_len, model="gpt-3.5-turbo-0125") == "This is a"


def test_truncate_no_truncation() -> None:
    text = "Short text"
    max_len = 20
    assert truncate_str(text, max_len, model="gpt-3.5-turbo-0125") == text


def test_truncate_empty_string() -> None:
    text = ""
    max_len = 5
    assert truncate_str(text, max_len, model="gpt-3.5-turbo-0125") == ""


def test_truncate_model_not_found() -> None:
    text = "This will use the base encoding."
    max_len = 3
    assert truncate_str(text, max_len, model="unknown-model") == "This will use"


def test_num_tokens_in_string() -> None:
    text = "This is a test sentence for the function."
    assert num_tokens_in_string(text, model="gpt-3.5-turbo-0125") == 9


def test_num_tokens_in_string_wrong_model() -> None:
    text = "This is a test sentence for the function."
    assert num_tokens_in_string(text, model="unknown-model") == 9


# Tests for num_tokens_from_messages function
def test_num_tokens_single_message() -> None:
    messages = [{"role": "user", "content": "Hello!"}]
    assert num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125") == 9


def test_num_tokens_multiple_messages() -> None:
    messages = [{"role": "system", "content": "System message."}, {"role": "user", "content": "User message."}]
    assert num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125") == 17


def test_num_tokens_with_names() -> None:
    messages = [
        {"role": "user", "name": "Alice", "content": "Hi!"},
        {"role": "user", "name": "Bob", "content": "Hello!"},
    ]
    assert num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125") == 19


def test_num_tokens_unknown_model() -> None:
    messages = [{"role": "user", "content": "Hello!"}]
    with pytest.raises(NotImplementedError):
        num_tokens_from_messages(messages, model="unknown-model")
