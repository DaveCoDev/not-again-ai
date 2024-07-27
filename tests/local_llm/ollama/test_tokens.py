import pytest

from not_again_ai.local_llm.ollama.tokens import (
    load_tokenizer,
    num_tokens_from_messages,
    num_tokens_in_string,
    truncate_str,
)

MODELS = [
    "phi3",
    "llama3:8b",
    "llama3.1:8b-instruct-q4_0",
    "gemma:7b-instruct",
    "qwen2",
    "granite-code:20b",
    "llama3-gradient:8b",
    "command-r:35b",
    "deepseek-coder-v2:16b",
    "other-model",
]


@pytest.fixture(params=MODELS)
def model(request):  # type: ignore
    return request.param


# Tests for truncate_str function
def test_truncate_normal(model: str) -> None:
    text = "This is a test sentence for the function."
    max_len = 3  # Assuming 'This is a' is within 3 tokens
    tokenizer = load_tokenizer(model)
    result = truncate_str(text, max_len, tokenizer=tokenizer)
    print(result)


def test_truncate_no_truncation(model: str) -> None:
    text = "Short text"
    max_len = 60
    tokenizer = load_tokenizer(model)
    assert truncate_str(text, max_len, tokenizer=tokenizer) == text


def test_truncate_empty_string(model: str) -> None:
    text = ""
    max_len = 5
    tokenizer = load_tokenizer(model)
    assert truncate_str(text, max_len, tokenizer=tokenizer) == ""


def test_truncate_model_not_found() -> None:
    text = "This will use the base encoding."
    max_len = 3
    tokenizer = load_tokenizer("unknown-model")
    result = truncate_str(text, max_len, tokenizer=tokenizer)
    print(result)


def test_num_tokens_in_string(model: str) -> None:
    text = "This is a test sentence for the function."
    tokenizer = load_tokenizer(model)
    result = num_tokens_in_string(text, tokenizer=tokenizer)
    print(result)


def test_num_tokens_in_string_wrong_model() -> None:
    text = "This is a test sentence for the function."
    tokenizer = load_tokenizer("unknown-model")
    result = num_tokens_in_string(text, tokenizer=tokenizer)
    print(result)


# Tests for num_tokens_from_messages function
def test_num_tokens_single_message(model: str) -> None:
    messages = [{"role": "user", "content": "Hello!"}]
    tokenizer = load_tokenizer(model)
    result = num_tokens_from_messages(messages, tokenizer=tokenizer)
    print(result)


def test_num_tokens_multiple_messages(model: str) -> None:
    messages = [
        {"role": "user", "content": "This is a user message."},
        {"role": "assistant", "content": "This is an assistant response."},
    ]
    tokenizer = load_tokenizer(model)
    result = num_tokens_from_messages(messages, tokenizer=tokenizer)
    print(f"{model} result: {result}")


def test_num_tokens_unknown_model() -> None:
    messages = [{"role": "user", "content": "Hello!"}]
    tokenizer = load_tokenizer("unknown-model")

    result = num_tokens_from_messages(messages, tokenizer=tokenizer)
    print(result)


if __name__ == "__main__":
    test_num_tokens_single_message(MODELS[2])
