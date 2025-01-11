import pytest

from not_again_ai.llm.chat_completion.types import MessageT, SystemMessage, UserMessage
from not_again_ai.llm.prompting import Tokenizer


@pytest.fixture(
    params=[
        {"model": "gpt-4o-mini-2024-07-18", "provider": "openai"},
        {"model": "o1-mini", "provider": "azure_openai"},
    ]
)
def tokenizer(request: pytest.FixtureRequest) -> Tokenizer:
    return Tokenizer(**request.param)


@pytest.fixture(
    params=[
        {"model": "gpt-4o-mini-2024-07-18", "provider": "openai"},
        {"model": "o1-mini", "provider": "azure_openai"},
        {"model": "gpt-4o-mini-2024-07-18", "provider": "mistral"},
        {"model": "unknown-model", "provider": "openai"},
    ]
)
def tokenizer_with_unsupported(request: pytest.FixtureRequest) -> Tokenizer:
    return Tokenizer(**request.param)


# This test only runs with OpenAI and Azure
def test_truncate_str(tokenizer_with_unsupported: Tokenizer) -> None:
    text = "This is a test sentence for the function."
    max_len = 3
    result = tokenizer_with_unsupported.truncate_str(text, max_len)
    print(result)


# This test runs with all three providers
def test_truncate_str_no_truncation(tokenizer: Tokenizer) -> None:
    text = "Short text"
    max_len = 20
    result = tokenizer.truncate_str(text, max_len)
    print(result)


def test_num_tokens_in_str(tokenizer_with_unsupported: Tokenizer) -> None:
    text = "This is a test sentence for the function."
    result = tokenizer_with_unsupported.num_tokens_in_str(text)
    print(result)


def test_num_tokens_allowed_special() -> None:
    tokenizer = Tokenizer(
        model="gpt-4o-mini-2024-07-18", provider="openai", allowed_special=set(), disallowed_special=()
    )
    text = "<|endoftext|>"
    result = tokenizer.num_tokens_in_str(text)
    print(result)


def test_num_tokens_multiple_messages(tokenizer_with_unsupported: Tokenizer) -> None:
    messages: list[MessageT] = [SystemMessage(content="System message."), UserMessage(content="User message.")]
    result = tokenizer_with_unsupported.num_tokens_in_messages(messages)
    print(result)
