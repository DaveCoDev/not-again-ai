import pytest

from not_again_ai.llm.openai_api.context_management import priority_truncation
from not_again_ai.llm.openai_api.tokens import load_tokenizer, num_tokens_from_messages

MODELS = ["gpt-3.5-turbo-0125"]


@pytest.fixture(params=MODELS)
def model(request):  # type: ignore
    return request.param


def test_priority_truncation_simple(model: str) -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Help me {{task}}"},
    ]
    vars = {"task": "write Python code for the fibonnaci sequence"}
    truncated_messages = priority_truncation(messages, vars, ["task"], 26, model=model)
    tokenizer = load_tokenizer(model)
    assert num_tokens_from_messages(truncated_messages, tokenizer=tokenizer, model=model) <= 26
    truncated_messages_expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Help me write Python code for"},
    ]
    assert truncated_messages == truncated_messages_expected


def test_priority_truncation_two_vars(model: str) -> None:
    """Two different variables, first fits but second needs to be truncated"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Help me {{task}} and I will give you {{gift}}!"},
    ]
    vars = {"task": "write Python code for the fibonnaci sequence", "gift": "a few million dollars"}
    truncated_messages = priority_truncation(messages, vars, ["task", "gift"], 39, model=model)
    tokenizer = load_tokenizer(model)
    assert num_tokens_from_messages(truncated_messages, tokenizer=tokenizer, model=model) <= 39
    truncated_messages_expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Help me write Python code for the fibonnaci and I will give you a few million!"},
    ]
    assert truncated_messages == truncated_messages_expected


def test_priority_truncation_two_vars_same_message(model: str) -> None:
    """Two of the same variables in one message, both get truncated equally"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Help me {{task}} and I will give you {{task}}!"},
    ]
    vars = {"task": "write Python code for the fibonnaci sequence"}
    truncated_messages = priority_truncation(messages, vars, ["task"], 36, model=model)
    tokenizer = load_tokenizer(model)
    assert num_tokens_from_messages(truncated_messages, tokenizer=tokenizer, model=model) <= 36
    truncated_messages_expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Help me write Python and I will give you write Python!"},
    ]
    assert truncated_messages == truncated_messages_expected


def test_priority_truncation_two_vars_two_messages(model: str) -> None:
    """Two of the same variables in two messages, both get truncated equally"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Help me {{task}}!"},
        {"role": "assistant", "content": "I will give you {{task}}!"},
    ]
    vars = {"task": "write Python code for the fibonnaci sequence"}
    truncated_messages = priority_truncation(messages, vars, ["task"], 39, model=model)
    tokenizer = load_tokenizer(model)
    assert num_tokens_from_messages(truncated_messages, tokenizer=tokenizer, model=model) <= 39
    truncated_messages_expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Help me write Python!"},
        {"role": "assistant", "content": "I will give you write Python!"},
    ]
    assert truncated_messages == truncated_messages_expected


def test_priority_truncation_missing_priority(model: str) -> None:
    """No truncation needed, test missing priority"""
    messages = [
        {"role": "user", "content": "Help me {{task}} and I will give you {{gift}}!"},
        {"role": "assistant", "content": "Ok! Here is the {{code}}!"},
    ]
    vars = {
        "task": "write Python code for the fibonnaci sequence",
        "gift": "a few million dollars",
        "code": "print('Hello World')",
    }
    truncated_messages = priority_truncation(messages, vars, ["task", "gift"], 200, model=model)
    tokenizer = load_tokenizer(model)
    assert num_tokens_from_messages(truncated_messages, tokenizer=tokenizer, model=model) <= 200
    truncated_messages_expected = [
        {
            "role": "user",
            "content": "Help me write Python code for the fibonnaci sequence and I will give you a few million dollars!",
        },
        {"role": "assistant", "content": "Ok! Here is the print('Hello World')!"},
    ]
    assert truncated_messages == truncated_messages_expected
