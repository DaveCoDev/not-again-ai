from not_again_ai.llm.prompts import chat_prompt

no_vars_prompt = [
    {
        "role": "system",
        "content": """This is a system message with no variables""",
    },
    {
        "role": "user",
        "content": """This is a user message with no variables""",
    },
]


place_extraction_prompt = [
    {
        "role": "system",
        "content": """- You are a helpful assistant trying to extract places that occur in a given text.
- You must identify all the places in the text and return them in a list like this: ["place1", "place2", "place3"].""",
    },
    {
        "role": "user",
        "content": """Here is the text I want you to extract places from:
{# The user's input text goes here -#}
{{text}}""",
    },
]


def test_chat_prompt_no_vars() -> None:
    messages_formatted = chat_prompt(no_vars_prompt, {})
    messages_expected = [
        {
            "role": "system",
            "content": """This is a system message with no variables""",
        },
        {
            "role": "user",
            "content": """This is a user message with no variables""",
        },
    ]
    assert messages_formatted == messages_expected


def test_chat_prompt_place_extraction() -> None:
    variables = {
        "text": "I went to Paris and Berlin.",
    }
    messages_formatted = chat_prompt(place_extraction_prompt, variables)
    messages_expected = [
        {
            "role": "system",
            "content": """- You are a helpful assistant trying to extract places that occur in a given text.
- You must identify all the places in the text and return them in a list like this: ["place1", "place2", "place3"].""",
        },
        {
            "role": "user",
            "content": """Here is the text I want you to extract places from:
I went to Paris and Berlin.""",
        },
    ]
    assert messages_formatted == messages_expected
