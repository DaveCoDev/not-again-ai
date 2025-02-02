from copy import deepcopy
from pathlib import Path

from pydantic import BaseModel

from not_again_ai.llm.chat_completion.types import (
    AssistantMessage,
    DeveloperMessage,
    Function,
    ImageContent,
    ImageDetail,
    ImageUrl,
    MessageT,
    SystemMessage,
    TextContent,
    ToolCall,
    UserMessage,
)
from not_again_ai.llm.prompting.compile_prompt import (
    compile_messages,
    compile_tools,
    create_image_url,
    pydantic_to_json_schema,
)

image_dir = Path(__file__).parent.parent / "sample_images"
sk_infographic = image_dir / "SKInfographic.png"
sk_diagram = image_dir / "SKDiagram.png"
cat_image = image_dir / "cat.jpg"
dog_image = image_dir / "dog.jpg"


def test_chat_prompt_no_vars() -> None:
    no_vars_prompt: list[MessageT] = [
        SystemMessage(
            content="This is a system message with no variables",
        ),
        UserMessage(
            content="This is a user message with no variables",
        ),
    ]

    messages_formatted = compile_messages(no_vars_prompt, {})
    messages_expected = [
        SystemMessage(
            content="This is a system message with no variables",
        ),
        UserMessage(
            content="This is a user message with no variables",
        ),
    ]
    assert messages_formatted == messages_expected


def test_compile_messages_place_extraction() -> None:
    place_extraction_prompt: list[MessageT] = [
        SystemMessage(
            content="""- You are a helpful assistant trying to extract places that occur in a given text.
- You must identify all the places in the text and return them in a list like this: ["place1", "place2", "place3"].""",
        ),
        UserMessage(
            content="""Here is the text I want you to extract places from:
{%- # The user's input text goes below %}
{{text}}""",
        ),
    ]
    variables = {
        "text": "I went to Paris and Berlin.",
    }
    messages_formatted = compile_messages(place_extraction_prompt, variables)
    messages_expected = [
        SystemMessage(
            content="""- You are a helpful assistant trying to extract places that occur in a given text.
- You must identify all the places in the text and return them in a list like this: ["place1", "place2", "place3"].""",
        ),
        UserMessage(
            content="""Here is the text I want you to extract places from:
I went to Paris and Berlin.""",
        ),
    ]
    assert messages_formatted == messages_expected


def test_compile_messages_vision() -> None:
    vision_prompt: list[MessageT] = [
        SystemMessage(
            content="You are a helpful assistant.",
        ),
        UserMessage(
            content=[
                TextContent(text="Describe the animal in the image in one word."),
                ImageContent(image_url=ImageUrl(url=create_image_url(cat_image), detail=ImageDetail.LOW)),
            ],
        ),
        AssistantMessage(
            content="{{ answer }}",
        ),
        UserMessage(
            content=[
                TextContent(text="What about this animal?"),
                ImageContent(image_url=ImageUrl(url=create_image_url(dog_image), detail=ImageDetail.LOW)),
            ],
        ),
    ]

    messages_formatted = compile_messages(vision_prompt, {"answer": "Cat"})
    messages_expected = [
        SystemMessage(
            content="You are a helpful assistant.",
        ),
        UserMessage(
            content=[
                TextContent(text="Describe the animal in the image in one word."),
                ImageContent(image_url=ImageUrl(url=create_image_url(cat_image), detail=ImageDetail.LOW)),
            ],
        ),
        AssistantMessage(
            content="Cat",
        ),
        UserMessage(
            content=[
                TextContent(text="What about this animal?"),
                ImageContent(image_url=ImageUrl(url=create_image_url(dog_image), detail=ImageDetail.LOW)),
            ],
        ),
    ]

    assert messages_formatted == messages_expected


def test_compile_messages_templates_other_string_fields() -> None:
    # Variables for Liquid template replacement.
    variables = {
        "user": "Alice",
        "func_var": "add",
        "toolcall_id": "toolcall1",
    }

    # Create messages with Liquid templates in 'name' fields and within a Function.
    system_message = SystemMessage(content="System content", name="System-{{user}}")
    developer_message = DeveloperMessage(content="Developer message", name="Dev-{{user}}")
    assistant_message = AssistantMessage(
        content="Assistant processing {{user}}",
        name="Assistant-{{user}}",
        tool_calls=[
            ToolCall(
                id="{{toolcall_id}}",
                function=Function(
                    name="function-{{func_var}}",
                    arguments={"op": "{{func_var}}", "constant": "nochange"},
                ),
            )
        ],
    )

    messages: list[MessageT] = [system_message, developer_message, assistant_message]
    compiled_messages = compile_messages(messages, variables)

    # The expected messages assume that Liquid templates in all string fields are processed.
    expected_system_message = SystemMessage(content="System content", name="System-Alice")
    expected_developer_message = DeveloperMessage(content="Developer message", name="Dev-Alice")
    expected_assistant_message = AssistantMessage(
        content="Assistant processing Alice",
        name="Assistant-Alice",
        tool_calls=[
            ToolCall(
                id="toolcall1",
                function=Function(
                    name="function-add",
                    arguments={"op": "add", "constant": "nochange"},
                ),
            )
        ],
    )
    expected_messages = [
        expected_system_message,
        expected_developer_message,
        expected_assistant_message,
    ]

    assert compiled_messages == expected_messages


def test_compile_messages_idempotence() -> None:
    """Test that compiling messages twice yields the same result."""
    msg = SystemMessage(content="Hello {{name}}", name="System-{{name}}")
    first_compile = compile_messages([msg], {"name": "World"})[0]
    second_compile = compile_messages([first_compile], {"name": "World"})[0]
    assert first_compile == second_compile


def test_compile_messages_does_not_modify_original() -> None:
    """Test that compile_messages does not modify the original messages."""
    original_msg = [SystemMessage(content="Hello {{name}}", name="System{{name}}")]
    messages_copy = deepcopy(original_msg)
    compiled_messages = compile_messages(original_msg, {"name": "Test"})
    # The original message should remain unchanged.
    assert original_msg == messages_copy
    # The compiled message should have substitutions applied.
    assert compiled_messages[0].content == "Hello Test"
    assert compiled_messages[0].name == "SystemTest"


def test_nested_structure_in_tool_calls() -> None:
    """Test that templating is recursively applied within nested structures such as tool_calls."""
    msg = AssistantMessage(
        content="Processing {{value}}",
        tool_calls=[
            ToolCall(
                id="{{call_id}}",
                function=Function(
                    name="func{{value}}",
                    arguments={"nested": {"key": "{{value}}"}},
                ),
            )
        ],
    )
    compiled_messages = compile_messages([msg], {"value": "100", "call_id": "call_1"})
    assert isinstance(compiled_messages[0], AssistantMessage), "Expected an AssistantMessage"
    compiled: AssistantMessage = compiled_messages[0]
    assert compiled.content == "Processing 100"
    assert compiled.tool_calls is not None
    tool_call = compiled.tool_calls[0]
    assert tool_call.id == "call_1"
    assert tool_call.function.name == "func100"
    assert tool_call.function.arguments == {"nested": {"key": "100"}}


def test_missing_variable_in_template() -> None:
    """Test that a missing variable in the template yields an empty substitution."""
    msg = SystemMessage(content="Hello {{name}}")
    compiled = compile_messages([msg], {})[0]
    # Assuming the missing variable renders as an empty string.
    assert compiled.content == "Hello "


def test_compile_tools_basic() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. {{location}}",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use, such as {{format}}.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        }
    ]

    expected_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use, such as celsius.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        }
    ]

    compiled_tools = compile_tools(tools, {"location": "San Francisco, CA", "format": "celsius"})
    assert compiled_tools == expected_tools


def test_compile_tools_no_change() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    assert compile_tools(tools, {}) == tools


def test_compile_tools_does_not_modify_original() -> None:
    """Test that compile_tools does not mutate the original list of tool dictionaries."""
    original_tools = [{"name": "Template: {{var}}"}]
    tools_copy = deepcopy(original_tools)
    variables = {"var": "A"}
    result = compile_tools(original_tools, variables)
    # Confirm that the original remains unchanged.
    assert original_tools == tools_copy
    # And that the result is modified as expected.
    expected = [{"name": "Template: A"}]
    assert result == expected


def test_pydantic_to_json_schema() -> None:
    class Step(BaseModel):
        explanation: str
        output: str

    class MathResponse(BaseModel):
        steps: list[Step]
        final_answer: str

    json_schema = pydantic_to_json_schema(MathResponse, schema_name="math_response", description="A math response")

    expected_schema = {
        "$defs": {
            "Step": {
                "properties": {
                    "explanation": {"title": "Explanation", "type": "string"},
                    "output": {"title": "Output", "type": "string"},
                },
                "required": ["explanation", "output"],
                "title": "Step",
                "type": "object",
                "additionalProperties": False,
            }
        },
        "properties": {
            "steps": {"items": {"$ref": "#/$defs/Step"}, "title": "Steps", "type": "array"},
            "final_answer": {"title": "Final Answer", "type": "string"},
        },
        "required": ["steps", "final_answer"],
        "title": "MathResponse",
        "type": "object",
        "additionalProperties": False,
    }

    assert json_schema["strict"] is True
    assert json_schema["name"] == "math_response"
    assert json_schema["description"] == "A math response"
    assert json_schema["schema"] == expected_schema
