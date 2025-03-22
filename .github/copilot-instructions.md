# Python Rules
- The user is using Python version >= 3.11 with uv as the Python package and project manager.
- Follow the Google Python Style Guide.
- Instead of importing `Optional` from typing, using the `| `syntax.
- Always add appropriate type hintssuch that the code would pass a mypy type check.
- For type hints, use `list`, not `List`. For example, if the variable is `[{"name": "Jane", "age": 32}, {"name": "Amy", "age": 28}]` the type hint should be `list[dict]`
- If the user is using Pydantic, it is version >=2.10
- Always prefer pathlib for dealing with files. Use `Path.open` instead of `open`.
- Prefer to use pendulum instead of datetime
- Prefer to use loguru instead of logging