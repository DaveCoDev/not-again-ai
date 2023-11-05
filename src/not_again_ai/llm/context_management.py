from pathlib import Path


def priority_truncation(
    path: Path, variables: dict[str, str], priority: list[str], token_limit: int
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": """- You are a helpful assistant trying to extract places that occur in a given text.""",
        }
    ]
