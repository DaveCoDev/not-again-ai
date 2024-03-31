import importlib.util

if (
    importlib.util.find_spec("liquid") is None
    or importlib.util.find_spec("openai") is None
    or importlib.util.find_spec("tiktoken") is None
):
    raise ImportError(
        "not_again_ai.llm requires the 'llm' extra to be installed. "
        "You can install it using 'pip install not_again_ai[llm]'."
    )
else:
    import liquid  # noqa: F401
    import openai  # noqa: F401
    import tiktoken  # noqa: F401
