try:
    import openai  # noqa
except ImportError:
    raise ImportError(  # noqa
        "not_again_ai.llm requires the 'llm' extra to be installed. "
        "You can install it using 'pip install not_again_ai[llm]'."
    )