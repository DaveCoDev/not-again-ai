import importlib.util

if importlib.util.find_spec("playwright") is None:
    raise ImportError(
        "not_again_ai.data requires the 'data' extra to be installed. "
        "You can install it using 'pip install not_again_ai[data]'."
    )
