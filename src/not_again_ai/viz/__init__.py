import importlib.util

if (
    importlib.util.find_spec("numpy") is None
    or importlib.util.find_spec("pandas") is None
    or importlib.util.find_spec("seaborn") is None
):
    raise ImportError(
        "not_again_ai.viz requires the 'viz' extra to be installed. "
        "You can install it using 'pip install not_again_ai[viz]'."
    )
else:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import seaborn  # noqa: F401
