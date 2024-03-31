import importlib.util

if (
    importlib.util.find_spec("numpy") is None
    or importlib.util.find_spec("scipy") is None
    or importlib.util.find_spec("sklearn") is None
):
    raise ImportError(
        "not_again_ai.statistics requires the 'statistics' extra to be installed. "
        "You can install it using 'pip install not_again_ai[statistics]'."
    )
else:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import sklearn  # noqa: F401
