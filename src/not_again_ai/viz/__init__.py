try:
    import numpy  # noqa
    import pandas  # noqa
    import seaborn  # noqa
except ImportError:
    raise ImportError(  # noqa
        "not_again_ai.viz requires the 'viz' extra to be installed. "
        "You can install it using 'pip install not_again_ai[viz]'."
    )
