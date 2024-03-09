try:
    import numpy
    import scipy
    import sklearn
except ImportError:
    raise ImportError(  # noqa
        "not_again_ai.statistics requires the 'statistics' extra to be installed. "
        "You can install it using 'pip install not_again_ai[statistics]'."
    )
