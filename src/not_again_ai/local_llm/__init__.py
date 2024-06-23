import importlib.util
import os

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

if (
    importlib.util.find_spec("liquid") is None
    or importlib.util.find_spec("ollama") is None
    or importlib.util.find_spec("openai") is None
    or importlib.util.find_spec("tiktoken") is None
    or importlib.util.find_spec("transformers") is None
):
    raise ImportError(
        "not_again_ai.local_llm requires the 'llm' and 'local_llm' extra to be installed. "
        "You can install it using 'pip install not_again_ai[llm,local_llm]'."
    )
else:
    import liquid  # noqa: F401
    import ollama  # noqa: F401
    import openai  # noqa: F401
    import tiktoken  # noqa: F401
    import transformers  # noqa: F401
    from transformers.utils import logging

    logging.disable_progress_bar()
    logging.set_verbosity_error()
