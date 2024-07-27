"""Hardcoded mapping from ollama model names to their associated HuggingFace tokenizer.

Given the way that Ollama models are tagged, we can against the first part of the model name,
i.e. all phi3 models will start with "phi3".
"""

OLLAMA_MODEL_MAPPING = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama3:": "nvidia/Llama3-ChatQA-1.5-8B",  # Using this version to get around needed to accept an agreement to get access to the tokenizer
    "llama3.1": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-1.1-7b-it",  # Requires HF_TOKEN set and accepting the agreement on the HF model page
    "qwen2": "Qwen/Qwen2-7B-Instruct",
    "granite-code": "ibm-granite/granite-34b-code-instruct",
    "llama3-gradient": "nvidia/Llama3-ChatQA-1.5-8B",
    "command-r": "CohereForAI/c4ai-command-r-v01",
    "deepseek-coder-v2": "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
}
