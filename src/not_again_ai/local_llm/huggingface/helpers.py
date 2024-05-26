from typing import Any

from transformers import AutoModelForCausalLM, AutoProcessor


def load_model(model_id: str, device_map: str = "cuda", trust_remote_code: bool = True) -> Any:
    """Load a model from Hugging Face."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
    )
    return model


def load_processor(model_id: str, trust_remote_code: bool = True) -> Any:
    """Load a processor from Hugging Face. This is typically used for multimodal language models."""
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    return processor
