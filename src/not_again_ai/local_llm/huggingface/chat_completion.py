from pathlib import Path
from typing import Any

from PIL import Image


def chat_completion_image(
    messages: list[dict[str, str]],
    images: list[Path] | None,
    model_processor: tuple[Any, Any],
    max_tokens: int | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """A wrapper around ision language model inference for multimodal language models from huggingface.

    Args:
        messages (list[dict[str, str]]): A list of messages to send to the model.
        images (list[Path] | None): A list of image paths to send to the model.
        model_processor (tuple[Any, Any]): A tuple containing the model and processor objects.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to None.
        temperature (float, optional): The temperature of the model. Increasing the temperature will make the model answer more creatively. Defaults to 0.7.

    Returns:
        dict[str, Any]: A dictionary with the following keys
            message (str): The content of the generated assistant message.
            completion_tokens (int): The number of tokens used by the model to generate the completion.
    """

    model, processor = model_processor

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if images:
        image_objects = [Image.open(image) for image in images]
        inputs = processor(prompt, image_objects, return_tensors="pt").to("cuda:0")
    else:
        inputs = processor(prompt, return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "num_beams": 1,
        "do_sample": True,
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    # Remove input tokens
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

    # Get the number of generated tokens
    completion_tokens = generate_ids.shape[1]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    response_data: dict[str, Any] = {}
    response_data["message"] = response[0]
    response_data["completion_tokens"] = completion_tokens
    return response_data
