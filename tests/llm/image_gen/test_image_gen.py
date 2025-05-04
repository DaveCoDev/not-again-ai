from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from not_again_ai.llm.image_gen import ImageGenRequest, create_image
from not_again_ai.llm.image_gen.providers.openai_api import openai_client

image_dir = Path(__file__).parents[1] / "sample_images"
body_lotion_image_path = image_dir / "body_lotion.png"
soap_image_path = image_dir / "soap.png"
sunlit_lounge_image_path = image_dir / "sunlit_lounge.png"
sunlit_lounge_mask_image_path = image_dir / "sunlit_lounge_mask.png"

save_dir = Path(__file__).parents[3] / ".nox" / "temp"
save_dir.mkdir(parents=True, exist_ok=True)
temp_image_path = save_dir / "temp_image.png"


@pytest.fixture(
    params=[
        {},
    ]
)
def openai_aoai_client_fixture(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return openai_client(**request.param)


def test_create_image(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    prompt = """A children's book drawing of a veterinarian using a stethoscope to 
listen to the heartbeat of a baby otter."""
    request = ImageGenRequest(
        prompt=prompt,
        model="gpt-image-1",
        quality="low",
    )
    response = create_image(request, "openai", openai_aoai_client_fixture)

    with temp_image_path.open("wb") as f:
        f.write(response.images[0])


def test_edit_image(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    prompt = """Generate a photorealistic image of a gift basket on a white background 
labeled 'Relax & Unwind' with a ribbon and handwriting-like font, 
containing all the items in the reference pictures."""
    request = ImageGenRequest(
        prompt=prompt,
        model="gpt-image-1",
        images=[body_lotion_image_path],
        quality="low",
    )
    response = create_image(request, "openai", openai_aoai_client_fixture)

    with temp_image_path.open("wb") as f:
        f.write(response.images[0])


def test_edit_images(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    prompt = """Generate a photorealistic image of a gift basket on a white background 
labeled 'Relax & Unwind' with a ribbon and handwriting-like font, 
containing all the items in the reference pictures."""
    request = ImageGenRequest(
        prompt=prompt,
        model="gpt-image-1",
        images=[body_lotion_image_path, soap_image_path],
        quality="low",
        size="1024x1024",
    )
    response = create_image(request, "openai", openai_aoai_client_fixture)

    with temp_image_path.open("wb") as f:
        f.write(response.images[0])


def test_edit_image_with_mask(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    request = ImageGenRequest(
        prompt="A sunlit indoor lounge area with a pool containing a flamingo",
        model="gpt-image-1",
        images=[sunlit_lounge_image_path],
        mask=sunlit_lounge_mask_image_path,
        quality="low",
        size="1024x1024",
    )
    response = create_image(request, "openai", openai_aoai_client_fixture)
    with temp_image_path.open("wb") as f:
        f.write(response.images[0])


def test_create_image_multiple(openai_aoai_client_fixture: Callable[..., Any]) -> None:
    prompt = """A high-quality 3D-rendered illustration of a color wheel logo. \
The design features eight symmetrical, petal-shaped leaves arranged in a perfect circular flower pattern. \
Each leaf is semi-transparent like colored glass, rendered in soft pastel tones including pink, orange, yellow, green, blue, and purple. \
The petals overlap slightly, creating gentle blended hues where they intersect. \
The background is a flat, light-toned surface with even, diffused lighting, giving the image a modern, polished, and professional appearance. No text."""

    request = ImageGenRequest(
        prompt=prompt,
        model="gpt-image-1",
        n=2,
        quality="medium",
        size="1536x1024",
        background="transparent",
        moderation="low",
    )

    response = create_image(request, "openai", openai_aoai_client_fixture)

    with temp_image_path.open("wb") as f:
        f.write(response.images[0])
    with (temp_image_path.parent / "temp_image_1.png").open("wb") as f:
        f.write(response.images[1])
