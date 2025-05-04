from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ImageGenRequest(BaseModel):
    prompt: str
    model: str
    images: list[Path] | None = Field(default=None)
    mask: Path | None = Field(default=None)
    n: int = Field(default=1)
    quality: str | None = Field(default=None)
    size: str | None = Field(default=None)
    background: str | None = Field(default=None)
    moderation: str | None = Field(default=None)


class ImageGenResponse(BaseModel):
    images: list[bytes]
    input_tokens: int
    output_tokens: int
    response_duration: float
    input_tokens_details: dict[str, Any] | None = Field(default=None)
