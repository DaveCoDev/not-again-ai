from typing import Any

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str
    dimensions: int | None = Field(default=None)


class EmbeddingObject(BaseModel):
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    embeddings: list[EmbeddingObject]
    total_tokens: int | None = Field(default=None)
    response_duration: float

    errors: str = Field(default="")
    extras: Any | None = Field(default=None)
