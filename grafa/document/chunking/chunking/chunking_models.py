"""Models for the chunker."""
from pydantic import BaseModel, Field


class ChunkOutput(BaseModel):
    """Model for the output of the chunker."""

    chunks: list[str] = Field(description="List of chunks")
    summary: str = Field(description="Summary of processed lines")
