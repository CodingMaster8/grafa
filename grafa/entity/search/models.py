"""Template model for the entity search."""
from pydantic import BaseModel, Field


class ConceptIdentificationOutput(BaseModel):
    """Output of the concept identification."""

    entities: list[str] = Field(..., description="List of identified concepts")
