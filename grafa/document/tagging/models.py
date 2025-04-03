"""Models for the tagging module."""

from typing import List

from pydantic import BaseModel, Field


class TaggingOutput(BaseModel):
    """
    Model for the output of the tagging module.

    Attributes
    ----------
    tags : List[str]
        A list of tags extracted from the text chunk.
    """

    tags: List[str] = Field(..., description="List of extracted tags")
