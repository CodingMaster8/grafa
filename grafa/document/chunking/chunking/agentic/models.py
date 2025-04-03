"""Models for the agentic chunker."""
from pydantic import BaseModel, Field, field_validator


class AgenticChunkingOutput(BaseModel):
    """Model for the output of the agentic chunker."""

    updated_summary: str = Field(description="Summary of processed lines")
    line_number: int = Field(description="Specific line index where text is cut")

    @field_validator("line_number")
    def validate_line_number_is_valid(cls, value):
        """Validate that the line number is a non-negative integer."""
        if not isinstance(value, int):
            raise ValueError(f"{value} is not a valid integer.")
        if value < 0:
            raise ValueError(f"line_number must be a non-negative integer. Got {value}.")
        return value
