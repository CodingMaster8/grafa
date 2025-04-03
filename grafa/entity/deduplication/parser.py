"""A parser that parses an integer from a string."""
from langchain.schema import BaseOutputParser


class IntegerOutputParser(BaseOutputParser[int]):
    """A parser that parses an integer from a string."""

    def get_format_instructions(self) -> str:
        """Get the format instructions for the parser."""
        return "Please respond with an integer value."

    def parse(self, text: str) -> int:
        """Parse the integer from the string."""
        try:
            return int(text.strip())
        except ValueError:
            raise ValueError(f"Could not parse '{text}' as an integer.")
