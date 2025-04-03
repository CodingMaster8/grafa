"""Agentic Chunker. Uses an LLM to determine the best line to cut the document.

Also provides a summary of the document.
"""

from .agentic_chunker import agentic_chunking

__all__ = ["agentic_chunking"]
