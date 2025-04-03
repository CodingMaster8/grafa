"""Semantic Agentic Chunker.

This module provides a semantic agentic chunker that uses an LLM to determine if a sentence should be part of a new chunk based on semantic meaning.
"""

from .semantic_agentic import semantic_agentic_chunking

__all__ = ["semantic_agentic_chunking"]
