"""Utilities for transcribing documents."""

import re


def transcription_extractor(query: str) -> str:
    """Extract transcription content from an LLM answer string.

    Parameters
    ----------
    query : str
        The LLM response string containing transcription XML tags

    Returns
    -------
    str
        The extracted transcription content, stripped of whitespace
    """
    # First try within transcription tags
    pattern = r"<transcription>(.*?)</transcription>"
    matches = re.findall(pattern, query, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try from opening tag
    pattern = r"<document_sections>(.*)"
    matches = re.findall(pattern, query, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Return everything if no tags found
    return query.strip()
