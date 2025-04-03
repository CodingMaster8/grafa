"""String utilities."""
import re
import unicodedata


def escape_yaml_string(yaml_str: str) -> str:
    """Escape a YAML string for safe storage in Neo4j."""
    return yaml_str.replace('"', '\\"').replace("\n", "\\n")


def unescape_yaml_string(yaml_str: str) -> str:
    """Unescape a YAML string stored in Neo4j."""
    return yaml_str.replace('\\"', '"').replace("\\n", "\n")


def clean_string(text: str) -> str:
    """Clean string by removing special characters while preserving letters and numbers from all languages.

    Parameters
    ----------
    text : str
        Text to clean

    Returns
    -------
    str
        Cleaned text with only letters, numbers and spaces
    """
    # Categories we want to keep:
    # Lu : Uppercase Letter
    # Ll : Lowercase Letter
    # Lt : Titlecase Letter
    # Lo : Other Letter (includes Chinese/Japanese/Korean characters)
    # Nd : Decimal Number
    # Zs : Space Separator
    ALLOWED_CATEGORIES = {"Lu", "Ll", "Lt", "Lo", "Nd", "Zs"}

    # Keep only allowed characters
    cleaned = "".join(c for c in text if unicodedata.category(c) in ALLOWED_CATEGORIES)

    # Replace multiple spaces with single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()
