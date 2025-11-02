"""Grafa is a Python library for generating, analyzing and querying knowledge graphs."""

from .client import GrafaClient, GrafaConfig
from .__about__ import __version__

__all__ = ["__version__", "GrafaClient", "GrafaConfig"]
