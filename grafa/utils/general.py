"""General utilities for the Grafa library."""

from typing import Any, Sequence


def is_iterable(i: Any) -> bool:
    """Check if the object is a non-string Iterable."""
    if isinstance(i, str) or not isinstance(i, Sequence):
        return False
    return True


def as_iterable(i: Any) -> Sequence[Any]:
    """Guard to wrap the object in a non-stirng Iterable."""
    if is_iterable(i):
        return i
    return [i]
