"""Centralised definitions for special tokens shared across modules."""

from __future__ import annotations

from typing import Tuple

RDF_OBJECT_LIST_TOKEN: str = "<OBJ_LIST>"
RDF_LIST_SEPARATOR_TOKEN: str = "|"

# Expose an immutable tuple so callers can iterate without mutating globals.
REQUIRED_SPECIAL_TOKENS: Tuple[str, ...] = (
    RDF_OBJECT_LIST_TOKEN,
    RDF_LIST_SEPARATOR_TOKEN,
)

__all__ = [
    "RDF_OBJECT_LIST_TOKEN",
    "RDF_LIST_SEPARATOR_TOKEN",
    "REQUIRED_SPECIAL_TOKENS",
]
