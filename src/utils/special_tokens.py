"""Centralised definitions for special tokens shared across modules."""

from __future__ import annotations

from typing import Tuple

RDF_OBJECT_LIST_TOKEN: str = "<OBJ_LIST>"
RDF_LIST_SEPARATOR_TOKEN: str = "|"

# Central list used to ensure that all task-specific markers are available
# regardless of the tokenizer generation used at runtime.
REQUIRED_SPECIAL_TOKENS: Tuple[str, ...] = (
    "<SOT>",
    "<EOT>",
    "<SUBJ>",
    "<PRED>",
    "<OBJ>",
    RDF_OBJECT_LIST_TOKEN,
    RDF_LIST_SEPARATOR_TOKEN,
    "<RDF2Text>",
    "<Text2RDF>",
    "<CONTINUERDF>",
    "<MASK>",
)

__all__ = [
    "RDF_OBJECT_LIST_TOKEN",
    "RDF_LIST_SEPARATOR_TOKEN",
    "REQUIRED_SPECIAL_TOKENS",
]
