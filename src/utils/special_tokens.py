"""Centralised definitions for special tokens shared across modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover - only used for type checking
    from transformers import PreTrainedTokenizerBase

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
    "dbr:",
    "dbo:",
)

__all__ = [
    "RDF_OBJECT_LIST_TOKEN",
    "RDF_LIST_SEPARATOR_TOKEN",
    "REQUIRED_SPECIAL_TOKENS",
    "ensure_required_special_tokens",
]


def ensure_required_special_tokens(tokenizer: "PreTrainedTokenizerBase") -> bool:
    """Ensure all required special tokens exist in ``tokenizer``.

    Returns ``True`` when the tokenizer vocabulary was extended with any missing
    tokens and ``False`` otherwise.  Raises ``RuntimeError`` if the tokenizer
    fails to add the required tokens to its vocabulary.
    """

    vocab = tokenizer.get_vocab()
    missing_tokens = [tok for tok in REQUIRED_SPECIAL_TOKENS if tok not in vocab]
    if not missing_tokens:
        return False

    tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
    refreshed_vocab = tokenizer.get_vocab()
    remaining = [tok for tok in missing_tokens if tok not in refreshed_vocab]
    if remaining:
        raise RuntimeError(
            "Tokenizer failed to register required special tokens: " + ", ".join(remaining)
        )
    return True
