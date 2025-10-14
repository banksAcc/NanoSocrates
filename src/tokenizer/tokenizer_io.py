"""Utility wrappers around Hugging Face tokenizers used by NanoSocrates."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from tokenizers import Tokenizer

from src.utils.special_tokens import REQUIRED_SPECIAL_TOKENS

LOGGER = logging.getLogger(__name__)


def ensure_runtime_special_tokens(
    tokenizer: Tokenizer,
    required_tokens: Sequence[str] = REQUIRED_SPECIAL_TOKENS,
) -> None:
    """Ensure *required_tokens* exist in *tokenizer* at runtime.

    When legacy vocabularies are missing freshly introduced markers we extend
    them on the fly so training/evaluation remain backward compatible.
    """

    missing = [tok for tok in required_tokens if tokenizer.token_to_id(tok) is None]
    if not missing:
        return

    added = 0
    add_special = getattr(tokenizer, "add_special_tokens", None)
    if callable(add_special):
        added = add_special(list(missing))
    else:
        add_tokens = getattr(tokenizer, "add_tokens", None)
        if callable(add_tokens):
            added = add_tokens(list(missing))

    if added:
        LOGGER.debug("Added %s to tokenizer vocabulary", missing)

    leftover = [tok for tok in missing if tokenizer.token_to_id(tok) is None]
    if leftover:
        LOGGER.warning(
            "Unable to register special tokens %s in tokenizer vocabulary", leftover
        )


class TokWrapper:
    """Thin convenience layer exposing a subset of :class:`Tokenizer` APIs."""
    def __init__(self, path: str) -> None:
        """Load the tokenizer from *path* and cache the pad token identifier."""
        self.tk = Tokenizer.from_file(path)
        ensure_runtime_special_tokens(self.tk)
        # Cache the pad identifier once so downstream callers can access it even
        # if the tokenizer does not expose a dedicated property, and fail fast
        # when the vocabulary misses the mandatory <pad> symbol.
        self._pad_id = self.tk.token_to_id("<pad>")
        if self._pad_id is None:
            raise ValueError("Tokenizer privo di <pad>: rigenera il BPE includendo <pad>.")

    def encode(self, text: str) -> list[int]:
        """Return the list of token ids representing *text*."""
        return self.tk.encode(text).ids

    def token_to_id(self, tok: str) -> int | None:
        """Map *tok* to its integer id without exposing the underlying tokenizer."""
        return self.tk.token_to_id(tok)

    def add_special_tokens(self, tokens: Iterable[str]) -> int:
        """Proxy ``add_special_tokens`` so downstream utilities can reuse it."""

        return self.tk.add_special_tokens(list(tokens))

    def add_tokens(self, tokens: Iterable[str]) -> int:
        """Proxy ``add_tokens`` keeping compatibility with the Tokenizer API."""

        return self.tk.add_tokens(list(tokens))

    @property
    def pad_id(self) -> int:
        """Expose the cached padding token id for downstream components."""
        return self._pad_id

    def vocab_size(self) -> int:
        """Return the current vocabulary size of the wrapped tokenizer."""
        return self.tk.get_vocab_size()
