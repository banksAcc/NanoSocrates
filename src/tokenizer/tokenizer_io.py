"""Utility wrappers around Hugging Face tokenizers used by NanoSocrates."""

from tokenizers import Tokenizer


class TokWrapper:
    """Thin convenience layer exposing a subset of :class:`Tokenizer` APIs."""
    def __init__(self, path: str) -> None:
        """Load the tokenizer from *path* and cache the pad token identifier."""
        self.tk = Tokenizer.from_file(path)
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

    @property
    def pad_id(self) -> int:
        """Expose the cached padding token id for downstream components."""
        return self._pad_id

    def vocab_size(self) -> int:
        """Return the current vocabulary size of the wrapped tokenizer."""
        return self.tk.get_vocab_size()
