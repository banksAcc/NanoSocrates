"""Regression tests for decoding utilities (greedy and beam search)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decoding.base import decode_to_text
from src.data.serialization import parse_rdf


class DummyTokenizer:
    def __init__(self, vocab: Iterable[str]) -> None:
        self._tokens = list(vocab)
        self._token_to_id = {tok: idx for idx, tok in enumerate(self._tokens)}

    def encode(self, text: str) -> list[int]:
        pad_id = self._token_to_id["<pad>"]
        return [pad_id]

    def token_to_id(self, token: str) -> int | None:
        return self._token_to_id.get(token)

    def id_to_token(self, idx: int) -> str | None:
        if 0 <= idx < len(self._tokens):
            return self._tokens[idx]
        return None

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        tokens = [self.id_to_token(int(idx)) or "" for idx in ids]
        if skip_special_tokens:
            tokens = [tok for tok in tokens if not (tok.startswith("<") and tok.endswith(">"))]
        return " ".join(tokens).strip()

    def get_vocab_size(self) -> int:
        return len(self._tokens)


class DummyTokWrapper:
    def __init__(self, vocab: Iterable[str]) -> None:
        self.tk = DummyTokenizer(vocab)
        pad_id = self.tk.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("Vocabulary must include <pad>")
        self._pad_id = pad_id

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text)

    def token_to_id(self, token: str) -> int | None:
        return self.tk.token_to_id(token)

    @property
    def pad_id(self) -> int:
        return int(self._pad_id)


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int, transitions: Dict[Tuple[int, ...], Dict[int, float]]) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.transitions = transitions

    def forward(self, encoder_input_ids, attention_mask, decoder_input_ids):
        batch_size, seq_len = decoder_input_ids.size()
        device = decoder_input_ids.device
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            -100.0,
            dtype=torch.float32,
            device=device,
        )
        for i in range(batch_size):
            prefix = tuple(int(tok) for tok in decoder_input_ids[i].tolist())
            dist = self.transitions.get(prefix, {})
            row = logits[i, -1]
            for token_id, score in dist.items():
                if 0 <= token_id < self.vocab_size:
                    row[token_id] = float(score)
        return {"logits": logits}


def _decode_ids(tokenizer: DummyTokWrapper, ids: Iterable[int]) -> list[str]:
    return [tokenizer.tk.id_to_token(int(idx)) or "" for idx in ids]


def test_rdf_grammar_masks_invalid_transitions():
    vocab = ["<pad>", "<SOT>", "<EOT>", "<SUBJ>", "<PRED>", "<OBJ>", "A", "B", "C"]
    tok = DummyTokWrapper(vocab)
    transitions = {
        (1,): {5: 5.0, 3: 4.0},  # Prefer <OBJ> but grammar should enforce <SUBJ>
        (1, 3): {6: 5.0},
        (1, 3, 6): {4: 5.0},
        (1, 3, 6, 4): {7: 5.0},
        (1, 3, 6, 4, 7): {5: 5.0},
        (1, 3, 6, 4, 7, 5): {8: 5.0},
        (1, 3, 6, 4, 7, 5, 8): {2: 5.0},
    }
    model = DummyModel(len(vocab), transitions)

    _, ids = decode_to_text(
        model,
        tok,
        "",
        max_new_tokens=7,
        device="cpu",
        return_ids=True,
        use_beam_search=False,
        enforce_rdf_grammar=True,
    )
    tokens = _decode_ids(tok, ids)

    assert tokens[0] == "<SUBJ>"
    triples = parse_rdf(" ".join(["<SOT>"] + tokens))
    assert triples == [("A", "B", "C")]


def test_default_decoding_does_not_force_rdf_structure_for_textual_tasks():
    vocab = ["<pad>", "<SOT>", "<EOT>", "<SUBJ>", "<PRED>", "<OBJ>", "hello"]
    tok = DummyTokWrapper(vocab)
    transitions = {
        (1,): {6: 5.0, 3: 4.0},  # Prefer "hello" over <SUBJ>
        (1, 6): {2: 5.0},
    }
    model = DummyModel(len(vocab), transitions)

    text, ids = decode_to_text(
        model,
        tok,
        "",
        max_new_tokens=5,
        device="cpu",
        return_ids=True,
        use_beam_search=False,
    )

    tokens = _decode_ids(tok, ids)
    assert tokens[0] == "hello"
    assert text.strip().startswith("hello")


def test_beam_search_avoids_repeated_trigrams_and_preserves_rdf_structure():
    vocab = [
        "<pad>",
        "<SOT>",
        "<EOT>",
        "<SUBJ>",
        "<PRED>",
        "<OBJ>",
        "A",
        "B",
        "C",
        "D",
    ]
    tok = DummyTokWrapper(vocab)
    transitions = {
        (1,): {3: 5.0},
        (1, 3): {6: 6.0, 9: 4.0},
        (1, 3, 6): {4: 5.0},
        (1, 3, 6, 4): {7: 5.0},
        (1, 3, 6, 4, 7): {5: 5.0},
        (1, 3, 6, 4, 7, 5): {8: 5.0},
        (1, 3, 6, 4, 7, 5, 8): {2: 5.0},
    }
    model = DummyModel(len(vocab), transitions)

    _, ids = decode_to_text(
        model,
        tok,
        "",
        max_new_tokens=20,
        device="cpu",
        return_ids=True,
        use_beam_search=True,
        enforce_rdf_grammar=True,
    )
    tokens = _decode_ids(tok, ids)

    # Beam search should select the highest scoring subject token ("A").
    assert tokens[1] == "A"

    triples = parse_rdf(" ".join(["<SOT>"] + tokens))
    assert triples == [("A", "B", "C")]

    # Check that no trigram is repeated in the generated sequence.
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    assert len(trigrams) == len(set(trigrams))


def test_beam_search_prefers_completed_sequences_over_partial_candidates():
    vocab = [
        "<pad>",
        "<SOT>",
        "<EOT>",
        "A",
        "B",
        "C",
        "D",
        "X",
    ]
    tok = DummyTokWrapper(vocab)
    transitions = {
        (1,): {3: 0.0, 6: 0.0},
        (1, 3): {4: 3.0, 2: 2.0},
        (1, 6): {2: 1.5, 7: 1.0},
    }
    model = DummyModel(len(vocab), transitions)

    _, ids = decode_to_text(
        model,
        tok,
        "",
        max_new_tokens=4,
        device="cpu",
        return_ids=True,
        use_beam_search=True,
        beam_size=2,
    )

    tokens = _decode_ids(tok, ids)

    assert tokens[-1] == "<EOT>", "beam search should prioritise finished sequences"
    assert "B" not in tokens, "partial continuations should not outrank completed beams"
