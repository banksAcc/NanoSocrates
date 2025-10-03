import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.decoding.base import decode_to_text, greedy_decode


class DummyModel:
    def eval(self):
        pass

    def __call__(self, inp, att, decoder_input_ids):
        seq_len = decoder_input_ids.shape[1]
        vocab_size = 3
        logits = torch.full((1, seq_len, vocab_size), -1e9, dtype=torch.float)
        logits[:, -1, 2] = 0  # force <EOT>
        return {"logits": logits}


class DummyTokenizer:
    def __init__(self):
        self.pad_id = 1
        self.tk = self
        self.decoded = None

    def encode(self, text):
        return [5]

    def token_to_id(self, token):
        mapping = {"<SOT>": 0, "<EOT>": 2}
        return mapping.get(token, None)

    def decode(self, ids):
        self.decoded = ids
        return "decoded"


def test_greedy_decode_uses_start_token_and_strips_on_decode():
    model = DummyModel()
    tok = DummyTokenizer()

    ids = greedy_decode(model, tok, "prompt", max_new_tokens=2)

    assert ids[0] == 0

    text = decode_to_text(model, tok, "prompt", max_new_tokens=2)

    assert tok.decoded == [2]
    assert text == "decoded"
