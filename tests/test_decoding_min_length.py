import torch

from src.decoding import base


class DummyTokenizer:
    def __init__(self):
        self._vocab = {"<pad>": 0, "<SOT>": 1, "<EOT>": 2, "token": 3}
        self.tk = self

    @property
    def pad_id(self):
        return 0

    def token_to_id(self, token: str):
        return self._vocab.get(token)

    def encode(self, text: str):
        return [3]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class DummyModel:
    def __init__(self, eot_id: int, fallback_id: int):
        self.eot_id = eot_id
        self.fallback_id = fallback_id

    def eval(self):
        return self

    def __call__(self, inp, att, decoder_input_ids=None, **_):
        step = decoder_input_ids.size(1)
        vocab = 6
        logits = torch.full((1, step, vocab), -1e3)
        logits[:, -1, self.eot_id] = 10.0
        logits[:, -1, self.fallback_id] = 9.0 - (step - 1)
        return {"logits": logits}


def test_greedy_decode_enforces_minimum_length():
    tok = DummyTokenizer()
    model = DummyModel(eot_id=2, fallback_id=3)
    text, ids = base.decode_to_text(
        model,
        tok,
        "<SOT>",
        max_new_tokens=3,
        device="cpu",
        min_new_tokens=1,
        return_ids=True,
    )
    assert ids[0] == tok.token_to_id("token")
    assert text.startswith(str(tok.token_to_id("token")))
