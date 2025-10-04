from pathlib import Path
import sys

import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model.transformer import TinySeq2Seq


@pytest.mark.parametrize(
    "use_rope,use_mla,ratio",
    [
        (False, False, 0.0),
        (True, False, 0.0),
        (False, True, 1.0),
        (True, True, 0.5),
    ],
)
@torch.no_grad()
def test_tinyseq2seq_forward_variants(use_rope, use_mla, ratio):
    torch.manual_seed(0)
    vocab_size = 32
    model = TinySeq2Seq(
        vocab_size=vocab_size,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        pad_id=0,
        use_rope=use_rope,
        use_mla=use_mla,
        interleave_ratio=ratio,
        max_position_embeddings=64,
    )

    input_ids = torch.randint(1, vocab_size, (2, 5))
    input_ids[0, -1] = 0
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, -1] = 0

    labels = torch.randint(1, vocab_size, (2, 6))
    labels[:, -1] = 0

    out = model(input_ids, attention_mask, labels=labels)
    logits = out["logits"]
    loss = out["loss"]

    assert logits.shape == (input_ids.size(0), labels.size(1) - 1, vocab_size)
    assert torch.all(torch.isfinite(logits))
    assert loss is not None
    assert torch.isfinite(loss)


@torch.no_grad()
def test_interleave_ratio_clamped():
    torch.manual_seed(1)
    model = TinySeq2Seq(
        vocab_size=16,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        pad_id=0,
        use_mla=True,
        interleave_ratio=2.5,
        max_position_embeddings=32,
    )
    input_ids = torch.randint(1, 16, (1, 4))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(1, 16, (1, 5))

    out = model(input_ids, attention_mask, labels=labels)
    logits = out["logits"]
    assert logits.shape[1] == labels.size(1) - 1
    assert torch.all(torch.isfinite(logits))
