from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from model.losses import sequence_loss_with_span_metrics


def _build_logits(vocab_size: int, correct_token: int, wrong: bool = False):
    logits = torch.full((1, 1, vocab_size), -10.0)
    target_token = (correct_token + 1) % vocab_size if wrong else correct_token
    logits[0, 0, target_token] = 10.0
    return logits


def test_sequence_loss_with_span_accuracy_correct():
    vocab_size = 8
    labels = torch.tensor([[5, 3]])  # first token is decoder input, second is target
    logits = _build_logits(vocab_size, correct_token=3)
    mask_positions = torch.tensor([[0]])
    mask_lengths = torch.tensor([[1]])
    loss, metrics = sequence_loss_with_span_metrics(
        logits,
        labels,
        pad_id=0,
        mask_positions=mask_positions,
        mask_lengths=mask_lengths,
        compute_metrics=True,
    )
    assert torch.isfinite(loss)
    assert metrics["mask_accuracy"] == 1.0


def test_sequence_loss_with_span_accuracy_incorrect():
    vocab_size = 8
    labels = torch.tensor([[9, 2]])
    logits = _build_logits(vocab_size, correct_token=2, wrong=True)
    mask_positions = torch.tensor([[0]])
    mask_lengths = torch.tensor([[1]])
    _, metrics = sequence_loss_with_span_metrics(
        logits,
        labels,
        pad_id=0,
        mask_positions=mask_positions,
        mask_lengths=mask_lengths,
        compute_metrics=True,
    )
    assert metrics["mask_accuracy"] == 0.0
