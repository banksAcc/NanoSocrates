"""Utility functions for computing seq2seq losses and accuracy."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _align_logits_and_labels(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align logits and labels for teacher-forced decoding."""

    if labels.size(1) == logits.size(1):
        return logits, labels
    if labels.size(1) == logits.size(1) + 1:
        return logits, labels[:, 1:]
    raise ValueError(
        "Incompatible sequence lengths: logits.size(1)="
        f"{logits.size(1)} labels.size(1)={labels.size(1)}"
    )


def sequence_loss_with_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    pad_id: int,
    compute_metrics: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute cross-entropy loss and optional exact-match rate."""

    logits_for_loss, target = _align_logits_and_labels(logits, labels)

    target_for_loss = target
    ignore_index = pad_id
    if (target == -100).any():
        ignore_index = -100
        if pad_id != ignore_index:
            target_for_loss = target_for_loss.masked_fill(target_for_loss == pad_id, ignore_index)

    loss = F.cross_entropy(
        logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
        target_for_loss.reshape(-1),
        ignore_index=ignore_index,
    )

    metrics: Dict[str, float] = {}
    if compute_metrics:
        with torch.no_grad():
            predictions = logits_for_loss.argmax(dim=-1)
            valid_mask = target_for_loss != ignore_index
            if valid_mask.any():
                comparison = (predictions == target) | ~valid_mask
                per_example = comparison.all(dim=1)
                metrics["exact_match"] = float(per_example.float().mean().item())

    return loss, metrics
