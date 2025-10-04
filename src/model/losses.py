"""Utility functions for computing seq2seq losses and auxiliary metrics."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _align_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tensors aligned for loss computation.

    The decoder may produce logits that are one step shorter than the labels
    (teacher forcing with implicit <SOT>). This helper mirrors the logic used in
    :class:`TinySeq2Seq` to keep the behaviour consistent in one place.
    """

    if labels.size(1) == logits.size(1):
        return logits, labels
    if labels.size(1) == logits.size(1) + 1:
        return logits, labels[:, 1:]
    raise ValueError(
        "labels length must match decoder_input_ids or be longer by one"
    )


def _compute_span_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_positions: Optional[torch.Tensor],
    mask_lengths: Optional[torch.Tensor],
    pad_id: int,
) -> Optional[float]:
    if mask_positions is None or mask_lengths is None:
        return None
    if mask_positions.numel() == 0:
        return None

    # ensure we work on CPU tensors for easier iteration
    pred_ids = logits.argmax(dim=-1)
    labels_cpu = labels.detach().cpu()
    preds_cpu = pred_ids.detach().cpu()
    pos_cpu = mask_positions.detach().cpu()
    len_cpu = mask_lengths.detach().cpu()

    total = 0
    correct = 0

    batch_size = labels_cpu.size(0)
    max_len = labels_cpu.size(1)

    for b in range(batch_size):
        positions = pos_cpu[b]
        lengths = len_cpu[b]
        for pos, span_len in zip(positions.tolist(), lengths.tolist()):
            if pos < 0 or span_len <= 0:
                continue
            end = pos + span_len
            if pos >= max_len or end > max_len:
                continue
            target_span = labels_cpu[b, pos:end]
            if (target_span == pad_id).any():
                continue
            pred_span = preds_cpu[b, pos:end]
            total += 1
            if torch.equal(target_span, pred_span):
                correct += 1

    if total == 0:
        return None
    return float(correct / total)


def sequence_loss_with_span_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    pad_id: int,
    mask_positions: Optional[torch.Tensor] = None,
    mask_lengths: Optional[torch.Tensor] = None,
    compute_metrics: bool = False,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute cross-entropy loss and optional span-based accuracy."""

    logits_for_loss, target = _align_logits_and_labels(logits, labels)
    loss = F.cross_entropy(
        logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
        target.reshape(-1),
        ignore_index=pad_id,
    )

    metrics: Dict[str, float] = {}
    if compute_metrics:
        acc = _compute_span_accuracy(
            logits_for_loss,
            target,
            mask_positions,
            mask_lengths,
            pad_id,
        )
        if acc is not None:
            metrics["mask_accuracy"] = acc

    return loss, metrics
