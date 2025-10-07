"""Utility functions for computing seq2seq losses and auxiliary metrics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _align_logits_and_labels(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensures logits and labels are aligned for loss computation.

    In a typical teacher-forcing setup, the decoder input is `labels[:, :-1]`,
    producing logits that align with `labels[:, 1:]`. This function handles
    that alignment.

    Args:
        logits: The raw output from the model's language model head.
            Shape: (batch_size, seq_len, vocab_size)
        labels: The ground truth target tensor.
            Shape: (batch_size, seq_len) or (batch_size, seq_len + 1)

    Returns:
        A tuple of (logits, labels) ready for loss calculation.

    Raises:
        ValueError: If the dimensions of logits and labels are incompatible.
    """
    if labels.size(1) == logits.size(1):
        return logits, labels
    if labels.size(1) == logits.size(1) + 1:
        # This is the standard case: labels include a start token that was
        # not predicted, so we align by removing it.
        return logits, labels[:, 1:]
    raise ValueError(
        f"Incompatible shapes: logits.size(1)={logits.size(1)} and "
        f"labels.size(1)={labels.size(1)}. Labels must be same length as logits "
        "or one token longer."
    )


@torch.inference_mode()
def _compute_span_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_positions: Optional[torch.Tensor],
    mask_lengths: Optional[torch.Tensor],
) -> Optional[float]:
    """Computes exact match accuracy for specified token spans.

    This metric is primarily used for the 'RDF Completion 1' task, where the
    model must predict a masked-out span of tokens.

    Args:
        logits: The model's output logits, aligned with labels.
        labels: The ground truth labels, aligned with logits.
        mask_positions: A tensor of start positions for each masked span.
            Shape: (batch_size, num_spans)
        mask_lengths: A tensor of lengths for each masked span.
            Shape: (batch_size, num_spans)

    Returns:
        The percentage of correctly predicted spans, or None if no valid
        spans are found.
    """
    if mask_positions is None or mask_lengths is None or mask_positions.numel() == 0:
        return None

    pred_ids = logits.argmax(dim=-1)
    total_spans = 0
    correct_spans = 0

    for i in range(logits.size(0)):  # Iterate over batch
        for pos, length in zip(mask_positions[i], mask_lengths[i]):
            pos, length = pos.item(), length.item()
            if length <= 0:
                continue

            end_pos = pos + length
            if end_pos > labels.size(1):
                continue

            total_spans += 1
            target_span = labels[i, pos:end_pos]
            pred_span = pred_ids[i, pos:end_pos]

            if torch.equal(target_span, pred_span):
                correct_spans += 1

    return float(correct_spans / total_spans) if total_spans > 0 else None


def sequence_loss_with_span_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    pad_id: int,
    mask_positions: Optional[torch.Tensor] = None,
    mask_lengths: Optional[torch.Tensor] = None,
    compute_metrics: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes cross-entropy loss and optional span-based accuracy.

    Args:
        logits: The raw output from the model.
        labels: The ground truth labels.
        pad_id: The ID of the padding token, to be ignored in the loss.
        mask_positions: Start positions of spans for accuracy calculation.
        mask_lengths: Lengths of spans for accuracy calculation.
        compute_metrics: If True, calculates and returns the span accuracy.

    Returns:
        A tuple containing:
        - The computed cross-entropy loss (scalar tensor).
        - A dictionary of computed metrics (e.g., {"mask_accuracy": 0.85}).
    """
    logits_for_loss, target = _align_logits_and_labels(logits, labels)

    loss = F.cross_entropy(
        logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
        target.reshape(-1),
        ignore_index=pad_id,
    )

    metrics: Dict[str, float] = {}
    if compute_metrics:
        accuracy = _compute_span_accuracy(
            logits_for_loss, target, mask_positions, mask_lengths
        )
        if accuracy is not None:
            metrics["mask_accuracy"] = accuracy

    return loss, metrics