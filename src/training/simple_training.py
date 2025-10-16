"""Minimal training utilities for the NanoSocrates seq2seq model."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from src.model.losses import _align_logits_and_labels
from src.training.dataloaders import PadCollator


def _pad_id_from_tokenizer(tokenizer: object) -> int:
    pad = getattr(tokenizer, "pad_id", None)
    if pad is not None:
        return int(pad)
    lookup = getattr(tokenizer, "token_to_id", None)
    if callable(lookup):
        token_id = lookup("<pad>")
        if token_id is None:
            raise ValueError("Tokenizer privo di <pad>: rigenera il vocabolario includendo <pad>.")
        return int(token_id)
    raise ValueError("Tokenizer incompatibile: impossibile recuperare l'id di <pad>.")


def build_dataloader(
    dataset: Dataset,
    tokenizer: object,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Create a basic DataLoader that only pads variable length sequences."""

    pad_id = _pad_id_from_tokenizer(tokenizer)
    collate = PadCollator(pad_id=pad_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )


def _move_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = None,
    max_batches: int | None = None,
) -> Dict[str, float]:
    """Run one optimisation epoch returning averaged loss/accuracy.

    Args:
        max_batches: when provided, limits the number of optimisation steps for
            the epoch. This is handy for quick sanity checks/overfit tests.
    """

    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    total_exact = 0.0
    total_batches = 0
    pending = 0

    for step, batch in enumerate(loader, start=1):
        batch = _move_to_device(batch, device)
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs["loss"]
        if loss is None:
            raise ValueError("Il modello deve restituire una loss per addestrare.")

        total_loss += float(loss.detach().item())
        total_exact += float(outputs.get("metrics", {}).get("exact_match", 0.0))
        total_batches += 1

        (loss / max(1, grad_accum_steps)).backward()
        pending += 1

        if pending == grad_accum_steps:
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pending = 0

        if max_batches is not None and step >= max_batches:
            break

    if pending:
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    denom = max(1, total_batches)
    return {
        "loss": total_loss / denom,
        "exact_match": total_exact / denom,
        "batches": float(total_batches),
    }


def _per_example_exact(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    pad_id: int,
) -> torch.Tensor:
    aligned_logits, target = _align_logits_and_labels(logits, labels)
    ignore_index = -100 if (target == -100).any() else pad_id
    target_for_metrics = target
    if ignore_index == -100 and pad_id != -100:
        target_for_metrics = target.masked_fill(target == pad_id, -100)

    predictions = aligned_logits.argmax(dim=-1)
    valid_mask = target_for_metrics != ignore_index
    comparison = (predictions == target) | ~valid_mask
    return comparison.all(dim=1).float()


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    pad_id: int,
    max_batches: int | None = None,
) -> Dict[str, object]:
    """Compute mean loss/exact-match and per-task exact match when available."""

    model.eval()
    total_loss = 0.0
    total_exact = 0.0
    total_batches = 0
    per_task: Dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = _move_to_device(batch, device)
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.get("loss")
            if loss is None:
                raise ValueError("Il modello deve restituire una loss in valutazione.")

            total_loss += float(loss.item())
            total_exact += float(outputs.get("metrics", {}).get("exact_match", 0.0))
            total_batches += 1

            logits = outputs.get("logits")
            tasks: Iterable[str] = batch.get("tasks") or []
            if logits is not None and tasks:
                per_example = _per_example_exact(logits, batch["labels"], pad_id=pad_id)
                for index, task in enumerate(tasks):
                    per_task[task].append(float(per_example[index].item()))

            if max_batches is not None and step >= max_batches:
                break

    denom = max(1, total_batches)
    summary: Dict[str, object] = {
        "loss": total_loss / denom,
        "exact_match": total_exact / denom,
        "batches": float(total_batches),
    }

    if per_task:
        summary["tasks"] = {
            task: {
                "exact_match": float(sum(values) / max(1, len(values))),
                "samples": len(values),
            }
            for task, values in per_task.items()
        }

    return summary
