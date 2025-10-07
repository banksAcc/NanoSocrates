"""Learning rate scheduler utilities.

This module provides functions to create learning rate schedulers with warmup
and decay, commonly used in training Transformers. The main entry point is
`create_scheduler`, which returns a callable compatible with `LambdaLR`.
"""

from __future__ import annotations

import math, torch
from typing import Callable

SCHEDULE_FN = Callable[[int, int, int], float]


def _clamp_step(step: int, total_steps: int) -> int:
    """Clamps a step to be within the valid range [0, total_steps]."""
    return max(0, min(step, total_steps))


def _cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    """Cosine decay schedule with linear warmup."""
    step = _clamp_step(step, total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    
    # After warmup, decay starts
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _linear_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear decay schedule with linear warmup."""
    step = _clamp_step(step, total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))

    # After warmup, decay starts
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 1.0 - progress


# Registry of available schedule functions
SCHEDULES: dict[str, SCHEDULE_FN] = {
    "cosine": _cosine_with_warmup,
    "linear": _linear_with_warmup,
}


def create_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    *,
    warmup_ratio: float = 0.0,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Creates a `LambdaLR` scheduler with warmup and decay.

    This factory function simplifies the creation of standard learning rate
    schedulers used in NLP.

    Args:
        name: The name of the schedule to use (e.g., 'cosine', 'linear').
        optimizer: The PyTorch optimizer to wrap.
        warmup_ratio: The fraction of `total_steps` to use for linear warmup.
        total_steps: The total number of training steps.
        min_lr_ratio: The minimum learning rate as a fraction of the initial
            learning rate. The LR will not decay below `initial_lr * min_lr_ratio`.

    Returns:
        A configured `LambdaLR` scheduler instance.

    Raises:
        ValueError: If `total_steps` is not positive or an unknown schedule
            name is provided.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be a positive integer.")

    schedule_fn = SCHEDULES.get(name.lower())
    if schedule_fn is None:
        raise ValueError(f"Unsupported scheduler '{name}'. Available: {list(SCHEDULES.keys())}")

    warmup_steps = int(total_steps * warmup_ratio)
    
    def lr_lambda(current_step: int) -> float:
        """Calculates the learning rate multiplier for a given step."""
        # Calculate the decay factor from the schedule function
        scale = schedule_fn(current_step, warmup_steps, total_steps)
        # Ensure the learning rate doesn't fall below the minimum ratio
        return max(min_lr_ratio, scale)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)