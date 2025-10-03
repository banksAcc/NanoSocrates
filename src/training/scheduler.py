"""Learning rate scheduler utilities."""

from __future__ import annotations

import math
from typing import Callable


def _clamp_step(step: int, total_steps: int) -> int:
    """Clamp ``step`` within ``[0, total_steps]`` to keep schedules stable."""

    return max(0, min(step, total_steps))


def _cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    step = _clamp_step(step, total_steps)
    if warmup_steps > 0 and step <= warmup_steps:
        return step / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _linear_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    step = _clamp_step(step, total_steps)
    if warmup_steps > 0 and step <= warmup_steps:
        return step / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return 0.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return max(0.0, 1.0 - progress)


SCHEDULES: dict[str, Callable[[int, int, int], float]] = {
    "cosine": _cosine_with_warmup,
    "linear": _linear_with_warmup,
}


def create_scheduler(
    name: str,
    *,
    warmup_ratio: float | None,
    warmup_steps: int | None,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    """Return a scheduler closure with warmup support.

    ``warmup_steps`` keeps backward compatibility with previous configurations,
    while ``warmup_ratio`` is the preferred modern API.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")

    schedule_fn = SCHEDULES.get(name.lower())
    if schedule_fn is None:
        raise ValueError(f"Unsupported scheduler '{name}'. Available: {sorted(SCHEDULES)}")

    min_lr_ratio = float(min_lr_ratio)
    if not math.isfinite(min_lr_ratio):
        raise ValueError("min_lr_ratio must be finite")
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    if warmup_steps is not None:
        warmup_steps = max(0, int(warmup_steps))
        if warmup_ratio is None:
            warmup_ratio = warmup_steps / total_steps
    else:
        warmup_ratio = 0.0 if warmup_ratio is None else max(0.0, warmup_ratio)
        warmup_steps = int(total_steps * warmup_ratio)

    warmup_steps = min(warmup_steps, total_steps)

    def scheduler(step: int) -> float:
        scale = schedule_fn(step, warmup_steps, total_steps)
        return max(min_lr_ratio, scale)

    return scheduler


def cosine_with_warmup(step: int, warmup: int, max_steps: int) -> float:
    """Backward compatible cosine schedule returning the raw scale factor."""

    return _cosine_with_warmup(step, warmup, max_steps)
