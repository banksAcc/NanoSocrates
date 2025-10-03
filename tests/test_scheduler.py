import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from training.scheduler import create_scheduler


def test_scheduler_respects_min_ratio_floor():
    total_steps = 100
    scheduler = create_scheduler(
        "cosine", warmup_ratio=0.1, warmup_steps=None, total_steps=total_steps, min_lr_ratio=0.05
    )

    assert math.isclose(scheduler(0), 0.05, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(scheduler(total_steps), 0.05, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(scheduler(total_steps + 20), 0.05, rel_tol=0.0, abs_tol=1e-9)


def test_scheduler_warmup_backward_compatibility():
    total_steps = 50
    ratio_scheduler = create_scheduler(
        "linear", warmup_ratio=0.2, warmup_steps=None, total_steps=total_steps, min_lr_ratio=0.0
    )
    steps_scheduler = create_scheduler(
        "linear", warmup_ratio=None, warmup_steps=10, total_steps=total_steps, min_lr_ratio=0.0
    )

    for step in (0, 5, 10, 25, 50):
        assert math.isclose(ratio_scheduler(step), steps_scheduler(step), rel_tol=1e-6, abs_tol=1e-9)
