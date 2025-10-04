"""Helper utilities for optional Weights & Biases logging."""
from __future__ import annotations

from typing import Dict, Tuple


def maybe_init_wandb(cfg: Dict[str, object]) -> Tuple[object, object]:
    """Initialise wandb run if configured, returning the run and module.

    Parameters
    ----------
    cfg: dict
        Configuration dictionary that may contain a ``"wandb"`` section.

    Returns
    -------
    tuple
        ``(run, module)`` when logging is enabled and the library is available,
        otherwise ``(None, None)``.
    """

    wandb_cfg = cfg.get("wandb") or {}
    mode = str(wandb_cfg.get("mode", "disabled") or "disabled")
    if mode.lower() == "disabled":
        return None, None

    try:
        import wandb as _wandb
    except ImportError as exc:  # pragma: no cover - depends on environment
        print(f"[wandb] libreria non disponibile ({exc}); logging disattivato.")
        return None, None

    init_kwargs = {
        "project": wandb_cfg.get("project"),
        "entity": wandb_cfg.get("entity"),
        "name": wandb_cfg.get("run_name") or wandb_cfg.get("name"),
        "tags": wandb_cfg.get("tags"),
        "mode": mode,
        "config": cfg,
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    if not init_kwargs.get("tags"):
        init_kwargs.pop("tags", None)

    try:
        run = _wandb.init(**init_kwargs)
        return run, _wandb
    except Exception as exc:  # pragma: no cover - rare fallback
        print(f"[wandb] init fallita ({exc}); logging disabilitato.")
        return None, None


def flatten_eval_metrics(report: Dict[str, object]) -> Dict[str, float]:
    """Flatten the nested evaluation report for logging on wandb."""

    flat: Dict[str, float] = {}
    splits = report.get("splits", {})
    for split_name, split_payload in splits.items():
        if not isinstance(split_payload, dict):
            continue
        if "avg_loss" in split_payload:
            flat[f"{split_name}/avg_loss"] = float(split_payload["avg_loss"])
        tasks = split_payload.get("tasks", {})
        for task_name, task_payload in tasks.items():
            if not isinstance(task_payload, dict):
                continue
            if "loss" in task_payload:
                flat[f"{split_name}/{task_name}/loss"] = float(task_payload["loss"])
            metrics_by_task = task_payload.get("metrics", {})
            for inner_task, metrics in metrics_by_task.items():
                if not isinstance(metrics, dict):
                    continue
                base = f"{split_name}/{task_name}/{inner_task}"
                for key, value in metrics.items():
                    if key == "samples":
                        flat[f"{base}/samples"] = float(value)
                    else:
                        flat[f"{base}/{key}"] = float(value)
    return flat
