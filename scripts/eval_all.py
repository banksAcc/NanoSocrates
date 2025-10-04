"""Script CLI per eseguire la valutazione completa."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from src.eval.evaluate import evaluate_from_config
from src.utils.config import add_common_overrides, apply_overrides, load_yaml


def _maybe_init_wandb(cfg) -> Tuple[object, object]:
    wandb_cfg = cfg.get("wandb") or {}
    mode = str(wandb_cfg.get("mode", "disabled") or "disabled")
    if mode.lower() == "disabled":
        return None, None
    try:
        import wandb as _wandb
    except ImportError as exc:  # pragma: no cover - dipende dall'ambiente
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
    except Exception as exc:  # pragma: no cover - fallback raro
        print(f"[wandb] init fallita ({exc}); logging disabilitato.")
        return None, None


def _flatten_for_logging(report: Dict[str, object]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    splits = report.get("splits", {})
    for split_name, split_payload in splits.items():
        if isinstance(split_payload, dict):
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


def _print_report(report: Dict[str, object]):
    print("=== Evaluation Report ===")
    for split_name, split_payload in report.get("splits", {}).items():
        print(f"\n[{split_name}]")
        if not split_payload.get("tasks"):
            print("  (nessun task)")
            continue
        if "avg_loss" in split_payload:
            print(f"  avg_loss: {split_payload['avg_loss']:.4f}")
        for task_name, task_payload in split_payload["tasks"].items():
            loss = task_payload.get("loss")
            samples = task_payload.get("num_samples", 0)
            print(f"  - {task_name} (n={samples}) loss={loss:.4f}" if loss is not None else f"  - {task_name} (n={samples})")
            metrics = task_payload.get("metrics", {})
            for inner_task, vals in metrics.items():
                numbers = [
                    f"{k}={vals[k]:.2f}" for k in sorted(vals) if k != "samples"
                ]
                if numbers:
                    print(f"      · {inner_task}: {', '.join(numbers)}")
                print(f"        samples={vals.get('samples', 0)}")


def main():
    ap = argparse.ArgumentParser(description="Valutazione multi-task")
    add_common_overrides(ap)
    ap.add_argument("--output", help="File JSON per salvare il report", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    cfg = apply_overrides(cfg, args.override)

    output_path = args.output or cfg.get("output_json")
    if output_path is None:
        output_path = Path(args.cfg).with_suffix(".report.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wandb_run, wandb_module = _maybe_init_wandb(cfg)
    try:
        report = evaluate_from_config(cfg)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report salvato in {output_path.resolve()}")
        _print_report(report)

        if wandb_run is not None:
            flat_metrics = _flatten_for_logging(report)
            try:
                wandb_run.log(flat_metrics)
            except Exception as exc:  # pragma: no cover - logging best effort
                print(f"[wandb] log fallito: {exc}")
    finally:
        if wandb_run is not None and wandb_module is not None:
            try:
                wandb_module.finish()
            except Exception as exc:  # pragma: no cover
                print(f"[wandb] finish fallita: {exc}")


if __name__ == "__main__":
    main()
