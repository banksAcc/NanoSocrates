"""Script CLI per eseguire la valutazione completa."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from src.eval.evaluate import evaluate_from_config
from src.utils.config import add_common_overrides, apply_overrides, apply_toy_paths, load_yaml
from src.utils.wandb_utils import flatten_eval_metrics, maybe_init_wandb


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
    if getattr(args, "toy", False):
        cfg = apply_toy_paths(cfg)
        print("[toy] dataset paths → data/processed/toy")
    cfg = apply_overrides(cfg, args.override)

    output_path = args.output or cfg.get("output_json")
    if output_path is None:
        output_path = Path(args.cfg).with_suffix(".report.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wandb_run, wandb_module = maybe_init_wandb(cfg)
    try:
        report = evaluate_from_config(cfg)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report salvato in {output_path.resolve()}")
        _print_report(report)

        if wandb_run is not None:
            flat_metrics = flatten_eval_metrics(report)
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
