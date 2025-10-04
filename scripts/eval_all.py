"""Entry point da riga di comando per la valutazione batch.

Lo script carica la configurazione YAML, esegue ``evaluate_from_config`` e si
occupa di inizializzare/chiudere correttamente la run W&B. La logica è
condivisa con ``src.cli`` tramite ``src.utils.wandb_logging``.
"""
from __future__ import annotations

import argparse
import json

from src.eval.evaluate import evaluate_from_config
from src.utils.config import load_yaml
from src.utils.wandb_logging import _flatten_for_logging, _maybe_init_wandb


def _log_metrics(wandb_run, result):
    if wandb_run is None:
        return
    payload = {
        "summary": result.get("summary", {}),
        "metrics": result.get("metrics", {}),
    }
    flat = _flatten_for_logging(payload)
    if flat:
        wandb_run.log(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Valuta tutti i task dal config")
    parser.add_argument("--config", "--cfg", dest="cfg", required=True, help="File YAML")
    args = parser.parse_args()

    cfg = load_yaml(args.cfg)

    wandb_run, _wandb_module = _maybe_init_wandb(cfg.get("wandb"), run_config=cfg)
    try:
        result = evaluate_from_config(cfg)
        _log_metrics(wandb_run, result)
        output_path = cfg.get("output_json")
        if output_path:
            print(f"[eval] metrics saved to {output_path}")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
