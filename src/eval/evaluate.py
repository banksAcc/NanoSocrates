"""Routine di valutazione per NanoSocrates.

Il modulo fornisce ``evaluate_from_config`` che legge una configurazione YAML
(e.g. usata da CLI e script) e restituisce un dizionario con le metriche per
ciascun task insieme a statistiche aggregate. La funzione supporta dataset
semplici dove le predizioni e i target sono salvati in JSONL con chiavi
``prediction`` e ``target`` (personalizzabili tramite config).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass
class TaskConfig:
    """Configurazione di un singolo task di valutazione."""

    name: str
    predictions: Path
    references: Path
    prediction_field: str = "prediction"
    reference_field: str = "target"
    metric: str = "accuracy"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TaskConfig":
        missing = [k for k in ("name", "predictions", "references") if k not in data]
        if missing:
            raise ValueError(f"Missing keys for evaluation task: {', '.join(missing)}")
        return cls(
            name=str(data["name"]),
            predictions=Path(data["predictions"]),
            references=Path(data["references"]),
            prediction_field=str(data.get("prediction_field", "prediction")),
            reference_field=str(data.get("reference_field", "target")),
            metric=str(data.get("metric", "accuracy")),
        )


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _compute_accuracy(
    predictions: Iterable[Any],
    references: Iterable[Any],
) -> tuple[int, int, float]:
    total = 0
    correct = 0
    for pred, ref in zip(predictions, references):
        total += 1
        if pred == ref:
            correct += 1
    accuracy = (correct / total) if total else 0.0
    return correct, total, accuracy


SUPPORTED_METRICS = {
    "accuracy": _compute_accuracy,
    "exact_match": _compute_accuracy,
}


def evaluate_from_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Esegue la valutazione dei task descritti nel config e ritorna le metriche."""

    tasks_cfg = cfg.get("tasks") or []
    tasks = [TaskConfig.from_mapping(task_cfg) for task_cfg in tasks_cfg]
    if not tasks:
        raise ValueError("Evaluation config must define at least one task")

    output_path = cfg.get("output_json")
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_by_task: dict[str, dict[str, Any]] = {}
    total_examples = 0
    total_correct = 0

    for task in tasks:
        records_pred = _load_jsonl(task.predictions)
        records_ref = _load_jsonl(task.references)
        if len(records_pred) != len(records_ref):
            raise ValueError(
                f"Mismatched number of records for task '{task.name}':"
                f" {len(records_pred)} predictions vs {len(records_ref)} references"
            )
        metric_fn = SUPPORTED_METRICS.get(task.metric.lower())
        if metric_fn is None:
            raise ValueError(
                f"Unsupported metric '{task.metric}' for task '{task.name}'"
            )
        preds = [record.get(task.prediction_field) for record in records_pred]
        refs = [record.get(task.reference_field) for record in records_ref]
        correct, total, accuracy = metric_fn(preds, refs)
        metrics_by_task[task.name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        total_examples += total
        total_correct += correct

    summary = {
        "tasks": len(tasks),
        "total_examples": total_examples,
        "micro_accuracy": (total_correct / total_examples) if total_examples else 0.0,
        "macro_accuracy": (
            sum(task_metrics["accuracy"] for task_metrics in metrics_by_task.values())
            / len(metrics_by_task)
            if metrics_by_task
            else 0.0
        ),
    }

    result = {
        "metrics": metrics_by_task,
        "summary": summary,
    }

    if output_path:
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    return result


__all__ = ["evaluate_from_config"]
