import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import cmd_evaluate


class DummyConfig:
    def __init__(self):
        self.data = {}

    def update(self, payload, allow_val_change=False):  # noqa: ARG002 - compat
        self.data.update(payload)


class DummyRun:
    def __init__(self):
        self.logged = []
        self.finished = False
        self.config = DummyConfig()

    def log(self, payload, step=None):  # noqa: D401 - compat con wandb
        self.logged.append((payload, step))

    def finish(self):
        self.finished = True


class DummyWandbModule:
    def __init__(self):
        self.init_kwargs = []
        self.runs = []

    def init(self, **kwargs):
        run = DummyRun()
        self.init_kwargs.append(kwargs)
        self.runs.append(run)
        return run


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_cmd_evaluate_logs_to_wandb(monkeypatch, tmp_path):
    predictions = tmp_path / "preds.jsonl"
    references = tmp_path / "refs.jsonl"
    metrics_path = tmp_path / "metrics.json"

    _write_jsonl(predictions, [{"prediction": "a"}, {"prediction": "b"}])
    _write_jsonl(references, [{"target": "a"}, {"target": "c"}])

    cfg = {
        "tasks": [
            {
                "name": "task_a",
                "predictions": str(predictions),
                "references": str(references),
            }
        ],
        "output_json": str(metrics_path),
        "wandb": {
            "project": "proj",
            "entity": "ent",
            "run_name": "test-run",
            "mode": "online",
        },
    }

    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    dummy_wandb = DummyWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)

    args = type("Args", (), {"cfg": str(cfg_path), "override": []})()
    cmd_evaluate(args)

    assert dummy_wandb.init_kwargs
    assert metrics_path.exists()

    run = dummy_wandb.runs[0]
    assert run.finished is True
    assert run.logged, "expected wandb.log to be called"

    payload = run.logged[0][0]
    assert "metrics/task_a/accuracy" in payload

    result_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert result_payload["metrics"]["task_a"]["accuracy"] == 0.5
