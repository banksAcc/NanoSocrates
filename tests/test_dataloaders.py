from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from training.dataloaders import _infer_task


def test_infer_task_rdfcomp1_from_mask_marker():
    assert _infer_task("rdfcomp1.train.jsonl", "<MASK> some prompt") == "rdfcomp1"