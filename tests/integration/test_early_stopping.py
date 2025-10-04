import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from training.loop import train_loop  # noqa: E402


class DummySeq2SeqDataset(Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length

    def __getitem__(self, idx):  # pragma: no cover - trivial
        value = idx % 2
        return {
            "input_ids": torch.tensor([idx], dtype=torch.long),
            "attention_mask": torch.tensor([1], dtype=torch.long),
            "labels": torch.tensor([value], dtype=torch.long),
        }


class DummyModel(torch.nn.Module):
    def __init__(self, val_losses):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(0.0))
        self.train_loss = 1.0
        self.val_losses = list(val_losses)
        self._eval_epoch = 0
        self._val_batches_per_epoch = 1
        self._val_batch_calls = 0
        self.completed_eval_epochs = 0

    def train(self, mode: bool = True):  # pragma: no cover - thin override
        super().train(mode)
        return self

    def eval(self):  # pragma: no cover - thin override
        self._val_batch_calls = 0
        return super().eval()

    def forward(self, input_ids, attention_mask, labels, **extra):  # pragma: no cover - simple
        del input_ids, attention_mask, extra
        device = labels.device
        base_loss = torch.tensor(self.train_loss, device=device, dtype=torch.float32)
        if self.training:
            return {"loss": self.param * 0 + base_loss}

        idx = min(self._eval_epoch, len(self.val_losses) - 1)
        val_loss = torch.tensor(self.val_losses[idx], device=device, dtype=torch.float32)
        output = {"loss": self.param * 0 + val_loss}
        self._val_batch_calls += 1
        if self._val_batch_calls >= self._val_batches_per_epoch:
            self._eval_epoch = min(self._eval_epoch + 1, len(self.val_losses) - 1)
            self.completed_eval_epochs += 1
            self._val_batch_calls = 0
        return output


def _run_training(tmp_path, val_losses, patience, min_delta):
    model = DummyModel(val_losses)
    train_dl = DataLoader(DummySeq2SeqDataset(4), batch_size=2, shuffle=False)
    val_dl = DataLoader(DummySeq2SeqDataset(1), batch_size=1, shuffle=False)
    model._val_batches_per_epoch = len(val_dl)

    save_dir = tmp_path / "checkpoints"
    cfg = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 2,
        "num_epochs": 10,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 0,
        "save_dir": str(save_dir),
        "wandb": {"mode": "disabled"},
        "early_stopping": {"patience": patience, "min_delta": min_delta},
    }
    stats = train_loop(
        model,
        train_dl,
        val_dl,
        cfg,
        device="cpu",
        pad_id=0,
        steps_per_epoch=len(train_dl),
        max_train_steps=0,
        wandb_run=None,
        wandb_module=None,
    )
    return stats, model, save_dir


def test_train_loop_stops_after_patience_is_reached(tmp_path):
    stats, model, save_dir = _run_training(
        tmp_path,
        val_losses=[1.0, 0.95, 0.95, 0.95, 0.5],
        patience=2,
        min_delta=0.0,
    )

    assert model.completed_eval_epochs == 4
    assert stats["best_epoch"] == 2
    assert stats["best_val"] == pytest.approx(0.95, rel=1e-6)

    ckpt_files = sorted(save_dir.glob("epoch*.pt"))
    assert len(ckpt_files) == 4
    assert not (save_dir / "epoch005.pt").exists()


def test_min_delta_requires_significant_improvement(tmp_path):
    stats, model, save_dir = _run_training(
        tmp_path,
        val_losses=[1.0, 0.995, 0.995, 0.5],
        patience=2,
        min_delta=0.01,
    )

    assert model.completed_eval_epochs == 3
    assert stats["best_epoch"] == 1
    assert stats["best_val"] == pytest.approx(1.0, rel=1e-6)

    ckpt_files = sorted(save_dir.glob("epoch*.pt"))
    assert len(ckpt_files) == 3
    assert not (save_dir / "epoch004.pt").exists()
