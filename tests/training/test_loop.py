import math
import pathlib
import sys

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.loop import TrainingLoop


class _CountingSGD(torch.optim.SGD):
    """SGD optimizer that tracks how often ``step`` is called."""

    def __init__(self, params, lr=0.1):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):  # type: ignore[override]
        self.step_calls += 1
        return super().step(closure)


class _DummyDataset(Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return {"input_ids": torch.tensor([float(idx)], dtype=torch.float32)}


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids):  # type: ignore[override]
        prediction = self.weight
        target = input_ids.mean()
        loss = (prediction - target).pow(2)
        return {"loss": loss, "metrics": None}


def test_optimizer_steps_flush_pending_gradients():
    grad_accum_steps = 3
    dataset_length = 5  # Leaves pending gradients after the last full accumulation

    train_dataset = _DummyDataset(dataset_length)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(_DummyDataset(1), batch_size=1)

    model = _DummyModel()
    optimizer = _CountingSGD(model.parameters(), lr=0.1)

    loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=None,
        device="cpu",
        use_amp=False,
        grad_accum_steps=grad_accum_steps,
        log_every_n_steps=10,
    )

    loop._train_epoch(epoch=1)

    expected_steps = math.ceil(dataset_length / grad_accum_steps)
    assert (
        optimizer.step_calls == expected_steps
    ), f"Expected {expected_steps} optimizer steps, got {optimizer.step_calls}"
