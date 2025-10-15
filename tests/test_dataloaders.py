"""Unit tests for the multi-task dataloader utilities."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.dataloaders import (
    MultiTaskDataset,
    Seq2SeqExample,
    create_multitask_dataloader,
)


class _DummyTokenizer:
    pad_id = 0


@pytest.fixture()
def toy_dataset() -> MultiTaskDataset:
    examples = [
        Seq2SeqExample(
            input_text="input",
            target_text="target",
            task="text2rdf",
            film=None,
            input_ids=[1, 2, 3],
            label_ids=[4, 5, 6, 7],
        )
        for _ in range(3)
    ]
    return MultiTaskDataset(examples)


def test_create_multitask_dataloader_disables_drop_last_for_small_dataset(caplog: pytest.LogCaptureFixture, toy_dataset: MultiTaskDataset) -> None:
    """When the dataset has less examples than the batch size we still emit batches."""

    caplog.set_level(logging.WARNING)
    loader = create_multitask_dataloader(
        toy_dataset,
        tokenizer=_DummyTokenizer(),
        batch_size=8,
        shuffle=True,
    )

    batch = next(iter(loader))

    # The sampler should have resampled items to keep the requested batch size.
    assert batch["input_ids"].shape[0] == 8
    # And a warning should explain the drop_last adjustment so users understand the log noise.
    assert any("disattivo drop_last" in record.getMessage() for record in caplog.records)
