import pathlib
import sys
from collections import Counter
from itertools import islice

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.dataloaders import MultiTaskDataset, MultiTaskSampler, Seq2SeqExample


def _make_example(task: str, index: int) -> Seq2SeqExample:
    token = index + 1
    return Seq2SeqExample(
        input_text=f"input-{task}-{index}",
        target_text=f"target-{task}-{index}",
        task=task,
        film=None,
        input_ids=[token],
        label_ids=[token],
    )


def _build_dataset(task_counts: dict[str, int]) -> MultiTaskDataset:
    items = []
    for task, count in task_counts.items():
        items.extend(_make_example(task, idx) for idx in range(count))
    return MultiTaskDataset(items)


def test_multitask_sampler_respects_proportional_allocation():
    dataset = _build_dataset({"a": 50, "b": 50})
    sampler = MultiTaskSampler(dataset, batch_size=5, ratios={"a": 0.6, "b": 0.4})

    expected_allocation = sampler._compute_batch_allocation()
    batches = list(islice(iter(sampler), 3))

    assert all(len(batch) == 5 for batch in batches)
    for batch in batches:
        task_counts = Counter(dataset.items[idx].task for idx in batch)
        assert task_counts == expected_allocation


def test_multitask_sampler_handles_more_tasks_than_batch_size():
    dataset = _build_dataset({"t0": 20, "t1": 20, "t2": 20, "t3": 20, "t4": 20})
    ratios = {"t0": 0.4, "t1": 0.2, "t2": 0.2, "t3": 0.1, "t4": 0.1}
    sampler = MultiTaskSampler(dataset, batch_size=3, ratios=ratios)

    expected_allocation = sampler._compute_batch_allocation()
    assert sum(expected_allocation.values()) == 3

    batch = next(iter(sampler))
    assert len(batch) == 3

    task_counts = Counter(dataset.items[idx].task for idx in batch)
    # All tasks expected to appear in the batch must respect the computed quotas
    for task, expected in expected_allocation.items():
        if expected:
            assert task_counts[task] == expected
        else:
            assert task not in task_counts or task_counts[task] == 0
