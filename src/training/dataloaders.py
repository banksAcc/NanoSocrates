"""Utilities to load JSONL datasets and build multi-task dataloaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.utils.io import read_jsonl

if TYPE_CHECKING:
    from tokenizers import Tokenizer


_TASK_ALIASES: Dict[str, str] = {
    "text2rdf": "text2rdf",
    "text_to_rdf": "text2rdf",
    "rdf2text": "rdf2text",
    "rdf_to_text": "rdf2text",
    "rdfcomp1": "rdfcomp1",
    "rdf_comp1": "rdfcomp1",
    "rdfcompletion_msk": "rdfcomp1",
    "rdfcompletion_mask": "rdfcomp1",
    "rdfcomp2": "rdfcomp2",
    "rdf_comp2": "rdfcomp2",
    "rdfcompletion_cont": "rdfcomp2",
    "rdfcompletion_continue": "rdfcomp2",
}


def _normalise_task_name(value: str | None, fallback: str) -> str:
    """Normalises raw task identifiers to the internal lowercase format."""

    if not value:
        return fallback
    key = value.lower()
    if key in _TASK_ALIASES:
        return _TASK_ALIASES[key]
    if key.title() in {"Text2RDF", "RDF2Text"}:
        return key.lower()
    return fallback


def _infer_task_from_path(path: str) -> str:
    """Guesses the task name from the filename when not explicitly provided."""

    name = Path(path).name.lower()
    for alias, canonical in _TASK_ALIASES.items():
        if alias in name:
            return canonical
    return "generic"


def _get_pad_id(tokenizer: Any) -> int:
    """Returns the padding token id from a tokenizer or wrapper."""

    if hasattr(tokenizer, "pad_id"):
        return int(getattr(tokenizer, "pad_id"))
    pad = tokenizer.token_to_id("<pad>")
    if pad is None:
        raise ValueError("Tokenizer privo di <pad>: rigenera il BPE includendo <pad>.")
    return int(pad)


def _encode(tokenizer: Any, text: str) -> List[int]:
    """Encodes text into token ids handling both HF Tokenizer and wrappers."""

    encoded = tokenizer.encode(text)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if isinstance(encoded, (list, tuple)):
        return [int(tok) for tok in encoded]
    return list(encoded)


@dataclass
class Seq2SeqExample:
    """Container for a pre-tokenised sequence-to-sequence example."""

    input_text: str
    target_text: str
    task: str
    film: str | None
    input_ids: List[int]
    label_ids: List[int]
    mask_positions: List[int] | None = None
    mask_lengths: List[int] | None = None


def _truncate(sequence: List[int], max_len: int) -> List[int]:
    return sequence[:max_len]


def _normalise_span_payload(spans: Any) -> List[tuple[int, int]]:
    """Converts heterogeneous span annotations into (start, length) tuples."""

    normalised: List[tuple[int, int]] = []

    if not spans:
        return normalised

    if isinstance(spans, Mapping):
        positions = spans.get("positions") or spans.get("starts")
        lengths = spans.get("lengths") or spans.get("span_lengths")
        if positions and lengths:
            for pos, length in zip(positions, lengths):
                try:
                    normalised.append((int(pos), max(0, int(length))))
                except (TypeError, ValueError):
                    continue
            return normalised

    for span in spans if isinstance(spans, Iterable) else []:
        if isinstance(span, Mapping):
            start = span.get("start") or span.get("position") or span.get("idx")
            end = span.get("end")
            length = span.get("length") or span.get("span_length")
            if start is None:
                continue
            try:
                start_i = int(start)
            except (TypeError, ValueError):
                continue
            if length is None and end is not None:
                try:
                    length = int(end) - start_i
                except (TypeError, ValueError):
                    length = None
            try:
                length_i = int(length) if length is not None else 0
            except (TypeError, ValueError):
                continue
            normalised.append((start_i, max(0, length_i)))
        elif isinstance(span, (list, tuple)) and span:
            try:
                start_i = int(span[0])
            except (TypeError, ValueError):
                continue
            length_i = None
            if len(span) >= 2:
                try:
                    length_i = int(span[1])
                except (TypeError, ValueError):
                    length_i = None
            if length_i is None and len(span) >= 3:
                try:
                    length_i = int(span[2]) - start_i
                except (TypeError, ValueError):
                    length_i = None
            normalised.append((start_i, max(0, length_i or 0)))

    return normalised


def _iter_examples(
    path: str,
    tokenizer: Any,
    *,
    max_len: int,
    task_hint: str | None = None,
    enable_entity_spans: bool = False,
) -> Iterator[Seq2SeqExample]:
    """Yields tokenised examples from a JSONL file."""

    inferred_task = _infer_task_from_path(path)
    for record in read_jsonl(path):
        source = str(record.get("input") or record.get("source") or record.get("text") or "")
        target = str(record.get("target") or record.get("output") or record.get("label") or "")
        if not source or not target:
            continue
        film = record.get("film")
        task_name = _normalise_task_name(record.get("task"), inferred_task)
        task_name = _normalise_task_name(task_hint, task_name)

        input_ids = _truncate(_encode(tokenizer, source), max_len)
        label_ids = _truncate(_encode(tokenizer, target), max_len)

        mask_positions: List[int] | None = None
        mask_lengths: List[int] | None = None
        if enable_entity_spans:
            raw_spans = (
                record.get("entity_spans")
                or record.get("mask_spans")
                or record.get("mask_positions")
            )
            spans = _normalise_span_payload(raw_spans)
            if not spans and "<mask>" in source.lower():
                spans = [(0, len(label_ids))] if label_ids else []

            if spans:
                valid_positions: List[int] = []
                valid_lengths: List[int] = []
                seq_len = len(label_ids)
                for start, length in spans:
                    if length <= 0:
                        continue
                    if start < 0:
                        start = 0
                    if start >= seq_len:
                        continue
                    end = min(seq_len, start + length)
                    length = max(0, end - start)
                    if length == 0:
                        continue
                    valid_positions.append(int(start))
                    valid_lengths.append(int(length))
                if valid_positions:
                    mask_positions = valid_positions
                    mask_lengths = valid_lengths

        yield Seq2SeqExample(
            input_text=source,
            target_text=target,
            task=task_name,
            film=film,
            input_ids=input_ids,
            label_ids=label_ids,
            mask_positions=mask_positions,
            mask_lengths=mask_lengths,
        )


class MultiTaskDataset(Dataset):
    """Simple dataset that stores pre-tokenised seq2seq examples."""

    def __init__(self, items: Sequence[Seq2SeqExample]):
        self.items: List[Seq2SeqExample] = list(items)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.items[index]
        payload: Dict[str, Any] = {
            "input_ids": example.input_ids,
            "labels": example.label_ids,
            "task": example.task,
            "raw_input": example.input_text,
            "raw_target": example.target_text,
        }
        if example.mask_positions is not None and example.mask_lengths is not None:
            payload["mask_positions"] = example.mask_positions
            payload["mask_lengths"] = example.mask_lengths
        if example.film is not None:
            payload["film"] = example.film
        return payload

    def select_first(self, n_examples: int) -> "MultiTaskDataset":
        """Returns a shallow copy containing only the first *n_examples*."""

        return MultiTaskDataset(self.items[: max(1, n_examples)])

    def task_counts(self) -> Counter:
        """Returns the number of examples per task."""

        return Counter(example.task for example in self.items)

    def task_fractions(self) -> Dict[str, float]:
        """Returns normalised task ratios useful for the sampler."""

        counts = self.task_counts()
        total = sum(counts.values()) or 1
        return {task: count / total for task, count in counts.items()}


class JsonlSeq2Seq(MultiTaskDataset):
    """Dataset specialised for loading a single JSONL file."""

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        *,
        max_len: int,
        task: str | None = None,
        enable_entity_spans: bool = False,
    ) -> None:
        self.path = str(path)
        self.max_len = int(max_len)
        self.pad_id = _get_pad_id(tokenizer)
        items = list(
            _iter_examples(
                self.path,
                tokenizer,
                max_len=self.max_len,
                task_hint=task,
                enable_entity_spans=enable_entity_spans,
            )
        )
        super().__init__(items)


class MultiTaskSampler(Sampler[List[int]]):
    """Sampler that draws balanced batches according to task ratios."""

    def __init__(
        self,
        dataset: MultiTaskDataset,
        batch_size: int,
        ratios: Dict[str, float],
        *,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.indices_by_task: Dict[str, List[int]] = defaultdict(list)
        for idx, example in enumerate(dataset.items):
            self.indices_by_task[example.task].append(idx)

        self.indices_by_task = {
            task: indices for task, indices in self.indices_by_task.items() if indices
        }
        if not self.indices_by_task:
            raise ValueError("Dataset privo di esempi: impossibile creare il sampler.")

        total_ratio = sum(ratios.get(task, 0.0) for task in self.indices_by_task)
        if total_ratio == 0.0:
            total_ratio = float(len(self.indices_by_task))
            ratios = {task: 1.0 for task in self.indices_by_task}

        self.task_ratios: Dict[str, float] = {
            task: float(ratios.get(task, 0.0)) / total_ratio or 1.0 / len(self.indices_by_task)
            for task in self.indices_by_task
        }

    def __iter__(self) -> Iterator[List[int]]:
        task_iters: Dict[str, Iterator[int]] = {}
        for task, indices in self.indices_by_task.items():
            random.shuffle(indices)
            task_iters[task] = iter(indices)

        num_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            num_batches += 1

        tasks = list(self.task_ratios.keys())
        for _ in range(num_batches):
            batch: List[int] = []
            for task_name, ratio in self.task_ratios.items():
                desired = max(1, int(round(self.batch_size * ratio)))
                for _ in range(desired):
                    iterator = task_iters.get(task_name)
                    if iterator is None:
                        continue
                    try:
                        batch.append(next(iterator))
                    except StopIteration:
                        random.shuffle(self.indices_by_task[task_name])
                        task_iters[task_name] = iter(self.indices_by_task[task_name])
                        batch.append(next(task_iters[task_name]))

            while len(batch) < self.batch_size:
                task_name = random.choice(tasks)
                iterator = task_iters.get(task_name)
                if iterator is None:
                    continue
                try:
                    batch.append(next(iterator))
                except StopIteration:
                    random.shuffle(self.indices_by_task[task_name])
                    task_iters[task_name] = iter(self.indices_by_task[task_name])
                    batch.append(next(task_iters[task_name]))

            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:  # pragma: no cover - straightforward math
        total = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            total += 1
        return total


def pad_collate(
    batch: Sequence[Dict[str, Any]],
    *,
    pad_id: int,
    label_pad_id: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Pads variable-length examples into a dense batch."""

    if label_pad_id is None:
        label_pad_id = pad_id

    def _pad_tensor(sequences: Sequence[Sequence[int]], value: int) -> torch.Tensor:
        tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=value)

    input_ids = _pad_tensor([ex["input_ids"] for ex in batch], pad_id)
    labels = _pad_tensor([ex["labels"] for ex in batch], label_pad_id)
    attention_mask = (input_ids != pad_id).long()

    collated: Dict[str, Any] = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

    if any("mask_positions" in ex for ex in batch):
        max_spans = max(len(ex.get("mask_positions", [])) for ex in batch)
        if max_spans:
            padded_pos = []
            padded_len = []
            for ex in batch:
                pos = ex.get("mask_positions", [])
                length = ex.get("mask_lengths", [])
                pad_pos = list(pos) + [0] * (max_spans - len(pos))
                pad_len = list(length) + [0] * (max_spans - len(length))
                padded_pos.append(torch.tensor(pad_pos, dtype=torch.long))
                padded_len.append(torch.tensor(pad_len, dtype=torch.long))
            collated["mask_positions"] = torch.stack(padded_pos)
            collated["mask_lengths"] = torch.stack(padded_len)

    collated["tasks"] = [ex.get("task", "") for ex in batch]
    collated["raw_input"] = [ex.get("raw_input") for ex in batch]
    collated["raw_target"] = [ex.get("raw_target") for ex in batch]
    collated["films"] = [ex.get("film") for ex in batch]
    return collated


class PadCollator:
    """Callable wrapper around :func:`pad_collate` that can be pickled."""

    def __init__(self, *, pad_id: int, label_pad_id: int | None = None) -> None:
        self.pad_id = int(pad_id)
        self.label_pad_id = label_pad_id if label_pad_id is None else int(label_pad_id)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return pad_collate(batch, pad_id=self.pad_id, label_pad_id=self.label_pad_id)


def create_multitask_dataloader(
    dataset: MultiTaskDataset,
    *,
    tokenizer: Any,
    batch_size: int,
    ratios: Dict[str, float] | None = None,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Creates a DataLoader configured for multi-task batches."""

    pad_id = _get_pad_id(tokenizer)
    collate = PadCollator(pad_id=pad_id)

    if shuffle:
        sampler = MultiTaskSampler(
            dataset,
            batch_size=batch_size,
            ratios=ratios or dataset.task_fractions(),
            drop_last=True,
        )
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=num_workers)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )
