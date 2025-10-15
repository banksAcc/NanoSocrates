"""Utilities to load JSONL datasets and build multi-task dataloaders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import random
import math
from collections import Counter, OrderedDict, defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.utils.io import read_jsonl
from src.utils.special_tokens import (
    REQUIRED_SPECIAL_TOKENS,
    RDF_LIST_SEPARATOR_TOKEN,
    RDF_OBJECT_LIST_TOKEN,
)

LOGGER = logging.getLogger(__name__)

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
    """Guess the task name from the filename when not explicitly provided."""
    name = Path(path).name.lower()
    for alias, canonical in _TASK_ALIASES.items():
        if alias in name:
            return canonical
    return "generic"


def _get_pad_id(tokenizer: Any) -> int:
    """Return the padding token id from a tokenizer or wrapper."""
    if hasattr(tokenizer, "pad_id"):
        return int(getattr(tokenizer, "pad_id"))
    pad = tokenizer.token_to_id("<pad>")
    if pad is None:
        raise ValueError("Tokenizer privo di <pad>: rigenera il BPE includendo <pad>.")
    return int(pad)


def _encode(tokenizer: Any, text: str) -> List[int]:
    """Encode text into token ids handling both HF Tokenizer and wrappers."""
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if isinstance(encoded, (list, tuple)):
        return [int(tok) for tok in encoded]
    return list(encoded)


def _token_to_id(tokenizer: Any, token: str) -> int | None:
    """Safely resolve *token* to its integer id if present in the vocabulary."""

    lookup = getattr(tokenizer, "token_to_id", None)
    if callable(lookup):
        token_id = lookup(token)
        return int(token_id) if token_id is not None else None
    return None


@dataclass
class Seq2SeqExample:
    """Represent a pre-tokenised sequence-to-sequence example."""

    input_text: str
    target_text: str
    task: str
    film: str | None
    input_ids: List[int]
    label_ids: List[int]


def _truncate(sequence: List[int], max_len: int) -> List[int]:
    """Clip *sequence* to at most *max_len* tokens to respect model limits."""
    return sequence[:max_len]


def compact_rdf(text: str) -> str:
    """Group repeated triples by subject/predicate to shorten RDF sequences."""

    if not text or RDF_OBJECT_LIST_TOKEN in text:
        return text

    if "<SOT>" not in text or "<OBJ>" not in text:
        return text

    tokens = text.split()
    prefix_tokens: List[str] = []
    suffix_tokens: List[str] = []
    groups: "OrderedDict[tuple[str, str], List[str]]" = OrderedDict()
    order: List[tuple[str, str]] = []

    i = 0
    n = len(tokens)
    while i < n and tokens[i] != "<SOT>":
        prefix_tokens.append(tokens[i])
        i += 1

    while i < n:
        if tokens[i] != "<SOT>":
            suffix_tokens = tokens[i:]
            break

        i += 1
        if i >= n or tokens[i] != "<SUBJ>":
            return text
        i += 1

        subj_tokens: List[str] = []
        while i < n and tokens[i] not in {"<PRED>", "<SOT>", "<EOT>"}:
            subj_tokens.append(tokens[i])
            i += 1
        if i >= n or tokens[i] != "<PRED>":
            return text
        i += 1

        pred_tokens: List[str] = []
        while i < n and tokens[i] not in {"<OBJ>", "<SOT>", "<EOT>"}:
            pred_tokens.append(tokens[i])
            i += 1
        if i >= n or tokens[i] != "<OBJ>":
            return text
        i += 1

        obj_tokens: List[str] = []
        while i < n and tokens[i] not in {"<EOT>", "<SOT>"}:
            obj_tokens.append(tokens[i])
            i += 1
        if i >= n or tokens[i] != "<EOT>":
            return text
        i += 1

        subj_text = " ".join(subj_tokens).strip()
        pred_text = " ".join(pred_tokens).strip()
        obj_text = " ".join(obj_tokens).strip()
        key = (subj_text, pred_text)

        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(obj_text)

    if not any(len(values) > 1 for values in groups.values()):
        return text

    parts: List[str] = []
    if prefix_tokens:
        parts.append(" ".join(prefix_tokens))

    for subj_text, pred_text in order:
        objects = groups.get((subj_text, pred_text), [])
        block: List[str] = ["<SOT>", "<SUBJ>"]
        if subj_text:
            block.append(subj_text)
        block.append("<PRED>")
        if pred_text:
            block.append(pred_text)
        if len(objects) > 1:
            block.append(RDF_OBJECT_LIST_TOKEN)
            block.append(f" {RDF_LIST_SEPARATOR_TOKEN} ".join(obj for obj in objects if obj))
        else:
            block.append("<OBJ>")
            if objects:
                block.append(objects[0])
        block.append("<EOT>")
        parts.append(" ".join(tok for tok in block if tok).strip())

    if suffix_tokens:
        parts.append(" ".join(suffix_tokens))

    return " ".join(part for part in parts if part).strip()


def _iter_examples(
    path: str,
    tokenizer: Any,
    *,
    max_len: int,
    task_hint: str | None = None,
) -> Iterator[Seq2SeqExample]:
    """Yield tokenised examples from a JSONL file."""
    inferred_task = _infer_task_from_path(path)
    pad_id = _get_pad_id(tokenizer)
    sot_id = _token_to_id(tokenizer, "<SOT>")
    eot_id = _token_to_id(tokenizer, "<EOT>")
    if sot_id is None:
        fallback = eot_id if eot_id is not None else pad_id
        if fallback is None:
            raise ValueError("Tokenizer privo di token di avvio (<SOT>, <EOT> o <pad>).")
        sot_id = int(fallback)
    prefix_len = 1 if sot_id is not None else 0
    suffix_len = 1 if eot_id is not None else 0
    for record in read_jsonl(path):
        source = str(record.get("input") or record.get("source") or record.get("text") or "")
        source = compact_rdf(source)
        target = str(record.get("target") or record.get("output") or record.get("label") or "")
        if not source or not target:
            continue
        film = record.get("film")
        task_name = _normalise_task_name(record.get("task"), inferred_task)
        task_name = _normalise_task_name(task_hint, task_name)

        input_ids = _truncate(_encode(tokenizer, source), max_len)
        # Encode targets keeping room for start/end control tokens so the
        # decoder receives the same prompts used during inference.
        prefix: List[int] = [int(sot_id)] if sot_id is not None else []
        suffix: List[int] = [int(eot_id)] if eot_id is not None else []

        content_budget = max(0, max_len - prefix_len - suffix_len)
        content_ids = _truncate(_encode(tokenizer, target), content_budget)
        label_ids = prefix + content_ids
        if suffix and len(label_ids) < max_len:
            label_ids.extend(suffix[: max(0, max_len - len(label_ids))])

        yield Seq2SeqExample(
            input_text=source,
            target_text=target,
            task=task_name,
            film=film,
            input_ids=input_ids,
            label_ids=label_ids,
        )


class MultiTaskDataset(Dataset):
    """Store pre-tokenised seq2seq examples in memory."""

    def __init__(self, items: Sequence[Seq2SeqExample]):
        """Create a dataset backed by an in-memory list of examples."""
        self.items: List[Seq2SeqExample] = list(items)

    def __len__(self) -> int:  # pragma: no cover - trivial
        """Return the number of stored examples."""

        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return the example payload ready for batching."""
        example = self.items[index]
        payload: Dict[str, Any] = {
            "input_ids": example.input_ids,
            "labels": example.label_ids,
            "task": example.task,
            "raw_input": example.input_text,
            "raw_target": example.target_text,
        }
        if example.film is not None:
            payload["film"] = example.film
        return payload

    def select_first(self, n_examples: int) -> "MultiTaskDataset":
        """Return a shallow copy containing only the first *n_examples*."""
        return MultiTaskDataset(self.items[: max(1, n_examples)])

    def task_counts(self) -> Counter:
        """Return the number of examples per task."""
        return Counter(example.task for example in self.items)

    def task_fractions(self) -> Dict[str, float]:
        """Return normalised task ratios useful for the sampler."""
        counts = self.task_counts()
        total = sum(counts.values()) or 1
        return {task: count / total for task, count in counts.items()}


class JsonlSeq2Seq(MultiTaskDataset):
    """Load and cache a single JSONL file as a seq2seq dataset."""

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        *,
        max_len: int,
        task: str | None = None,
    ) -> None:
        """Materialise the JSONL file contents into :class:`Seq2SeqExample` objects."""
        self.path = str(path)
        self.max_len = int(max_len)
        self.pad_id = _get_pad_id(tokenizer)
        items = list(
            _iter_examples(
                self.path,
                tokenizer,
                max_len=self.max_len,
                task_hint=task,
            )
        )
        super().__init__(items)


class MultiTaskSampler(Sampler[List[int]]):
    """Draw balanced batches according to task ratios."""

    def __init__(
        self,
        dataset: MultiTaskDataset,
        batch_size: int,
        ratios: Dict[str, float],
        *,
        drop_last: bool = False,
    ) -> None:
        """Build task-specific index pools so batches can be balanced on demand."""
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
        """Yield balanced batches honouring the requested task ratios."""
        task_iters: Dict[str, Iterator[int]] = {}
        for task, indices in self.indices_by_task.items():
            random.shuffle(indices)
            task_iters[task] = iter(indices)

        num_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            num_batches += 1

        allocation = self._compute_batch_allocation()
        tasks = list(self.task_ratios.keys())
        active_tasks = [task for task, count in allocation.items() if count > 0]
        sampling_tasks = active_tasks or tasks
        for _ in range(num_batches):
            batch: List[int] = []
            for task_name, desired in allocation.items():
                if desired <= 0:
                    continue
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
                task_name = random.choice(sampling_tasks)
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

    def _compute_batch_allocation(self) -> Dict[str, int]:
        """Return the per-task sample counts for a balanced batch."""

        tasks = list(self.task_ratios.keys())
        if not tasks or self.batch_size <= 0:
            return {task: 0 for task in tasks}

        desired_counts = {
            task: self.task_ratios[task] * self.batch_size for task in tasks
        }
        allocations = {task: math.floor(desired_counts[task]) for task in tasks}
        remainders = {
            task: desired_counts[task] - allocations[task] for task in tasks
        }
        remaining = self.batch_size - sum(allocations.values())

        if len(tasks) > self.batch_size and remaining > 0:
            zero_tasks = [task for task in tasks if allocations[task] == 0]
            zero_tasks.sort(key=lambda t: self.task_ratios[t], reverse=True)
            for task in zero_tasks:
                if remaining <= 0:
                    break
                allocations[task] += 1
                remainders[task] = 0.0
                remaining -= 1

        if remaining > 0:
            for task in sorted(tasks, key=lambda t: remainders[t], reverse=True):
                if remaining <= 0:
                    break
                if remainders[task] <= 0:
                    continue
                allocations[task] += 1
                remaining -= 1
                remainders[task] = 0.0

        if remaining > 0:
            for task in sorted(tasks, key=lambda t: self.task_ratios[t], reverse=True):
                if remaining <= 0:
                    break
                allocations[task] += 1
                remaining -= 1

        total = sum(allocations.values())
        if total > self.batch_size:
            excess = total - self.batch_size
            for task in sorted(tasks, key=lambda t: (self.task_ratios[t], allocations[t])):
                if excess <= 0:
                    break
                removable = min(allocations[task], excess)
                if removable <= 0:
                    continue
                allocations[task] -= removable
                excess -= removable
        elif total < self.batch_size:
            deficit = self.batch_size - total
            for task in sorted(tasks, key=lambda t: self.task_ratios[t], reverse=True):
                if deficit <= 0:
                    break
                allocations[task] += 1
                deficit -= 1

        return allocations

    def __len__(self) -> int:  # pragma: no cover - straightforward math
        """Return the number of batches that will be produced."""

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
    """Pad variable-length examples into a dense batch."""
    if label_pad_id is None:
        label_pad_id = pad_id

    def _pad_tensor(sequences: Sequence[Sequence[int]], value: int) -> torch.Tensor:
        """Convert python lists into padded dense tensors with padding *value*."""
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

    collated["tasks"] = [ex.get("task", "") for ex in batch]
    collated["raw_input"] = [ex.get("raw_input") for ex in batch]
    collated["raw_target"] = [ex.get("raw_target") for ex in batch]
    collated["films"] = [ex.get("film") for ex in batch]
    return collated


class PadCollator:
    """Wrap :func:`pad_collate` in a picklable callable."""

    def __init__(self, *, pad_id: int, label_pad_id: int | None = None) -> None:
        """Store padding identifiers to be reused by worker processes."""
        self.pad_id = int(pad_id)
        self.label_pad_id = label_pad_id if label_pad_id is None else int(label_pad_id)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pad *batch* using the configured identifiers."""
        return pad_collate(batch, pad_id=self.pad_id, label_pad_id=self.label_pad_id)


class StaticBatchLoader:
    """Yield a pre-collated batch multiple times without touching the dataset.

    The overfit sanity check only needs to repeatedly optimise on the same
    examples. Building a full :class:`~torch.utils.data.DataLoader` (with the
    sampler, worker initialisation and collation) for every epoch introduces a
    measurable overhead, especially when the dataset has already been sliced
    down to a handful of samples.  This lightweight loader simply materialises a
    single batch once and then replays it ``repeats`` times, exposing the same
    interface as a regular DataLoader for the parts of the training loop that
    rely on ``__len__`` and ``__iter__``.
    """

    def __init__(self, batch: Dict[str, torch.Tensor], *, repeats: int = 1) -> None:
        if not isinstance(batch, dict):  # pragma: no cover - defensive
            raise TypeError("StaticBatchLoader richiede un batch dict-like")
        self._batch = batch
        self._repeats = max(1, int(repeats))

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for _ in range(self._repeats):
            yield self._batch

    def __len__(self) -> int:
        return self._repeats


def create_multitask_dataloader(
    dataset: MultiTaskDataset,
    *,
    tokenizer: Any,
    batch_size: int,
    ratios: Dict[str, float] | None = None,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader configured for multi-task batches."""
    pad_id = _get_pad_id(tokenizer)
    collate = PadCollator(pad_id=pad_id)

    if shuffle:
        drop_last = True
        dataset_size = len(dataset)
        if dataset_size < batch_size:
            LOGGER.warning(
                "Il dataset contiene %d esempi ma batch_size=%d: disattivo drop_last per evitare epoche vuote.",
                dataset_size,
                batch_size,
            )
            drop_last = False

        sampler = MultiTaskSampler(
            dataset,
            batch_size=batch_size,
            ratios=ratios or dataset.task_fractions(),
            drop_last=drop_last,
        )
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=num_workers)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )


def materialise_single_batch(
    dataset: MultiTaskDataset,
    *,
    tokenizer: Any,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Return a single collated batch from *dataset*.

    The function draws the first ``batch_size`` elements (or fewer if the
    dataset is smaller), applies the standard padding collator and returns the
    resulting tensors.  It is primarily used by the overfit command to cache the
    batch once and reuse it across epochs.
    """

    if len(dataset) == 0:
        raise ValueError("Dataset vuoto: impossibile materializzare un batch")

    pad_id = _get_pad_id(tokenizer)
    collate = PadCollator(pad_id=pad_id)
    take = min(batch_size, len(dataset)) or 1
    examples = [dataset[idx] for idx in range(take)]
    return collate(examples)
