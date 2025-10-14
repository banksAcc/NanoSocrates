"""Data loading utilities built on top of the 🤗 datasets library."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, List, Optional

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader

from src.training.dataloaders import PadCollator, compact_rdf

LOGGER = logging.getLogger(__name__)


class JSONLSeq2SeqDataModule:
    """Load JSON Lines files with :mod:`datasets` and prepare PyTorch loaders."""

    def __init__(
        self,
        tokenizer: Any,
        data_files: Mapping[str, str],
        *,
        batch_size: int = 8,
        max_input_length: int = 512,
        max_target_length: int = 128,
        num_workers: int = 0,
        task_name: str | None = None,
        shuffle_train: bool = True,
    ) -> None:
        if "train" not in data_files or "validation" not in data_files:
            raise ValueError("'data_files' deve contenere le chiavi 'train' e 'validation'.")

        self.tokenizer = tokenizer
        self.data_files = dict(data_files)
        self.batch_size = int(batch_size)
        self.max_input_length = int(max_input_length)
        self.max_target_length = int(max_target_length)
        self.num_workers = int(num_workers)
        self.task_name = str(task_name or "generic")
        self.shuffle_train = bool(shuffle_train)

        self.pad_id: Optional[int] = None
        self.sot_id: Optional[int] = None
        self.eot_id: Optional[int] = None

        self._label_prefix: List[int] = []
        self._label_suffix: List[int] = []

        self.processed_dataset: DatasetDict | None = None
        self.train_dataset = None
        self.val_dataset = None

    # ------------------------------------------------------------------ utils
    def _get_token_id(self, token: str) -> Optional[int]:
        lookup = getattr(self.tokenizer, "token_to_id", None)
        if callable(lookup):
            token_id = lookup(token)
            return int(token_id) if token_id is not None else None
        return None

    def _resolve_pad_id(self) -> int:
        if hasattr(self.tokenizer, "pad_id") and getattr(self.tokenizer, "pad_id") is not None:
            return int(getattr(self.tokenizer, "pad_id"))
        pad_id = self._get_token_id("<pad>")
        if pad_id is None:
            raise ValueError("Tokenizer privo di <pad>: rigenera il BPE includendo <pad>.")
        return pad_id

    @staticmethod
    def _encode_object(encoded: Any) -> List[int]:
        if hasattr(encoded, "ids"):
            return [int(tok) for tok in encoded.ids]
        if isinstance(encoded, (list, tuple)):
            return [int(tok) for tok in encoded]
        return [int(tok) for tok in encoded]

    def _encode_batch(self, texts: Sequence[str], max_len: int) -> List[List[int]]:
        if not texts:
            return []
        encode_batch = getattr(self.tokenizer, "encode_batch", None)
        if callable(encode_batch):
            encodings = encode_batch(list(texts))
            sequences = [self._encode_object(enc)[:max_len] for enc in encodings]
        else:
            encode = getattr(self.tokenizer, "encode")
            sequences = [self._encode_object(encode(text))[:max_len] for text in texts]
        return sequences

    def _encode_targets(self, targets: Sequence[str]) -> List[List[int]]:
        prefix = list(self._label_prefix)
        suffix = list(self._label_suffix)
        content_budget = max(0, self.max_target_length - len(prefix) - len(suffix))

        content_sequences = self._encode_batch(targets, content_budget)
        processed: List[List[int]] = []
        for content in content_sequences:
            label = prefix + content
            if suffix and len(label) < self.max_target_length:
                needed = min(len(suffix), self.max_target_length - len(label))
                label.extend(suffix[:needed])
            processed.append(label)
        return processed

    @staticmethod
    def _columns_to_remove(dataset: DatasetDict) -> List[str]:
        keep = {"film", "task"}
        columns: set[str] = set()
        for split in dataset.keys():
            columns.update(dataset[split].column_names)
        return sorted(col for col in columns if col not in keep)

    @staticmethod
    def _normalise_task(value: Any, fallback: str) -> str:
        if value is None:
            return fallback
        text = str(value).strip()
        return text if text else fallback

    @staticmethod
    def _extract_sources(batch: Mapping[str, Sequence[Any]]) -> Sequence[str]:
        for key in ("input", "source", "text"):
            if key in batch:
                return batch[key]
        return []

    @staticmethod
    def _extract_targets(batch: Mapping[str, Sequence[Any]]) -> Sequence[str]:
        for key in ("target", "output", "label"):
            if key in batch:
                return batch[key]
        return []

    @staticmethod
    def _has_valid_pair(example: Mapping[str, Any]) -> bool:
        source = (
            example.get("input")
            or example.get("source")
            or example.get("text")
            or ""
        )
        target = (
            example.get("target")
            or example.get("output")
            or example.get("label")
            or ""
        )
        return bool(str(source).strip()) and bool(str(target).strip())

    # ----------------------------------------------------------------- dataset
    def setup(self, stage: str | None = None) -> None:
        """Materialise and preprocess the dataset splits."""

        raw_dataset = load_dataset("json", data_files=self.data_files)
        raw_dataset = raw_dataset.filter(self._has_valid_pair)

        self.pad_id = self._resolve_pad_id()
        self.sot_id = self._get_token_id("<SOT>")
        self.eot_id = self._get_token_id("<EOT>")
        self._label_prefix = [int(self.sot_id)] if self.sot_id is not None else []
        self._label_suffix = [int(self.eot_id)] if self.eot_id is not None else []

        remove_columns = self._columns_to_remove(raw_dataset)

        def preprocess_function(batch: Mapping[str, Sequence[Any]]) -> Dict[str, Iterable[Any]]:
            raw_sources = self._extract_sources(batch)
            raw_targets = self._extract_targets(batch)
            if len(raw_sources) != len(raw_targets):
                raise ValueError("Numero di input e target incoerente nel batch.")

            compact_inputs = [compact_rdf(str(src or "")) for src in raw_sources]
            targets = [str(tgt or "") for tgt in raw_targets]

            model_inputs: Dict[str, Iterable[Any]] = {
                "input_ids": self._encode_batch(compact_inputs, self.max_input_length),
                "labels": self._encode_targets(targets),
                "raw_input": compact_inputs,
                "raw_target": targets,
            }

            tasks = batch.get("task")
            if tasks is not None:
                model_inputs["task"] = [self._normalise_task(task, self.task_name) for task in tasks]
            else:
                model_inputs["task"] = [self.task_name] * len(compact_inputs)

            films = batch.get("film")
            if films is not None:
                model_inputs["film"] = [str(film) if film is not None else None for film in films]

            return model_inputs

        processed_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=remove_columns,
        )

        self.processed_dataset = processed_dataset
        self.train_dataset = processed_dataset.get("train")
        self.val_dataset = processed_dataset.get("validation")

        LOGGER.info(
            "Dataset caricato: train=%d, validation=%d",
            len(self.train_dataset) if self.train_dataset is not None else 0,
            len(self.val_dataset) if self.val_dataset is not None else 0,
        )

    # --------------------------------------------------------------- dataloader
    def _build_dataloader(self, dataset, *, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("setup() deve essere chiamato prima di creare i dataloader.")
        if self.pad_id is None:
            raise RuntimeError("Impossibile costruire il dataloader senza pad_id valido.")

        collate = PadCollator(pad_id=self.pad_id)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for the training split."""

        return self._build_dataloader(self.train_dataset, shuffle=self.shuffle_train)

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader for the validation split."""

        return self._build_dataloader(self.val_dataset, shuffle=False)
