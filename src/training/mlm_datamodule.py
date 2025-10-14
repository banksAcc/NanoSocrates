"""Data utilities for masked language modeling using Hugging Face collators."""

from __future__ import annotations

from typing import List, Optional, Sequence

from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from src.utils.special_tokens import REQUIRED_SPECIAL_TOKENS


class MaskedTextDataset(Dataset):
    """Dataset that tokenizes raw texts on-the-fly for MLM training."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        filtered = [text for text in texts if text and text.strip()]
        if not filtered:
            raise ValueError("MaskedTextDataset requires at least one non-empty text")
        self.examples: List[str] = filtered
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, List[int]]:
        text = self.examples[index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            add_special_tokens=True,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


class MLMDataModule:
    """Utility class that prepares DataLoaders for masked language modelling."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        train_texts: Sequence[str],
        val_texts: Optional[Sequence[str]] = None,
        *,
        max_length: int = 256,
        batch_size: int = 32,
        mlm_probability: float = 0.15,
        num_workers: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.train_texts: List[str] = list(train_texts)
        self.val_texts: Optional[List[str]] = list(val_texts) if val_texts is not None else None
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.mlm_probability = float(mlm_probability)
        self.num_workers = int(num_workers)
        self.shuffle = bool(shuffle)

        self.data_collator: Optional[DataCollatorForLanguageModeling] = None
        self.train_dataset: Optional[MaskedTextDataset] = None
        self.val_dataset: Optional[MaskedTextDataset] = None

    def _ensure_special_tokens(self) -> None:
        vocab = self.tokenizer.get_vocab()
        additional_tokens = [tok for tok in REQUIRED_SPECIAL_TOKENS if tok not in vocab]
        if additional_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})

    def setup(self) -> None:
        """Initialise datasets and the masking collator."""

        self._ensure_special_tokens()
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )
        self.train_dataset = MaskedTextDataset(
            self.train_texts,
            self.tokenizer,
            max_length=self.max_length,
        )
        if self.val_texts is not None:
            self.val_dataset = MaskedTextDataset(
                self.val_texts,
                self.tokenizer,
                max_length=self.max_length,
            )
        else:
            self.val_dataset = None

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None or self.data_collator is None:
            raise RuntimeError("setup() must be called before requesting a dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        if self.data_collator is None:
            raise RuntimeError("setup() must be called before requesting a dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

*** End of File ***
