"""Dataloaders and collators for multitask training."""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

if TYPE_CHECKING:
    from tokenizers import Tokenizer


class MultiTaskSampler(Sampler[int]):
    """A custom sampler for multitask learning.

    This sampler ensures that each batch contains a mix of examples from
    different tasks, according to specified ratios. It first groups the dataset
    by task name and then, for each batch, samples a proportional number of
    indices from each task group.

    This approach helps maintain a stable and balanced training signal when
    combining multiple objectives.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        ratios: dict[str, float],
        drop_last: bool = False,
    ):
        """Initializes the MultiTaskSampler.

        Args:
            dataset: A PyTorch Dataset where each item is a dictionary
                containing a "task" key.
            batch_size: The total number of samples per batch.
            ratios: A dictionary mapping task names to their desired ratio
                in each batch (e.g., {"task_a": 0.5, "task_b": 0.5}).
            drop_last: If True, the sampler will drop the last batch if its size
                is less than batch_size.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.ratios = ratios
        self.drop_last = drop_last

        # Group indices by task
        self.indices_by_task = defaultdict(list)
        for i, item in enumerate(cast(Any, self.dataset)):
            self.indices_by_task[item["task"]].append(i)

        self.num_samples = len(self.dataset)
        self.task_names = list(self.ratios.keys())
        
        # Normalize ratios to sum to 1
        total_ratio = sum(self.ratios.values())
        self.task_ratios = {task: r / total_ratio for task, r in self.ratios.items()}

    def __iter__(self):
        # Shuffle indices within each task group at the start of each epoch
        for task in self.indices_by_task:
            random.shuffle(self.indices_by_task[task])

        # Create iterators for each task's indices
        task_iters = {task: iter(indices) for task, indices in self.indices_by_task.items()}
        
        num_batches = self.num_samples // self.batch_size
        if not self.drop_last and self.num_samples % self.batch_size != 0:
            num_batches += 1

        for _ in range(num_batches):
            batch_indices = []
            for task_name, ratio in self.task_ratios.items():
                num_task_samples = int(self.batch_size * ratio)
                
                # Fetch indices for the current task
                task_samples = []
                for _ in range(num_task_samples):
                    try:
                        task_samples.append(next(task_iters[task_name]))
                    except StopIteration:
                        # If a task runs out of examples, reshuffle and restart its iterator
                        random.shuffle(self.indices_by_task[task_name])
                        task_iters[task_name] = iter(self.indices_by_task[task_name])
                        task_samples.append(next(task_iters[task_name]))
                
                batch_indices.extend(task_samples)
            
            # Handle rounding errors by filling up to batch_size
            while len(batch_indices) < self.batch_size:
                random_task = random.choice(self.task_names)
                try:
                    batch_indices.append(next(task_iters[random_task]))
                except StopIteration:
                    random.shuffle(self.indices_by_task[random_task])
                    task_iters[random_task] = iter(self.indices_by_task[random_task])
                    batch_indices.append(next(task_iters[random_task]))
            
            # Shuffle the final batch to mix tasks
            random.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self) -> int:
        return self.num_samples


class MultiTaskCollator:
    """Collator for creating and padding multi-task batches.

    This class orchestrates the preparation of a batch by delegating to
    task-specific helper functions. It handles the final padding of all tensor
    fields to ensure they have consistent lengths within the batch.
    """

    def __init__(self, tokenizer: Tokenizer, max_length: int):
        """Initializes the MultiTaskCollator.

        Args:
            tokenizer: The tokenizer instance.
            max_length: The maximum sequence length for padding and truncation.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.token_to_id("[PAD]")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collates a list of features into a padded batch.

        Args:
            features: A list of individual data points from the dataset.

        Returns:
            A dictionary of padded tensors ready for the model.
        """
        prepared_features = []
        for item in features:
            task_handler = self._get_task_handler(item["task"])
            prepared_features.append(task_handler(item))

        # Collate fields from all prepared features
        first = prepared_features[0]
        batch = {}
        for k in first.keys():
            # Pad each field to the max length in the batch for that field
            # The tokenizer's pad method is highly optimized for this.
            field_values = [f[k] for f in prepared_features if k in f and f[k] is not None]
            if not field_values:
                continue

            # Convert to tensors before padding
            if isinstance(field_values[0], list):
                field_values = [torch.tensor(val, dtype=torch.long) for val in field_values]
            
            batch[k] = torch.nn.utils.rnn.pad_sequence(
                field_values, batch_first=True, padding_value=self.pad_token_id
            )
        
        # Ensure 'attention_mask' is created for the input
        if "input_ids" in batch:
            batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).long()

        return batch

    def _get_task_handler(self, task_name: str):
        """Returns the appropriate data preparation function for a given task."""
        if task_name == "Text2RDF":
            return self._prepare_text2rdf
        if task_name == "RDF2Text":
            return self._prepare_rdf2text
        if task_name == "RDFCompletion_Msk":
            return self._prepare_rdf_completion_mask
        if task_name == "RDFCompletion_Cont":
            return self._prepare_rdf_completion_continue
        raise ValueError(f"Unknown task name: {task_name}")

    def _prepare_text2rdf(self, item: dict) -> dict:
        """Prepares a Text-to-RDF example."""
        # Input: text + task_token
        # Target: rdf
        return {
            "input_ids": item["text_ids"] + item["task_ids"],
            "labels": item["rdf_ids"],
        }
        
    def _prepare_rdf2text(self, item: dict) -> dict:
        """Prepares an RDF-to-Text example."""
        # Input: rdf + task_token
        # Target: text
        return {
            "input_ids": item["rdf_ids"] + item["task_ids"],
            "labels": item["text_ids"],
        }

    def _prepare_rdf_completion_mask(self, item: dict) -> dict:
        """Prepares an RDF masked slot prediction example."""
        # Input: rdf_with_mask + task_token
        # Target: masked span (in 'labels'), rest is padded
        
        input_ids = item["rdf_masked_ids"] + item["task_ids"]
        labels = torch.full((len(input_ids),), self.pad_token_id, dtype=torch.long)
        
        # The true label is just the masked span
        target_span = torch.tensor(item["rdf_masked_span_ids"], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": target_span,
            "mask_positions": torch.tensor([item["mask_start_pos"]], dtype=torch.long),
            "mask_lengths": torch.tensor([len(target_span)], dtype=torch.long),
        }

    def _prepare_rdf_completion_continue(self, item: dict) -> dict:
        """Prepares an RDF continuation example."""
        # Input: partial_rdf + task_token
        # Target: remaining_rdf
        return {
            "input_ids": item["rdf_context_ids"] + item["task_ids"],
            "labels": item["rdf_continuation_ids"],
        }


def create_multitask_dataloader(
    dataset: Dataset,
    batch_size: int,
    tokenizer: Tokenizer,
    max_length: int,
    ratios: dict[str, float],
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Creates a DataLoader for multi-task training.

    Args:
        dataset: The dataset containing examples for all tasks.
        batch_size: The target batch size.
        tokenizer: The tokenizer for creating the collator.
        max_length: Maximum sequence length.
        ratios: Dictionary of task ratios for sampling.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the data. If True, MultiTaskSampler is used.

    Returns:
        A configured PyTorch DataLoader.
    """
    collator = MultiTaskCollator(tokenizer, max_length)
    
    if shuffle:
        sampler = MultiTaskSampler(dataset, batch_size, ratios, drop_last=True)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
        )
    else:
        # For validation/testing, no special sampler is needed
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            shuffle=False,
        )