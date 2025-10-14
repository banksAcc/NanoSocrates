"""Wrapper utilities for Hugging Face masked language models."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import AutoModelForMaskedLM, PreTrainedModel, PreTrainedTokenizerBase

from src.utils.special_tokens import REQUIRED_SPECIAL_TOKENS


def ensure_tokenizer_special_tokens(tokenizer: PreTrainedTokenizerBase) -> None:
    """Ensure that project-specific markers exist in the provided tokenizer."""

    vocab = tokenizer.get_vocab()
    additional_tokens = [tok for tok in REQUIRED_SPECIAL_TOKENS if tok not in vocab]
    if additional_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})


class MaskedLMTaskModule(nn.Module):
    """Thin wrapper around :class:`AutoModelForMaskedLM` with utility metrics."""

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        *,
        resize_token_embeddings: bool = True,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        ensure_tokenizer_special_tokens(self.tokenizer)

        kwargs = dict(model_kwargs or {})
        self.model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

        if resize_token_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        result: Dict[str, Any] = {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

        if labels is not None and outputs.logits is not None:
            with torch.no_grad():
                mask = labels != -100
                if torch.any(mask):
                    predictions = outputs.logits.argmax(dim=-1)
                    correct = ((predictions == labels) & mask).sum().item()
                    total = mask.sum().item()
                    result["metrics"] = {
                        "mask_accuracy": float(correct / total) if total else 0.0
                    }
        return result

*** End of File ***
