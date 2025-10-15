"""Wrapper utilities for Hugging Face masked language models."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import AutoModelForMaskedLM, PreTrainedModel, PreTrainedTokenizerBase

from src.utils.special_tokens import ensure_required_special_tokens


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
        ensure_required_special_tokens(self.tokenizer)

        kwargs = dict(model_kwargs or {})
        self.model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

        if resize_token_embeddings:
            embedding_layer = self.model.resize_token_embeddings(len(self.tokenizer))
            if embedding_layer.num_embeddings != len(self.tokenizer):
                raise RuntimeError(
                    "Model embeddings do not match tokenizer vocabulary size after resizing"
                )

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
