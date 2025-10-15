"""Encoder-decoder transformer inspired by T5 for seq2seq tasks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .layers import T5Transformer
from .losses import sequence_loss_with_accuracy


class TinySeq2Seq(nn.Module):
    """Encoder-decoder transformer tailored to NanoSocrates tasks.

    The architecture mirrors the high level structure of models such as T5:

    * shared token embeddings for encoder and decoder inputs;
    * relative position biases injected inside every attention block;
    * pre-layer normalisation and residual connections on every sub-layer.

    Compared to the previous iteration the implementation is intentionally
    streamlined: legacy variants relying on rotary embeddings, multi linear
    attention or span metrics have been removed to keep the codebase focused on
    the sequence-to-sequence setting required by the project brief.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        nhead: int = 6,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        pad_id: int = 1,
        tie_embeddings: bool = True,
        *,
        max_position_embeddings: int = 2048,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        layer_norm_epsilon: float = 1e-6,
    ) -> None:
        """Initializes the TinySeq2Seq model.
        Args:
            vocab_size: The size of the vocabulary.
            d_model: The dimensionality of the model's embeddings and hidden states.
            nhead: The number of attention heads.
            num_encoder_layers: The number of layers in the encoder.
            num_decoder_layers: The number of layers in the decoder.
            dim_feedforward: The dimension of the feed-forward networks.
            dropout: The dropout rate.
            pad_id: The vocabulary ID for the padding token.
            tie_embeddings: Whether to tie the input and output embedding weights.
            max_position_embeddings: The maximum sequence length for positional encodings.
            relative_attention_num_buckets: Number of buckets for T5 relative
                position bias. Only used if architecture is 't5'.
            relative_attention_max_distance: Maximum distance for T5 relative
                position bias. Only used if architecture is 't5'.
            layer_norm_epsilon: Epsilon for LayerNorm layers.
        """
        super().__init__()
        self.pad_id = pad_id
        self.max_position_embeddings = int(max_position_embeddings)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # T5 uses scaled embeddings; keeping the same behaviour ensures the
        # network remains numerically stable even with tied weights.
        self.embedding_scale = math.sqrt(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        self.tfm = T5Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            layer_norm_epsilon=layer_norm_epsilon,
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.emb.weight

        self._checkpoint_config = {
            "d_model": int(d_model),
            "nhead": int(nhead),
            "enc_layers": int(num_encoder_layers),
            "dec_layers": int(num_decoder_layers),
            "ff_dim": int(dim_feedforward),
            "dropout": float(dropout),
            "max_len": int(max_position_embeddings),
            "relative_attention_num_buckets": int(relative_attention_num_buckets),
            "relative_attention_max_distance": int(relative_attention_max_distance),
            "layer_norm_epsilon": float(layer_norm_epsilon),
        }

    @staticmethod
    def _subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generates a causal mask for the decoder."""
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), 1)

    def export_config(self) -> dict[str, object]:
        """Returns the minimal configuration required to rebuild the model."""
        return dict(self._checkpoint_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Performs a forward pass through the model.
        Args:
            input_ids: The input token IDs for the encoder.
                Shape: (batch_size, src_seq_len)
            attention_mask: The attention mask for the encoder input, where 1 indicates
                a valid token and 0 indicates padding.
                Shape: (batch_size, src_seq_len)
            decoder_input_ids: The input token IDs for the decoder. If not provided,
                it is automatically generated from `labels` by shifting them right.
                Shape: (batch_size, tgt_seq_len)
            labels: The target token IDs for computing the loss. The last token is
                used for loss calculation but not as input to the decoder.
                Shape: (batch_size, tgt_seq_len + 1)

        Returns:
            A dictionary containing:
            - "logits": The model's output logits. Shape: (batch_size, tgt_seq_len, vocab_size)
            - "loss": The computed cross-entropy loss, if `labels` are provided.
            - "metrics": A dictionary containing the exact-match rate when labels are provided.
        """
        device = input_ids.device
        
        # Prepare encoder inputs
        enc = self.emb(input_ids) * self.embedding_scale
        enc = self.dropout_layer(enc)
        src_key_padding_mask = attention_mask == 0

        # Prepare decoder inputs
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("Either decoder_input_ids or labels must be provided.")
            if labels.size(1) < 2:
                raise ValueError("labels must have length >= 2 to build decoder_input_ids.")
            decoder_input_ids = labels[:, :-1]

        dec_in = self.emb(decoder_input_ids) * self.embedding_scale
        dec_in = self.dropout_layer(dec_in)

        tgt_causal_mask = self._subsequent_mask(decoder_input_ids.size(1), device)
        tgt_key_padding_mask = decoder_input_ids == self.pad_id

        memory = self.tfm.encode(enc, key_padding_mask=src_key_padding_mask)
        out = self.tfm.decode(
            dec_in,
            memory,
            self_attn_mask=tgt_causal_mask,
            self_key_padding_mask=tgt_key_padding_mask,
            cross_key_padding_mask=src_key_padding_mask,
        )

        logits = self.lm_head(out)

        # Compute loss and metrics if labels are available
        loss, metrics = None, None
        if labels is not None:
            loss, metrics = sequence_loss_with_accuracy(
                logits=logits,
                labels=labels,
                pad_id=self.pad_id,
                compute_metrics=True,
            )

        payload = {"logits": logits, "loss": loss}
        if metrics:
            payload["metrics"] = metrics

        return payload
