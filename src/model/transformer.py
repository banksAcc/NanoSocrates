"""Seq2Seq transformer model with optional MLA, RoPE, and T5 variants."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .layers import CustomTransformer, SinusoidalPE, T5Transformer
from .losses import sequence_loss_with_span_metrics


class TinySeq2Seq(nn.Module):
    """A flexible sequence-to-sequence Transformer model.

    This class implements a standard Transformer encoder-decoder architecture
    with several optional customisations, including:
    - A T5-style architecture with relative position biases.
    - A "vanilla" architecture using either the standard `nn.Transformer` or
      a custom implementation that supports features like Rotary Positional
      Embeddings (RoPE) and Multi-Linear Attention (MLA).
    - Tied input and output embeddings.
    - Optional computation of span-based metrics for tasks like masked slot
      prediction.

    Attributes:
        pad_id (int): The padding token ID, used for creating masks.
        architecture (str): The selected model architecture ('vanilla' or 't5').
        emb (nn.Embedding): The token embedding layer.
        pe (nn.Module | None): The positional encoding layer (e.g., SinusoidalPE).
        tfm (nn.Module): The core Transformer module.
        lm_head (nn.Linear): The final linear layer to produce logits.
        uses_custom_architecture (bool): Flag indicating if a custom Transformer
            implementation is used instead of `nn.Transformer`.
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
        use_mla: bool = False,
        use_rope: bool = False,
        interleave_ratio: float = 0.0,
        max_position_embeddings: int = 2048,
        compute_span_metrics: bool = False,
        architecture: str = "vanilla",
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
            use_mla: If True, enables Multi-Linear Attention in the custom 'vanilla'
                transformer. Incompatible with 't5' architecture.
            use_rope: If True, enables Rotary Positional Embeddings in the custom
                'vanilla' transformer. Incompatible with 't5' architecture.
            interleave_ratio: The ratio for interpolating between standard attention
                and MLA. A value > 0 activates the custom transformer.
            max_position_embeddings: The maximum sequence length for positional encodings.
            compute_span_metrics: If True, computes span-based metrics during loss
                calculation, useful for tasks like masked prediction.
            architecture: The model variant to use. Can be 'vanilla' or 't5'.
            relative_attention_num_buckets: Number of buckets for T5 relative
                position bias. Only used if architecture is 't5'.
            relative_attention_max_distance: Maximum distance for T5 relative
                position bias. Only used if architecture is 't5'.
            layer_norm_epsilon: Epsilon for LayerNorm layers.
        """
        super().__init__()
        self.pad_id = pad_id
        self.compute_span_metrics = bool(compute_span_metrics)
        arch = (architecture or "vanilla").lower()
        if arch not in {"vanilla", "t5"}:
            raise ValueError("architecture must be 'vanilla' or 't5'")
        self.architecture = arch

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embedding_scale = math.sqrt(d_model) if self.architecture == "t5" else 1.0
        self.dropout_layer = nn.Dropout(dropout)

        # This flag determines if a custom implementation is needed.
        # It's true for T5 or for 'vanilla' with RoPE/MLA features.
        self.uses_custom_architecture: bool = False

        if self.architecture == "t5":
            if use_rope or use_mla or abs(float(interleave_ratio)) > 1e-8:
                raise ValueError("T5 architecture does not support RoPE/MLA/interleave options")
            self.pe = None
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
            self.uses_custom_architecture = True
        else:  # vanilla architecture
            self.pe = None if use_rope else SinusoidalPE(d_model, max_len=max_position_embeddings)
            
            # Activate custom transformer if RoPE, MLA, or interpolation is used.
            if use_rope or use_mla or interleave_ratio > 0.0:
                self.uses_custom_architecture = True
                self.tfm = CustomTransformer(
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_mla=use_mla,
                    interleave_ratio=interleave_ratio,
                    use_rope=use_rope,
                    max_position_embeddings=max_position_embeddings,
                )
            else:
                self.tfm = nn.Transformer(
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.emb.weight

    @staticmethod
    def _subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generates a causal mask for the decoder."""
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *,
        mask_positions: torch.Tensor | None = None,
        mask_lengths: torch.Tensor | None = None,
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
            mask_positions: Start positions of masked spans for metric calculation.
                Shape: (batch_size, num_spans)
            mask_lengths: Lengths of masked spans for metric calculation.
                Shape: (batch_size, num_spans)

        Returns:
            A dictionary containing:
            - "logits": The model's output logits. Shape: (batch_size, tgt_seq_len, vocab_size)
            - "loss": The computed cross-entropy loss, if `labels` are provided.
            - "metrics": A dictionary of computed span metrics, if `compute_span_metrics`
              is True and labels are provided.
        """
        device = input_ids.device
        
        # Prepare encoder inputs
        enc = self.emb(input_ids) * self.embedding_scale
        if self.pe is not None:
            enc = self.pe(enc)
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
        if self.pe is not None:
            dec_in = self.pe(dec_in)
        dec_in = self.dropout_layer(dec_in)
        
        tgt_causal_mask = self._subsequent_mask(decoder_input_ids.size(1), device)
        tgt_key_padding_mask = decoder_input_ids == self.pad_id

        # Execute transformer pass based on architecture
        if self.architecture == "t5":
            memory = self.tfm.encode(enc, key_padding_mask=src_key_padding_mask)
            out = self.tfm.decode(
                dec_in,
                memory,
                self_attn_mask=tgt_causal_mask,
                self_key_padding_mask=tgt_key_padding_mask,
                cross_key_padding_mask=src_key_padding_mask,
            )
        elif self.uses_custom_architecture:
            memory = self.tfm.encode(enc, src_key_padding_mask=src_key_padding_mask)
            out = self.tfm.decode(
                dec_in,
                memory,
                tgt_mask=tgt_causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
        else:  # Standard nn.Transformer
            memory = self.tfm.encoder(enc, src_key_padding_mask=src_key_padding_mask)
            out = self.tfm.decoder(
                dec_in,
                memory,
                tgt_mask=tgt_causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            
        logits = self.lm_head(out)

        # Compute loss and metrics if labels are available
        loss, metrics = None, None
        if labels is not None:
            loss, metrics = sequence_loss_with_span_metrics(
                logits=logits,
                labels=labels,
                pad_id=self.pad_id,
                mask_positions=mask_positions,
                mask_lengths=mask_lengths,
                compute_metrics=self.compute_span_metrics,
            )
            
        payload = {"logits": logits, "loss": loss}
        if metrics:
            payload["metrics"] = metrics
            
        return payload