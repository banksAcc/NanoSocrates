"""Seq2Seq transformer model with optional MLA and RoPE variants."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import CustomTransformer, SinusoidalPE
from .losses import sequence_loss_with_span_metrics


class TinySeq2Seq(nn.Module):
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
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.compute_span_metrics = bool(compute_span_metrics)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pe = None if use_rope else SinusoidalPE(d_model, max_len=max_position_embeddings)

        self._use_custom = bool(use_rope or use_mla or interleave_ratio > 0.0)
        if self._use_custom:
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
        device = input_ids.device
        enc = self.emb(input_ids)
        if self.pe is not None:
            enc = self.pe(enc)
        src_kpm = attention_mask == 0

        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("decoder_input_ids or labels must be provided")
            if labels.size(1) < 2:
                raise ValueError("labels must have length >= 2 to build decoder_input_ids")
            decoder_input_ids = labels[:, :-1]

        y_inp = decoder_input_ids

        dec = self.emb(y_inp)
        if self.pe is not None:
            dec = self.pe(dec)
        tgt_mask = self._subsequent_mask(y_inp.size(1), device)
        tgt_kpm = y_inp == self.pad_id

        if self._use_custom:
            mem = self.tfm.encode(enc, src_key_padding_mask=src_kpm)
            out = self.tfm.decode(
                dec,
                mem,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=src_kpm,
            )
        else:
            mem = self.tfm.encoder(enc, src_key_padding_mask=src_kpm)
            out = self.tfm.decoder(
                dec,
                mem,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=src_kpm,
            )
        logits = self.lm_head(out)

        loss = None
        metrics: dict[str, float] | None = None

        if labels is not None:
            loss, metrics = sequence_loss_with_span_metrics(
                logits,
                labels,
                pad_id=self.pad_id,
                mask_positions=mask_positions,
                mask_lengths=mask_lengths,
                compute_metrics=self.compute_span_metrics,
            )
        payload = {"logits": logits, "loss": loss}
        if metrics:
            payload["metrics"] = metrics
        return payload

