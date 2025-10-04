"""Seq2Seq transformer model with optional MLA and RoPE variants."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import CustomTransformer, SinusoidalPE


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
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
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

        if labels is not None:
            if labels.size(1) == logits.size(1):
                y_tgt = labels
            elif labels.size(1) == logits.size(1) + 1:
                y_tgt = labels[:, 1:]
            else:
                raise ValueError("labels length must match decoder_input_ids or be longer by one")
            logits_for_loss = logits[:, -y_tgt.size(1) :, :]
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_id)
            loss = loss_fct(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                y_tgt.reshape(-1),
            )
        return {"logits": logits, "loss": loss}

