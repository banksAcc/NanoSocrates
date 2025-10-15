"""T5-inspired encoder-decoder building blocks."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    """Learned relative position bias as introduced in T5."""

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        *,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or num_buckets <= 0:
            raise ValueError("num_heads and num_buckets must be > 0")
        self.num_heads = int(num_heads)
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.bidirectional = bool(bidirectional)
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = max(1, self.max_distance)
        rp = relative_position
        if self.bidirectional:
            half = num_buckets // 2
            sign = (rp > 0).to(torch.long) * half
            rp = rp.abs()
        else:
            half = num_buckets
            sign = torch.zeros_like(rp, dtype=torch.long)
            rp = (-rp).clamp_min(0)

        max_exact = half // 2
        is_small = rp < max_exact
        if max_exact > 0 and max_distance > max_exact:
            log_ratio = math.log(max_distance / max_exact)
            large_pos = max_exact + (
                torch.log(rp.float() / max_exact + 1e-6) / log_ratio
            ) * (half - max_exact)
        else:
            large_pos = torch.zeros_like(rp, dtype=rp.dtype)
        large_pos = torch.min(
            large_pos.to(dtype=rp.dtype),
            torch.full_like(rp, half - 1, dtype=rp.dtype),
        )

        buckets = torch.where(is_small, rp, large_pos).to(torch.long)
        return sign + buckets

    def forward(
        self,
        query_length: int,
        key_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        context_pos = torch.arange(query_length, device=device)[:, None]
        memory_pos = torch.arange(key_length, device=device)[None, :]
        relative_position = memory_pos - context_pos
        buckets = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0).to(dtype)


class T5Attention(nn.Module):
    """Multi-head attention with optional relative position bias."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        *,
        relative_bias: Optional[RelativePositionBias] = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.relative_bias = relative_bias

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = tensor.size()
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, dim = tensor.size()
        return tensor.transpose(1, 2).reshape(batch, seq_len, heads * dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_value_states: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self._reshape(self.q(hidden_states))
        if key_value_states is None:
            key = self._reshape(self.k(hidden_states))
            value = self._reshape(self.v(hidden_states))
        else:
            key = self._reshape(self.k(key_value_states))
            value = self._reshape(self.v(key_value_states))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.relative_bias is not None and position_bias is None:
            position_bias = self.relative_bias(
                query.size(-2), key.size(-2), device=query.device, dtype=query.dtype
            )
        if position_bias is not None:
            scores = scores + position_bias

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise TypeError("attn_mask must be boolean for T5Attention")
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores.masked_fill(attn_mask.to(device=scores.device), torch.finfo(scores.dtype).min)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = torch.matmul(weights, value)
        return self.out(self._merge(output))


class T5FeedForward(nn.Module):
    """Gated feed-forward network used by T5."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.wi_1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.wo = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.wi_0(x))
        hidden = self.wi_1(x)
        return self.dropout(self.wo(gate * hidden))


class T5EncoderLayer(nn.Module):
    """Single encoder layer with pre-layer norm."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        layer_norm_epsilon: float,
        relative_bias: RelativePositionBias,
    ) -> None:
        super().__init__()
        self.self_attn = T5Attention(d_model, nhead, dropout, relative_bias=relative_bias)
        self.feed_forward = T5FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.norm1(hidden_states)
        attn_out = self.self_attn(normed, key_padding_mask=key_padding_mask)
        x = hidden_states + self.dropout(attn_out)
        normed_ff = self.norm2(x)
        ff_out = self.feed_forward(normed_ff)
        return x + self.dropout(ff_out)


class T5DecoderLayer(nn.Module):
    """Single decoder layer with masked self-attention and cross-attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        layer_norm_epsilon: float,
        self_relative_bias: RelativePositionBias,
    ) -> None:
        super().__init__()
        self.self_attn = T5Attention(d_model, nhead, dropout, relative_bias=self_relative_bias)
        self.cross_attn = T5Attention(d_model, nhead, dropout, relative_bias=None)
        self.feed_forward = T5FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        *,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.norm1(hidden_states)
        self_attn_out = self.self_attn(
            normed,
            key_padding_mask=self_key_padding_mask,
            attn_mask=self_attn_mask,
        )
        x = hidden_states + self.dropout(self_attn_out)

        normed_cross = self.norm2(x)
        cross_out = self.cross_attn(
            normed_cross,
            key_value_states=memory,
            key_padding_mask=cross_key_padding_mask,
        )
        x = x + self.dropout(cross_out)

        normed_ff = self.norm3(x)
        ff_out = self.feed_forward(normed_ff)
        return x + self.dropout(ff_out)


class T5Transformer(nn.Module):
    """Stack of encoder and decoder layers with relative position biases."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        *,
        relative_attention_num_buckets: int,
        relative_attention_max_distance: int,
        layer_norm_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        encoder_bias = RelativePositionBias(
            nhead,
            relative_attention_num_buckets,
            relative_attention_max_distance,
            bidirectional=True,
        )
        decoder_bias = RelativePositionBias(
            nhead,
            relative_attention_num_buckets,
            relative_attention_max_distance,
            bidirectional=False,
        )
        self.encoder_layers = nn.ModuleList(
            [
                T5EncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    layer_norm_epsilon=layer_norm_epsilon,
                    relative_bias=encoder_bias,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                T5DecoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    layer_norm_epsilon=layer_norm_epsilon,
                    self_relative_bias=decoder_bias,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout)

    def encode(
        self,
        src: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.encoder_layers:
            output = layer(output, key_padding_mask=key_padding_mask)
        return self.dropout(self.encoder_norm(output))

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt
        for layer in self.decoder_layers:
            output = layer(
                output,
                memory,
                self_key_padding_mask=self_key_padding_mask,
                self_attn_mask=self_attn_mask,
                cross_key_padding_mask=cross_key_padding_mask,
            )
        return self.dropout(self.decoder_norm(output))
