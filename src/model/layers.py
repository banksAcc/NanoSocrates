"""Core transformer layers and positional encodings."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE).

    The implementation follows the formulation used in GPT-NeoX/LLama style
    models where cos/sin caches are generated lazily and applied to query/key
    projections inside the attention module.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding requires an even dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()
        sin_cached = emb.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def get_cos_sin(self, seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = int(seq_len * 1.1)
            self._build_cache(self.max_seq_len_cached)
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (tensor * cos) + (_rotate_half(tensor) * sin)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.scale = head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, value)


class MultiLinearAttention(nn.Module):
    """A lightweight multi-linear attention approximation.

    The module implements an "elu + 1" feature map similar to Performer. It is
    primarily intended for ablation experiments; when masks that would break
    the linearity (e.g. causal masks) are provided the caller should fall back
    to standard attention.
    """

    def __init__(self, head_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.feature_map = lambda x: F.elu(x) + 1
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.feature_map(query)
        k = self.feature_map(key)
        v = value
        if key_padding_mask is not None:
            valid = (~key_padding_mask).unsqueeze(1).unsqueeze(-1).to(query.dtype)
            k = k * valid
            v = v * valid
        kv = torch.einsum("bhsd,bhsf->bhdf", k, v)
        z = torch.einsum("bhtd,bhd->bht", q, k.sum(dim=2)) + self.eps
        out = torch.einsum("bhtd,bhdf->bhtf", q, kv)
        out = out / z.unsqueeze(-1)
        return self.dropout(out)


class HybridAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        use_mla: bool = False,
        interleave_ratio: float = 0.0,
        use_rope: bool = False,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_mla = use_mla
        self.interleave_ratio = float(interleave_ratio)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scaled_dot = ScaledDotProductAttention(self.head_dim, dropout)
        self.mla = MultiLinearAttention(self.head_dim, dropout) if use_mla else None
        self.use_rope = use_rope
        self.rope = (
            RotaryEmbedding(self.head_dim, max_position_embeddings=max_position_embeddings)
            if use_rope
            else None
        )

    @staticmethod
    def _reshape_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch, seq_len, dim = x.size()
        head_dim = dim // num_heads
        return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, dim = x.size()
        return x.transpose(1, 2).reshape(batch, seq_len, heads * dim)

    @staticmethod
    def _prepare_attn_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() == 2:
            return mask.unsqueeze(0).unsqueeze(0)
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        if mask.dim() != 4:
            raise ValueError("Unsupported attention mask dimensions")
        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask = self._prepare_attn_mask(attn_mask)
        q = self._reshape_heads(self.q_proj(query), self.num_heads)
        k = self._reshape_heads(self.k_proj(key), self.num_heads)
        v = self._reshape_heads(self.v_proj(value), self.num_heads)

        if self.use_rope and self.rope is not None:
            cos_q, sin_q = self.rope.get_cos_sin(q.size(-2), q.device, q.dtype)
            cos_k, sin_k = self.rope.get_cos_sin(k.size(-2), k.device, k.dtype)
            q = apply_rotary_pos_emb(q, cos_q, sin_q)
            k = apply_rotary_pos_emb(k, cos_k, sin_k)

        dot_out = self.scaled_dot(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        use_mla = self.use_mla and self.interleave_ratio > 0.0 and attn_mask is None and self.mla is not None
        if use_mla:
            mla_out = self.mla(q, k, v, key_padding_mask=key_padding_mask)
            ratio = max(0.0, min(1.0, self.interleave_ratio))
            if ratio >= 1.0:
                attn_out = mla_out
            elif ratio <= 0.0:
                attn_out = dot_out
            else:
                attn_out = (1 - ratio) * dot_out + ratio * mla_out
        else:
            attn_out = dot_out

        attn_out = self._merge_heads(attn_out)
        return self.out_proj(attn_out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        use_mla: bool,
        interleave_ratio: float,
        use_rope: bool,
        max_position_embeddings: int,
    ) -> None:
        super().__init__()
        self.self_attn = HybridAttention(
            d_model,
            nhead,
            dropout,
            use_mla=use_mla,
            interleave_ratio=interleave_ratio,
            use_rope=use_rope,
            max_position_embeddings=max_position_embeddings,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src = src + self.dropout1(
            self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        )
        src = self.norm1(src)
        src = src + self.dropout2(self.ff(src))
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        use_mla: bool,
        interleave_ratio: float,
        use_rope: bool,
        max_position_embeddings: int,
    ) -> None:
        super().__init__()
        self.self_attn = HybridAttention(
            d_model,
            nhead,
            dropout,
            use_mla=use_mla,
            interleave_ratio=interleave_ratio,
            use_rope=use_rope,
            max_position_embeddings=max_position_embeddings,
        )
        self.cross_attn = HybridAttention(
            d_model,
            nhead,
            dropout,
            use_mla=use_mla,
            interleave_ratio=interleave_ratio,
            use_rope=use_rope,
            max_position_embeddings=max_position_embeddings,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt = tgt + self.dropout1(
            self.self_attn(
                tgt,
                tgt,
                tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )
        )
        tgt = self.norm1(tgt)
        tgt = tgt + self.dropout2(
            self.cross_attn(
                tgt,
                memory,
                memory,
                attn_mask=None,
                key_padding_mask=memory_key_padding_mask,
            )
        )
        tgt = self.norm2(tgt)
        tgt = tgt + self.dropout3(self.ff(tgt))
        tgt = self.norm3(tgt)
        return tgt


class CustomTransformer(nn.Module):
    """Minimal transformer stack used for optional model variants."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        *,
        use_mla: bool,
        interleave_ratio: float,
        use_rope: bool,
        max_position_embeddings: int,
    ) -> None:
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    use_mla=use_mla,
                    interleave_ratio=interleave_ratio,
                    use_rope=use_rope,
                    max_position_embeddings=max_position_embeddings,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    use_mla=use_mla,
                    interleave_ratio=interleave_ratio,
                    use_rope=use_rope,
                    max_position_embeddings=max_position_embeddings,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.encoder_layers:
            output = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.encoder_norm(output)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt
        for layer in self.decoder_layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.decoder_norm(output)

