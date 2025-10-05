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


class RelativePositionBias(nn.Module):
    """Implements T5-style relative position bias with bucketing."""

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        *,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.num_heads = int(num_heads)
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.bidirectional = bool(bidirectional)
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = max(1, self.max_distance)
        relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)

        if self.bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = relative_position.abs()
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = max(1, num_buckets // 2)
        is_small = relative_position < max_exact
        log_ratio = math.log(max_distance / max_exact) if max_distance > max_exact else 1.0
        large_pos = max_exact + (
            torch.log(relative_position.float() / max_exact + 1e-6) / log_ratio
        ) * (num_buckets - max_exact)
        large_pos = large_pos.long()
        large_pos = torch.min(large_pos, torch.full_like(large_pos, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, large_pos)
        return relative_buckets

    def forward(
        self,
        query_length: int,
        key_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(bucket)
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values.to(dtype)


class T5Attention(nn.Module):
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
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.relative_bias = relative_bias
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = tensor.size()
        tensor = tensor.view(batch, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _merge(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, dim = tensor.size()
        return tensor.transpose(1, 2).reshape(batch, seq_len, heads * dim)

    def _prepare_attention_mask(self, mask: Optional[torch.Tensor], target: torch.Tensor) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() != 4:
            raise ValueError("Unsupported attention mask dimensions")
        if mask.size(-1) != target.size(-1):
            mask = mask.expand(-1, -1, -1, target.size(-1))
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self._reshape(self.q(hidden_states))
        if key_value_states is None:
            key_states = hidden_states
            value_states = hidden_states
        else:
            key_states = key_value_states
            value_states = key_value_states
        key = self._reshape(self.k(key_states))
        value = self._reshape(self.v(value_states))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.relative_bias is not None and position_bias is None:
            position_bias = self.relative_bias(
                query_length=query.size(-2),
                key_length=key.size(-2),
                device=query.device,
                dtype=query.dtype,
            )
        if position_bias is not None:
            scores = scores + position_bias

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            attn_mask = self._prepare_attention_mask(attention_mask, scores)
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + attn_mask.to(scores.dtype)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = self._merge(attn_output)
        return self.out(attn_output)


class T5FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.wi = nn.Linear(d_model, dim_feedforward * 2, bias=False)
        self.wo = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wi = self.wi(x)
        gate, value = wi.chunk(2, dim=-1)
        gated = F.gelu(gate) * value
        return self.dropout(self.wo(gated))


class T5EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        layer_norm_epsilon: float,
        relative_bias: Optional[RelativePositionBias],
    ) -> None:
        super().__init__()
        self.self_attn = T5Attention(
            d_model,
            nhead,
            dropout,
            relative_bias=relative_bias,
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.feed_forward = T5FeedForward(d_model, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)
        self.relative_bias = relative_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        position_bias = None
        if self.relative_bias is not None:
            position_bias = self.relative_bias(
                query_length=normed.size(1),
                key_length=normed.size(1),
                device=normed.device,
                dtype=normed.dtype,
            )
        attn_out = self.self_attn(
            normed,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attn_out)
        hidden_states = hidden_states + self.dropout(self.feed_forward(self.ff_layer_norm(hidden_states)))
        return hidden_states


class T5DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        layer_norm_epsilon: float,
        self_relative_bias: Optional[RelativePositionBias],
    ) -> None:
        super().__init__()
        self.self_attn = T5Attention(
            d_model,
            nhead,
            dropout,
            relative_bias=self_relative_bias,
        )
        self.cross_attn = T5Attention(d_model, nhead, dropout, relative_bias=None)
        self.self_attn_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.cross_attn_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.feed_forward = T5FeedForward(d_model, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)
        self.relative_bias = self_relative_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        *,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.self_attn_layer_norm(hidden_states)
        position_bias = None
        if self.relative_bias is not None:
            position_bias = self.relative_bias(
                query_length=normed.size(1),
                key_length=normed.size(1),
                device=normed.device,
                dtype=normed.dtype,
            )
        self_attn_out = self.self_attn(
            normed,
            attention_mask=self_attn_mask,
            key_padding_mask=self_key_padding_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(self_attn_out)

        normed_cross = self.cross_attn_layer_norm(hidden_states)
        cross_out = self.cross_attn(
            normed_cross,
            key_value_states=memory,
            key_padding_mask=cross_key_padding_mask,
        )
        hidden_states = hidden_states + self.dropout(cross_out)

        hidden_states = hidden_states + self.dropout(self.feed_forward(self.ff_layer_norm(hidden_states)))
        return hidden_states


class T5Transformer(nn.Module):
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
            num_buckets=relative_attention_num_buckets,
            max_distance=relative_attention_max_distance,
            bidirectional=True,
        )
        decoder_bias = RelativePositionBias(
            nhead,
            num_buckets=relative_attention_num_buckets,
            max_distance=relative_attention_max_distance,
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
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.encoder_layers:
            output = layer(
                output,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
            )
        output = self.encoder_norm(output)
        return self.dropout(output)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt
        for layer in self.decoder_layers:
            output = layer(
                output,
                memory,
                self_attn_mask=self_attn_mask,
                self_key_padding_mask=self_key_padding_mask,
                cross_key_padding_mask=cross_key_padding_mask,
            )
        output = self.decoder_norm(output)
        return self.dropout(output)

