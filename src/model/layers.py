"""Core transformer layers, attention mechanisms, and positional encodings.

This module provides the building blocks for constructing Transformer models,
including several architectural variants like standard Transformers, T5-style
models with relative position biases, and custom models with features like
Rotary Positional Embeddings (RoPE) and Multi-Linear Attention (MLA).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding.

    This module injects information about the relative or absolute position of
    tokens in the sequence. The positional encodings have the same dimension as
    the embeddings so that they can be summed. The implementation is based on
    the original Transformer paper "Attention Is All You Need".

    The positional encodings are registered as a buffer and are not considered
    model parameters.
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        """Initializes the SinusoidalPE module.

        Args:
            d_model: The dimensionality of the embeddings.
            max_len: The maximum sequence length that this model might ever see.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x: The input tensor of embeddings.
               Shape: (batch_size, seq_len, d_model)

        Returns:
            The input tensor with added positional encodings.
            Shape: (batch_size, seq_len, d_model)
        """
        # Add positional encodings up to the sequence length of the input
        return x + self.pe[:, : x.size(1)]


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE).

    RoPE encodes absolute position information with a rotation matrix and naturally
    incorporates explicit relative position dependency in self-attention. It has
    become a popular choice in modern language models like LLaMA.

    This implementation pre-computes the sinusoidal frequencies and applies them
    to query and key projections within the attention module.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ) -> None:
        """Initializes the RotaryEmbedding module.

        Args:
            dim: The dimension of the rotary embeddings, typically `head_dim`.
            max_position_embeddings: The maximum sequence length for pre-computing
                the cache.
            base: The base for the sinusoidal frequencies.

        Raises:
            ValueError: If `dim` is not an even number.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding requires an even dimension.")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int) -> None:
        """Pre-computes and caches the cosine and sine matrices."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves or re-computes the cos/sin caches for a given sequence length.

        If `seq_len` exceeds the cached length, the cache is rebuilt with a
        slightly larger size to avoid frequent re-computation.

        Args:
            seq_len: The sequence length of the current batch.
            device: The device of the input tensor.
            dtype: The data type of the input tensor.

        Returns:
            A tuple containing the cosine and sine caches.
        """
        if seq_len > self.max_seq_len_cached:
            # Rebuild cache with some headroom
            self.max_seq_len_cached = int(seq_len * 1.2)
            self._build_cache(self.max_seq_len_cached)

        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half of the hidden dimensions."""
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Applies rotary positional embeddings to the input tensor.

    Args:
        tensor: The input tensor (query or key).
                Shape: (batch, n_heads, seq_len, head_dim)
        cos: The cosine cache. Shape: (seq_len, head_dim)
        sin: The sine cache. Shape: (seq_len, head_dim)

    Returns:
        The tensor with applied rotary embeddings.
    """
    # Reshape cos and sin to be broadcastable
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (tensor * cos) + (_rotate_half(tensor) * sin)


class ScaledDotProductAttention(nn.Module):
    """Computes scaled dot-product attention.

    This is the core component of the multi-head attention mechanism, calculating
    attention scores and producing the weighted sum of values.
    """

    def __init__(self, head_dim: int, dropout: float = 0.0) -> None:
        """Initializes the ScaledDotProductAttention module.

        Args:
            head_dim: The dimensionality of each attention head.
            dropout: The dropout rate applied to the attention weights.
        """
        super().__init__()
        self.scale = head_dim**-0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs the scaled dot-product attention.

        Args:
            query: Query tensor. Shape: (batch, n_heads, q_len, head_dim)
            key: Key tensor. Shape: (batch, n_heads, k_len, head_dim)
            value: Value tensor. Shape: (batch, n_heads, v_len, head_dim)
            attn_mask: Causal or other attention mask. Can be boolean or float.
                       Shape: (q_len, k_len) or broadcastable.
            key_padding_mask: Mask for padding tokens in the key sequence.
                              Shape: (batch, k_len)

        Returns:
            The output tensor after attention.
            Shape: (batch, n_heads, q_len, head_dim)
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            # Reshape for broadcasting: (batch, 1, 1, k_len)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                # Allows for float masks (e.g., from relative position biases)
                scores = scores + attn_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, value)


class MultiLinearAttention(nn.Module):
    """A lightweight multi-linear attention approximation.

    This module implements a feature map similar to Performer's, using `elu + 1`.
    It is intended for ablation experiments and offers a linear complexity
    alternative to standard softmax attention. It should not be used with masks
    that break linearity, such as causal masks.
    """

    def __init__(self, head_dim: int, dropout: float = 0.0) -> None:
        """Initializes the MultiLinearAttention module.

        Args:
            head_dim: The dimensionality of each attention head.
            dropout: The dropout rate applied to the output.
        """
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
        """Performs the multi-linear attention computation.

        Args:
            query: Query tensor. Shape: (batch, n_heads, q_len, head_dim)
            key: Key tensor. Shape: (batch, n_heads, k_len, head_dim)
            value: Value tensor. Shape: (batch, n_heads, v_len, head_dim)
            key_padding_mask: Mask for padding tokens in the key sequence.
                              Shape: (batch, k_len)

        Returns:
            The output tensor after attention.
            Shape: (batch, n_heads, q_len, head_dim)
        """
        q = self.feature_map(query)
        k = self.feature_map(key)

        if key_padding_mask is not None:
            valid = (~key_padding_mask).unsqueeze(1).unsqueeze(-1).to(query.dtype)
            k = k * valid

        kv = torch.einsum("bhsd,bhsf->bhdf", k, value)
        z = torch.einsum("bhtd,bhd->bht", q, k.sum(dim=2)) + self.eps
        out = torch.einsum("bhtd,bhdf->bhtf", q, kv) / z.unsqueeze(-1)
        return self.dropout(out)


class HybridAttention(nn.Module):
    """A flexible multi-head attention module.

    This module combines several features:
    - Standard scaled dot-product attention.
    - Optional Rotary Positional Embeddings (RoPE).
    - Optional interpolation with Multi-Linear Attention (MLA) for efficiency.
    """

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
        """Initializes the HybridAttention module.

        Args:
            d_model: The total dimensionality of the input.
            num_heads: The number of attention heads.
            dropout: The dropout rate.
            use_mla: Whether to enable the MLA variant.
            interleave_ratio: The ratio to interpolate between standard attention
                and MLA. If > 0, MLA is used.
            use_rope: Whether to enable Rotary Positional Embeddings.
            max_position_embeddings: Max sequence length for RoPE cache.

        Raises:
            ValueError: If d_model is not divisible by num_heads.
        """
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
        """Splits the last dimension into (num_heads, head_dim)."""
        batch, seq_len, dim = x.size()
        head_dim = dim // num_heads
        return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        """Merges the head and dimension axes back into a single dimension."""
        batch, heads, seq_len, dim = x.size()
        return x.transpose(1, 2).reshape(batch, seq_len, heads * dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs the hybrid attention forward pass.

        Args:
            query: Query tensor. Shape: (batch, q_len, d_model)
            key: Key tensor. Shape: (batch, k_len, d_model)
            value: Value tensor. Shape: (batch, v_len, d_model)
            attn_mask: Causal or other attention mask.
            key_padding_mask: Mask for padding tokens in the key sequence.

        Returns:
            The output tensor. Shape: (batch, q_len, d_model)
        """
        # Project and reshape Q, K, V
        q = self._reshape_heads(self.q_proj(query), self.num_heads)
        k = self._reshape_heads(self.k_proj(key), self.num_heads)
        v = self._reshape_heads(self.v_proj(value), self.num_heads)

        # Apply RoPE if enabled
        if self.use_rope and self.rope is not None:
            cos_q, sin_q = self.rope.get_cos_sin(q.size(-2), q.device, q.dtype)
            cos_k, sin_k = self.rope.get_cos_sin(k.size(-2), k.device, k.dtype)
            q = apply_rotary_pos_emb(q, cos_q, sin_q)
            k = apply_rotary_pos_emb(k, cos_k, sin_k)

        # Standard attention is always computed as a base
        dot_out = self.scaled_dot(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # MLA is only used if enabled, no causal mask is present, and ratio > 0
        use_mla = self.use_mla and self.interleave_ratio > 0.0 and attn_mask is None and self.mla is not None
        if use_mla:
            mla_out = self.mla(q, k, v, key_padding_mask=key_padding_mask)
            ratio = max(0.0, min(1.0, self.interleave_ratio))
            # Interpolate between standard and linear attention
            attn_out = (1 - ratio) * dot_out + ratio * mla_out
        else:
            attn_out = dot_out

        attn_out = self._merge_heads(attn_out)
        return self.out_proj(attn_out)


class PositionwiseFeedForward(nn.Module):
    """A standard position-wise feed-forward network."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """A single layer of the custom Transformer encoder."""

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
        # Pre-LN is more common in modern transformers, but this follows Post-LN
        # from the original paper, consistent with nn.Transformer.
        # x -> Self-Attention -> Add & Norm
        x = src
        attn_out = self.self_attn(
            x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-Forward -> Add & Norm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class TransformerDecoderLayer(nn.Module):
    """A single layer of the custom Transformer decoder."""

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
            d_model, nhead, dropout, use_rope=use_rope, max_position_embeddings=max_position_embeddings,
        )
        self.cross_attn = HybridAttention(
            d_model, nhead, dropout, use_mla=use_mla, interleave_ratio=interleave_ratio,
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
        x = tgt
        # Masked Self-Attention
        attn_out = self.self_attn(
            x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = self.norm1(x + self.dropout1(attn_out))

        # Cross-Attention
        cross_attn_out = self.cross_attn(
            x, memory, memory, attn_mask=None, key_padding_mask=memory_key_padding_mask
        )
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # Feed-Forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


class CustomTransformer(nn.Module):
    """A custom Transformer encoder-decoder stack.

    This module provides an interface similar to `nn.Transformer` but is built
    using the custom `TransformerEncoderLayer` and `TransformerDecoderLayer`,
    allowing for features like RoPE and MLA.
    """

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
                    d_model, nhead, dim_feedforward, dropout,
                    use_mla=use_mla, interleave_ratio=interleave_ratio,
                    use_rope=use_rope, max_position_embeddings=max_position_embeddings,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, nhead, dim_feedforward, dropout,
                    use_mla=use_mla, interleave_ratio=interleave_ratio,
                    use_rope=use_rope, max_position_embeddings=max_position_embeddings,
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
                output, memory,
                tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.decoder_norm(output)


# ==============================================================================
# T5-Style Components
# ==============================================================================


class RelativePositionBias(nn.Module):
    """Implements T5-style relative position bias with bucketing.

    This allows the model to learn position-dependent attention biases without
    relying on absolute positional encodings.
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        *,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if num_buckets <= 0 or num_heads <= 0:
            raise ValueError("num_buckets and num_heads must be positive")
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        # Learnable embedding for each bucket and head
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Calculates the bucket index for each relative position."""
        num_buckets = self.num_buckets
        max_dist = self.max_distance
        relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)

        if self.bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = relative_position.abs()
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # For positions beyond max_exact, use a logarithmic scale
        log_ratio = math.log(max_dist / max_exact) if max_dist > max_exact else 1.0
        large_pos = max_exact + (
            torch.log(relative_position.float() / max_exact + 1e-6) / log_ratio
        ) * (num_buckets - max_exact)
        large_pos = torch.min(large_pos.long(), torch.full_like(large_pos, num_buckets - 1))

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
        """Computes the relative position bias tensor.

        Returns:
            A tensor of shape (1, num_heads, query_length, key_length)
        """
        context_pos = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_pos = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_pos - context_pos

        bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(bucket)
        # Reshape to (1, num_heads, query_length, key_length)
        return values.permute(2, 0, 1).unsqueeze(0).to(dtype)


class T5Attention(nn.Module):
    """T5-style multi-head self-attention.

    This version uses pre-LayerNorm (as is standard in T5) and supports
    relative position biases. Projections (q, k, v, out) are bias-free.
    """

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
        """Splits the last dimension into (num_heads, head_dim)."""
        batch, seq_len, dim = tensor.size()
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, tensor: torch.Tensor) -> torch.Tensor:
        """Merges the head and dimension axes back into a single dimension."""
        batch, heads, seq_len, dim = tensor.size()
        return tensor.transpose(1, 2).reshape(batch, seq_len, heads * dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_value_states: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs the T5 attention forward pass.

        Args:
            hidden_states: Input tensor. Shape: (batch, seq_len, d_model)
            key_value_states: Optional key/value states for cross-attention.
            key_padding_mask: Mask for padding tokens.
            position_bias: Pre-computed position bias tensor.

        Returns:
            The output tensor.
        """
        is_cross_attention = key_value_states is not None
        query = self._reshape(self.q(hidden_states))

        if is_cross_attention:
            key = self._reshape(self.k(key_value_states))
            value = self._reshape(self.v(key_value_states))
        else: # Self-attention
            key = self._reshape(self.k(hidden_states))
            value = self._reshape(self.v(hidden_states))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply position bias if available
        if self.relative_bias is not None and position_bias is None:
            position_bias = self.relative_bias(
                query.size(-2), key.size(-2), device=query.device, dtype=query.dtype
            )
        if position_bias is not None:
            scores += position_bias

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, K_len)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = self._merge(torch.matmul(attn_weights, value))
        return self.out(attn_output)


class T5FeedForward(nn.Module):
    """T5-style gated feed-forward network (Gated GELU)."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        # T5 uses a wider intermediate layer for the gating mechanism
        self.wi_0 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.wi_1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.wo = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.wi_0(x))
        hidden = self.wi_1(x)
        gated = gate * hidden
        return self.dropout(self.wo(gated))


class T5EncoderLayer(nn.Module):
    """A single layer of the T5 encoder, using Pre-LayerNorm."""

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
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.feed_forward = T5FeedForward(d_model, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN: Norm -> Attention -> Add
        normed = self.norm1(hidden_states)
        attn_out = self.self_attn(normed, key_padding_mask=key_padding_mask)
        x = hidden_states + self.dropout(attn_out)

        # Pre-LN: Norm -> FF -> Add
        normed_ff = self.norm2(x)
        ff_out = self.feed_forward(normed_ff)
        x = x + self.dropout(ff_out)
        return x


class T5DecoderLayer(nn.Module):
    """A single layer of the T5 decoder, using Pre-LayerNorm."""

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
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.feed_forward = T5FeedForward(d_model, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        *,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Masked Self-Attention (Pre-LN)
        normed = self.norm1(hidden_states)
        self_attn_out = self.self_attn(normed, key_padding_mask=self_key_padding_mask)
        x = hidden_states + self.dropout(self_attn_out)

        # Cross-Attention (Pre-LN)
        normed_cross = self.norm2(x)
        cross_out = self.cross_attn(
            normed_cross,
            key_value_states=memory,
            key_padding_mask=cross_key_padding_mask,
        )
        x = x + self.dropout(cross_out)

        # Feed-Forward (Pre-LN)
        normed_ff = self.norm3(x)
        ff_out = self.feed_forward(normed_ff)
        x = x + self.dropout(ff_out)
        return x


class T5Transformer(nn.Module):
    """A T5-style Transformer encoder-decoder stack."""

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
        # Encoder uses bidirectional relative positions, decoder uses unidirectional
        encoder_bias = RelativePositionBias(
            nhead, relative_attention_num_buckets, relative_attention_max_distance, bidirectional=True
        )
        decoder_bias = RelativePositionBias(
            nhead, relative_attention_num_buckets, relative_attention_max_distance, bidirectional=False
        )
        self.encoder_layers = nn.ModuleList(
            [
                T5EncoderLayer(
                    d_model, nhead, dim_feedforward, dropout,
                    layer_norm_epsilon=layer_norm_epsilon, relative_bias=encoder_bias
                ) for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                T5DecoderLayer(
                    d_model, nhead, dim_feedforward, dropout,
                    layer_norm_epsilon=layer_norm_epsilon, self_relative_bias=decoder_bias
                ) for _ in range(num_decoder_layers)
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
        # Final LayerNorm after the stack
        output = self.encoder_norm(output)
        return self.dropout(output)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt
        for layer in self.decoder_layers:
            output = layer(
                output, memory,
                self_key_padding_mask=self_key_padding_mask,
                cross_key_padding_mask=cross_key_padding_mask
            )
        # Final LayerNorm after the stack
        output = self.decoder_norm(output)
        return self.dropout(output)