import torch
import torch.nn as nn
import math


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed (non-learnable) sinusoidal positional embeddings from 'Attention Is All You Need'."""
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model // 2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it moves with .to(device) but is not a parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns positional embeddings for positions 0..seq_len-1."""
        return self.pe[:, :seq_len, :]  # (1, T, d_model)


class RotaryPositionalEmbedding(nn.Module):
    """Standard Rotary Position Embedding (RoPE).

    Applied to Q and K inside attention, not added to input embeddings.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Precompute frequency bands: theta_i = 1 / (base^(2i/d))
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq)  # (head_dim // 2,)

        # Precompute cos and sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=torch.float)  # (seq_len,)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, head_dim // 2)
        # Duplicate for pairing: [θ0, θ1, ..., θ0, θ1, ...]
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)  # (seq_len, head_dim)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        """Rotate the second half of the last dimension to pair with the first half."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        """
        Apply RoPE to Q and K.
        q, k: (B, T, n_heads, head_dim)
        Returns: rotated q, k with same shape
        """
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


def get_pos_embedding(config) -> nn.Module:
    """Factory returning the appropriate positional embedding module.

    For 'rope': returns a RotaryPositionalEmbedding (applied inside attention)
    For 'absolute': returns nn.Embedding (learnable, added to input)
    For 'sinusoidal': returns SinusoidalPositionalEmbedding (fixed, added to input)
    """
    if config.pos_emb == "absolute":
        return nn.Embedding(config.max_seq_len, config.d_model)
    elif config.pos_emb == "sinusoidal":
        return SinusoidalPositionalEmbedding(config.max_seq_len, config.d_model)
    elif config.pos_emb == "rope":
        head_dim = config.d_model // config.n_heads
        return RotaryPositionalEmbedding(head_dim, config.max_seq_len)
    else:
        raise ValueError(f"Unknown pos_emb '{config.pos_emb}'")
