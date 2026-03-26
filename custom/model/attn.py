import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pos_emb: str = "absolute", rope_module=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pos_emb = pos_emb
        self.rope = rope_module  # RotaryPositionalEmbedding instance (only when pos_emb == "rope")

        # Projections
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=True)
        self.c_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape

        # Project to q, k, v
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape into heads: (B, T, n_heads, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # Apply RoPE if configured
        if self.pos_emb == "rope" and self.rope is not None:
            q, k = self.rope(q, k, seq_len=T)

        out = self._attention(q, k, v, is_causal=True)
        out = out.reshape(B, T, self.d_model)
        return self.c_proj(out)

    def _attention(self, q, k, v, is_causal: bool):
        """
        q: (B, Tq, H, Dh)
        k: (B, Tk, H, Dh)
        v: (B, Tk, H, Dh)
        """
        q = q.transpose(1, 2)  # (B, H, Tq, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.,
            is_causal=is_causal
        )
        return out.transpose(1, 2)  # (B, Tq, H, Dh)
