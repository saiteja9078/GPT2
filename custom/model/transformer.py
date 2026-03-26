import torch
import torch.nn as nn
import torch.nn.functional as F
from custom.model.attn import MultiHeadAttention
from custom.model.activations import get_activation
from custom.model.norms import get_norm


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, activation_name: str = "gelu"):
        super().__init__()
        self.is_gated = activation_name in {"swiglu", "geglu"}
        self.activation = get_activation(activation_name)

        if self.is_gated:
            # For gated activations, scale hidden_dim to 2/3 so total MLP params
            # match non-gated variants (same trick as LLaMA/PaLM).
            # Non-gated MLP params:  2 * d_model * hidden_dim
            # Gated MLP params:      3 * d_model * gated_hidden_dim
            # Setting gated_hidden_dim = 2/3 * hidden_dim equalizes them.
            gated_hidden_dim = (2 * hidden_dim) // 3
            self.c_fc = nn.Linear(d_model, 2 * gated_hidden_dim, bias=True)
            self.c_proj = nn.Linear(gated_hidden_dim, d_model, bias=True)
        else:
            self.c_fc = nn.Linear(d_model, hidden_dim, bias=True)
            self.c_proj = nn.Linear(hidden_dim, d_model, bias=True)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, config, rope_module=None):
        super().__init__()
        d_model = config.d_model
        hidden_dim = 4 * d_model
        self.norm_position = config.norm_position

        # Normalization layers
        self.ln_1 = get_norm(config.norm_type, d_model)
        self.ln_2 = get_norm(config.norm_type, d_model)

        # Attention
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=config.n_heads,
            pos_emb=config.pos_emb,
            rope_module=rope_module,
        )

        # Feed-forward
        self.mlp = MLP(d_model, hidden_dim, config.activation)

    def forward(self, x):
        if self.norm_position == "pre":
            # Pre-norm: norm → sublayer → residual (GPT-2 style)
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            # Post-norm: sublayer → residual → norm (original Transformer style)
            x = self.ln_1(x + self.attn(x))
            x = self.ln_2(x + self.mlp(x))
        return x
