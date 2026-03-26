import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


def get_norm(norm_type: str, d_model: int) -> nn.Module:
    """Factory function returning the requested normalization module."""
    if norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    elif norm_type == "rmsnorm":
        return RMSNorm(d_model)
    else:
        raise ValueError(f"Unknown norm_type '{norm_type}'. Choose from ['layernorm', 'rmsnorm']")
