import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU gated activation: SiLU(x_gate) * x_value"""
    def forward(self, x):
        x_gate, x_value = x.chunk(2, dim=-1)
        return F.silu(x_gate) * x_value


class GeGLU(nn.Module):
    """GeGLU gated activation: GELU(x_gate) * x_value"""
    def forward(self, x):
        x_gate, x_value = x.chunk(2, dim=-1)
        return F.gelu(x_gate) * x_value


class ReLUActivation(nn.Module):
    def forward(self, x):
        return F.relu(x)


class GELUActivation(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class SiLUActivation(nn.Module):
    def forward(self, x):
        return F.silu(x)


def get_activation(name: str) -> nn.Module:
    """Factory function returning the requested activation module."""
    activations = {
        "relu": ReLUActivation,
        "gelu": GELUActivation,
        "silu": SiLUActivation,
        "swiglu": SwiGLU,
        "geglu": GeGLU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations.keys())}")
    return activations[name]()
