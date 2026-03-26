"""
Test script for the customizable GPT-2 model.
Runs a forward pass and prints params + output shape.

Usage (from project root):
    python3.9 -m custom.test
"""

import torch
from custom.model.config import GPT2Config
from custom.model.gpt2 import GPT2


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


if __name__ == "__main__":
    config = GPT2Config.from_json("custom/config.json")
    print(f"\nConfig: {config}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2(config).to(device)
    model.eval()

    B, T = 2, 64
    x = torch.randint(0, config.vocab_size, (B, T), device=device)

    with torch.no_grad():
        logits = model(x)

    total, trainable = count_parameters(model)

    print(f"Model name  : {config.model_name}")
    print(f"Output shape: {logits.shape}")
    print(f"Total params: {format_params(total)}")
    print(f"Trainable   : {format_params(trainable)}")
