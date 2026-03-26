import torch
import torch.nn as nn
from custom.model.config import GPT2Config
from custom.model.transformer import TransformerBlock
from custom.model.norms import get_norm
from custom.model.positional import get_pos_embedding


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len

        # Token embedding
        self.wte = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embedding
        self.pos_emb_module = get_pos_embedding(config)

        # For RoPE, the module is shared across all transformer blocks
        rope_module = self.pos_emb_module if config.pos_emb == "rope" else None

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config, rope_module=rope_module)
            for _ in range(config.n_blocks)
        ])

        # Final layer norm
        self.ln_f = get_norm(config.norm_type, config.d_model)

        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.wte.weight

    def forward(self, x):
        B, T = x.shape
        assert T <= self.max_seq_len, \
            f"Cannot forward sequence of length {T}, max_seq_len is {self.max_seq_len}"

        tok_emb = self.wte(x)  # (B, T, d_model)

        if self.config.pos_emb == "absolute":
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
            pos_emb = self.pos_emb_module(pos)  # (T, d_model)
            x = tok_emb + pos_emb
        elif self.config.pos_emb == "sinusoidal":
            pos_emb = self.pos_emb_module(T)  # (1, T, d_model)
            x = tok_emb + pos_emb
        elif self.config.pos_emb == "rope":
            # RoPE is applied inside attention, not added to embeddings
            x = tok_emb

        for layer in self.transformer_blocks:
            x = layer(x)

        x = self.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab_size)

    def save_model(self, save_dir: str):
        """Save model weights and config using the auto-generated model name."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        model_name = self.config.model_name

        # Save weights
        weights_path = os.path.join(save_dir, f"{model_name}.pt")
        torch.save(self.state_dict(), weights_path)

        # Save config alongside
        config_path = os.path.join(save_dir, f"{model_name}_config.json")
        self.config.to_json(config_path)

        print(f"Model saved: {weights_path}")
        print(f"Config saved: {config_path}")
        return weights_path, config_path

    @classmethod
    def load_model(cls, config_path: str, weights_path: str, device: str = "cpu"):
        """Load model from saved config and weights."""
        config = GPT2Config.from_json(config_path)
        model = cls(config).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        return model
