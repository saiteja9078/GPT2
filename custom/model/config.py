import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class GPT2Config:
    # Architecture hyperparams
    vocab_size: int = 50_257
    max_seq_len: int = 1024
    d_model: int = 768
    n_heads: int = 12
    n_blocks: int = 12

    # Ablation knobs
    pos_emb: str = "absolute"       # "absolute" | "sinusoidal" | "rope"
    activation: str = "gelu"        # "gelu" | "relu" | "silu" | "swiglu" | "geglu"
    norm_type: str = "layernorm"    # "layernorm" | "rmsnorm"
    norm_position: str = "pre"      # "pre" | "post"

    # Validation
    _valid_pos_emb = {"absolute", "sinusoidal", "rope"}
    _valid_activation = {"gelu", "relu", "silu", "swiglu", "geglu"}
    _valid_norm_type = {"layernorm", "rmsnorm"}
    _valid_norm_position = {"pre", "post"}

    def __post_init__(self):
        assert self.pos_emb in self._valid_pos_emb, \
            f"pos_emb must be one of {self._valid_pos_emb}, got '{self.pos_emb}'"
        assert self.activation in self._valid_activation, \
            f"activation must be one of {self._valid_activation}, got '{self.activation}'"
        assert self.norm_type in self._valid_norm_type, \
            f"norm_type must be one of {self._valid_norm_type}, got '{self.norm_type}'"
        assert self.norm_position in self._valid_norm_position, \
            f"norm_position must be one of {self._valid_norm_position}, got '{self.norm_position}'"

    @property
    def model_name(self) -> str:
        """Generate a descriptive model name from config for saving."""
        return f"gpt2_{self.pos_emb}_{self.activation}_{self.norm_type}_{self.norm_position}"

    @property
    def is_gated_activation(self) -> bool:
        """SwiGLU and GeGLU are gated activations that split the hidden dim."""
        return self.activation in {"swiglu", "geglu"}

    @classmethod
    def from_json(cls, path: str) -> "GPT2Config":
        with open(path, "r") as f:
            data = json.load(f)
        # Filter keys to only those that are fields in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values() if not f.name.startswith("_")}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_json(self, path: str):
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def __repr__(self):
        return (
            f"GPT2Config(\n"
            f"  model_name={self.model_name},\n"
            f"  vocab_size={self.vocab_size}, max_seq_len={self.max_seq_len},\n"
            f"  d_model={self.d_model}, n_heads={self.n_heads}, n_blocks={self.n_blocks},\n"
            f"  pos_emb={self.pos_emb}, activation={self.activation},\n"
            f"  norm_type={self.norm_type}, norm_position={self.norm_position}\n"
            f")"
        )
