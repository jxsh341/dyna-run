from dataclasses import dataclass, field


@dataclass
class MoEConfig:
    n_experts: int = 8
    top_k: int = 2
    d_model: int = 128
    d_ff: int = 256
    capacity_factor: float = 1.25
    eval_capacity_factor: float = 2.0
    noisy_gating: bool = True
    aux_loss_coef: float = 0.01


@dataclass
class ModelConfig:
    vocab_size: int = 512
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 256
    moe: MoEConfig = field(default_factory=MoEConfig)
    use_moe: bool = True
