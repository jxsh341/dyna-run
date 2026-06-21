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
    heterogeneous: bool = False
    expert_dims: list = None
    prune_interval: int = 0
    prune_threshold: float = 0.05

    def __post_init__(self):
        if self.n_experts < 1:
            raise ValueError("n_experts must be at least 1")
        if self.top_k < 1 or self.top_k > self.n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        if self.d_model < 1:
            raise ValueError("d_model must be at least 1")
        if self.d_ff < 1:
            raise ValueError("d_ff must be at least 1")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")
        if self.eval_capacity_factor <= 0:
            raise ValueError("eval_capacity_factor must be positive")
        if self.aux_loss_coef < 0:
            raise ValueError("aux_loss_coef must be non-negative")
        if self.expert_dims is not None:
            if len(self.expert_dims) != self.n_experts:
                raise ValueError("expert_dims must contain one dimension per expert")
            if any(dim < 1 for dim in self.expert_dims):
                raise ValueError("expert_dims must contain positive dimensions")
        if self.prune_interval < 0:
            raise ValueError("prune_interval must be non-negative")
        if self.prune_threshold < 0:
            raise ValueError("prune_threshold must be non-negative")


@dataclass
class ModelConfig:
    vocab_size: int = 512
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 256
    moe: MoEConfig = field(default_factory=MoEConfig)
    use_moe: bool = True

    def __post_init__(self):
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be at least 1")
        if self.d_model < 1:
            raise ValueError("d_model must be at least 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if self.n_heads < 1:
            raise ValueError("n_heads must be at least 1")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")
        if self.moe.d_model != self.d_model:
            raise ValueError("moe.d_model must match model d_model")
