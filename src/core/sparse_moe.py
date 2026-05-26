import torch
import torch.nn as nn
from .router import Router
from .experts import Experts


class SparseMoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, d_ff: int, top_k: int = 2,
                 noisy_gating: bool = True, aux_loss_coef: float = 0.01,
                 heterogeneous: bool = False, expert_dims: list = None):
        super().__init__()
        self.router = Router(d_model, n_experts, noisy_gating)
        self.experts = Experts(n_experts, d_model, d_ff, expert_dims=expert_dims)
        self.top_k = top_k
        self.n_experts = n_experts
        self.aux_loss_coef = aux_loss_coef
        self.heterogeneous = heterogeneous
        self.pruning_mask = None
        if heterogeneous:
            self.expert_bias = nn.Parameter(torch.zeros(n_experts))
        else:
            self.register_buffer("expert_bias", torch.zeros(n_experts))

    def forward(self, x):
        gates, indices = self.router(x)
        if self.heterogeneous:
            gates = gates * (1 + self.expert_bias)
            gate_sum = gates.sum(dim=-1, keepdim=True)
            gates = gates / (gate_sum + 1e-10)
            _, indices = torch.topk(gates, k=self.top_k, dim=-1)
        out = self.experts(x, indices)
        aux_loss = self.router.load_balancing_loss(gates, indices)
        return out, gates, indices, aux_loss * self.aux_loss_coef

    def set_pruning_mask(self, active_experts):
        self.router.set_expert_mask(active_experts)

    def clear_pruning_mask(self):
        self.router.clear_expert_mask()
        self.pruning_mask = None

    def set_dense_mode(self, enabled: bool):
        self._dense_mode = enabled

    def forward_dense(self, x):
        all_outputs = []
        for expert in self.experts.experts:
            all_outputs.append(expert(x))
        out = torch.stack(all_outputs).mean(dim=0)
        return out
