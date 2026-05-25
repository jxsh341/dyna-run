import torch
import torch.nn as nn
from .router import Router
from .experts import Experts


class SparseMoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, d_ff: int, top_k: int = 2,
                 noisy_gating: bool = True, aux_loss_coef: float = 0.01):
        super().__init__()
        self.router = Router(d_model, n_experts, noisy_gating)
        self.experts = Experts(n_experts, d_model, d_ff)
        self.top_k = top_k
        self.n_experts = n_experts
        self.aux_loss_coef = aux_loss_coef

    def forward(self, x):
        gates, indices = self.router(x)
        out = self.experts(x, indices)
        aux_loss = self.router.load_balancing_loss(gates, indices)
        return out, gates, indices, aux_loss * self.aux_loss_coef

    def set_dense_mode(self, enabled: bool):
        self._dense_mode = enabled

    def forward_dense(self, x):
        all_outputs = []
        for expert in self.experts.experts:
            all_outputs.append(expert(x))
        out = torch.stack(all_outputs).mean(dim=0)
        return out
