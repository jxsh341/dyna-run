import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, d_model: int, n_experts: int, noisy_gating: bool = True):
        super().__init__()
        self.n_experts = n_experts
        self.noisy_gating = noisy_gating
        self.w_gate = nn.Linear(d_model, n_experts, bias=False)
        if noisy_gating:
            self.w_noise = nn.Linear(d_model, n_experts, bias=False)
            self._init_noise()

    def _init_noise(self):
        nn.init.zeros_(self.w_noise.weight)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _top_k_gating(self, x):
        gate_logits = self.w_gate(x)
        if self.training and self.noisy_gating:
            noise = self.w_noise(x)
            noise_std = F.softplus(noise) + 1e-6
            noisy_logits = gate_logits + torch.randn_like(gate_logits) * noise_std
            gates = F.softmax(noisy_logits, dim=-1)
        else:
            gates = F.softmax(gate_logits, dim=-1)
        top_k_values, top_k_indices = torch.topk(gates, k=min(self.n_experts, 2), dim=-1)
        mask = F.one_hot(top_k_indices, num_classes=self.n_experts).float()
        mask = mask.sum(dim=-2)
        gates = gates * mask
        gates_sum = gates.sum(dim=-1, keepdim=True)
        gates = gates / (gates_sum + 1e-10)
        return gates, top_k_indices, gate_logits

    def forward(self, x):
        gates, indices, logits = self._top_k_gating(x)
        return gates, indices

    def load_balancing_loss(self, gates, indices, tokens_per_expert=None):
        if tokens_per_expert is None:
            tokens_per_expert = torch.zeros(self.n_experts, device=indices.device)
            for e in range(self.n_experts):
                tokens_per_expert[e] = (indices == e).any(dim=-1).sum().float()
        total_tokens = tokens_per_expert.sum()
        expert_fraction = tokens_per_expert / (total_tokens + 1e-10)
        router_prob = gates.mean(dim=(0, 1))
        loss = (expert_fraction * router_prob).sum() * self.n_experts
        return loss
