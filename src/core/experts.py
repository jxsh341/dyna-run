import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        hidden_dim = 2 * d_ff
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            SwiGLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class Experts(nn.Module):
    def __init__(self, n_experts: int, d_model: int, d_ff: int, expert_dims: list = None):
        super().__init__()
        dims = expert_dims if expert_dims else [d_ff] * n_experts
        self.experts = nn.ModuleList(
            [Expert(d_model, dims[i]) for i in range(n_experts)]
        )
        self.n_experts = n_experts
        self.expert_dims = dims

    def forward(self, x, expert_indices):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        indices_flat = expert_indices.view(-1, expert_indices.shape[-1])
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices_flat == i).any(dim=-1)
            if mask.any():
                out[mask] = out[mask] + expert(x_flat[mask])
        return out.view(batch_size, seq_len, d_model)
