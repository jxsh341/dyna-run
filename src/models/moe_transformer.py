import torch
import torch.nn as nn
from src.core.config import ModelConfig
from src.core.sparse_moe import SparseMoE


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.weight


def precompute_rope(dim, max_len, theta=10000.0):
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float) / half))
    t = torch.arange(max_len, dtype=torch.float)
    angles = t[:, None] * freqs[None, :]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos[:x.shape[-2], :].view(1, 1, x.shape[-2], half)
    sin = sin[:x.shape[-2], :].view(1, 1, x.shape[-2], half)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        cos, sin = precompute_rope(self.head_dim, max_seq_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = attn + mask
        attn = attn.softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class MoETransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = Attention(config.d_model, config.n_heads, config.max_seq_len)
        self.moe = SparseMoE(
            d_model=config.d_model,
            n_experts=config.moe.n_experts,
            d_ff=config.moe.d_ff,
            top_k=config.moe.top_k,
            noisy_gating=config.moe.noisy_gating,
            aux_loss_coef=config.moe.aux_loss_coef,
            heterogeneous=config.moe.heterogeneous,
            expert_dims=config.moe.expert_dims,
        )
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        moe_out, gates, indices, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, gates, indices, aux_loss


class MoETransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [MoETransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, x):
        x = self.token_embedding(x)
        all_gates = []
        all_indices = []
        total_aux_loss = 0.0
        for block in self.blocks:
            x, gates, indices, aux_loss = block(x)
            all_gates.append(gates)
            all_indices.append(indices)
            total_aux_loss = total_aux_loss + aux_loss
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, all_gates, all_indices, total_aux_loss

    def generate(self, idx, max_new_tokens=50, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, gates, indices, aux_loss = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @property
    def device(self):
        return next(self.parameters()).device
