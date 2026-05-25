import torch
import torch.nn as nn
from src.core.config import ModelConfig, MoEConfig
from src.core.dense_block import DenseBlock
from src.models.moe_transformer import RMSNorm, Attention, precompute_rope, apply_rope


class DenseTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        total_expert_params = config.moe.n_experts * config.moe.d_ff
        self.attn = Attention(config.d_model, config.n_heads, config.max_seq_len)
        self.ffn = DenseBlock(config.d_model, total_expert_params)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DenseTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [DenseTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, x):
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens=50, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @property
    def device(self):
        return next(self.parameters()).device
