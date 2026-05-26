import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.core.config import ModelConfig


class TextDataset(Dataset):
    def __init__(self, text, seq_len, vocab_size=512):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars[:vocab_size])}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = min(vocab_size, len(self.stoi))
        self.seq_len = seq_len
        self.data = [self.stoi.get(c, 0) for c in text if c in self.stoi]
        self.data = torch.tensor(self.data[:len(self.data) // seq_len * seq_len])

    def __len__(self):
        return len(self.data) // self.seq_len - 1

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


class Trainer:
    def __init__(self, model, config: ModelConfig, device="cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        self.checkpoint_dir = Path("data/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.pruning_scheduler = None
        if config.moe.prune_interval > 0 and config.use_moe:
            from src.core.pruning import PruningScheduler
            self.pruning_scheduler = PruningScheduler(
                model, config.moe.n_experts,
                prune_interval=config.moe.prune_interval,
                prune_threshold=config.moe.prune_threshold,
            )

    def train(self, text, max_steps=2000, seq_len=128, batch_size=16):
        dataset = TextDataset(text, seq_len, self.config.vocab_size)
        self.config.vocab_size = dataset.vocab_size
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.to(self.device)
        self.model.train()
        step = 0
        losses = []
        while step < max_steps:
            for x, y in loader:
                if step >= max_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)
                logits, gates, indices, aux_loss = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss = loss + aux_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                losses.append(loss.item())
                if self.pruning_scheduler:
                    self.pruning_scheduler.step(indices)
                if step % 200 == 0:
                    print(f"  step {step}/{max_steps} | loss: {loss.item():.4f} | aux_loss: {aux_loss.item():.4f}")
                step += 1
        self._save_checkpoint()
        return losses

    def _save_checkpoint(self):
        path = self.checkpoint_dir / "moe_demo.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "vocab_size": self.config.vocab_size,
        }, path)
        print(f"  checkpoint saved to {path}")

    def load_checkpoint(self):
        path = self.checkpoint_dir / "moe_demo.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.config.vocab_size = ckpt["vocab_size"]
            self.model.load_state_dict(ckpt["model_state_dict"])
            return True
        return False
