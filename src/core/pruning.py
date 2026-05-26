import torch
from collections import Counter


class PruningScheduler:
    def __init__(self, model, n_experts, prune_interval=500, prune_threshold=0.05):
        self.model = model
        self.n_experts = n_experts
        self.prune_interval = prune_interval
        self.prune_threshold = prune_threshold
        self._utilization_history = {e: [] for e in range(n_experts)}
        self._pruned_experts = set()
        self._step = 0

    def step(self, all_indices):
        self._step += 1
        for layer_indices in all_indices:
            counts = Counter(layer_indices[:, :, :].unique().tolist())
            for e in range(self.n_experts):
                self._utilization_history[e].append(counts.get(e, 0))
        if self._step % self.prune_interval == 0:
            self._apply_pruning()

    def _apply_pruning(self):
        recent_steps = min(5, len(next(iter(self._utilization_history.values()))))
        for e in range(self.n_experts):
            recent = self._utilization_history[e][-recent_steps:]
            avg = sum(recent) / max(recent_steps, 1)
            total_avg = sum(
                sum(self._utilization_history[ee][-recent_steps:])
                for ee in range(self.n_experts)
            ) / max(self.n_experts, 1)
            if total_avg > 0 and avg / total_avg < self.prune_threshold:
                self._pruned_experts.add(e)
        active = [e for e in range(self.n_experts) if e not in self._pruned_experts]
        if active:
            for block in self.model.blocks:
                block.moe.set_pruning_mask(active)

    def clear_pruning(self):
        self._pruned_experts.clear()
        for block in self.model.blocks:
            block.moe.clear_pruning_mask()

    def get_pruned_experts(self):
        return set(self._pruned_experts)

    def get_utilization(self):
        return {e: self._utilization_history[e] for e in range(self.n_experts)}
