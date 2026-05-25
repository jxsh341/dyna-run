import numpy as np
from collections import Counter


class RoutingMetrics:
    def __init__(self, tracer):
        self.tracer = tracer
        self.records = tracer.records

    def load_balancing_score(self, n_experts):
        counts = Counter(r["expert_id"] for r in self.records)
        total = sum(counts.values()) or 1
        fractions = np.array([counts.get(i, 0) / total for i in range(n_experts)])
        ideal = 1.0 / n_experts
        return 1.0 - np.abs(fractions - ideal).mean() / (2 * (1 - ideal))

    def expert_specialization(self, n_experts):
        expert_layers = {i: [] for i in range(n_experts)}
        for r in self.records:
            expert_layers[r["expert_id"]].append(r["layer"])
        specialization = {}
        for eid, layers in expert_layers.items():
            if layers:
                spec = 1.0 - np.std(layers) / (max(layers) - min(layers) + 1e-10)
                specialization[eid] = spec
            else:
                specialization[eid] = 0.0
        return specialization

    def top_k_confidence(self):
        confidences = []
        token_keys = set((r["token_pos"], r["layer"], r.get("step", 0)) for r in self.records)
        for key in token_keys:
            weights = [r["weight"] for r in self.records
                       if (r["token_pos"], r["layer"], r.get("step", 0)) == key]
            if len(weights) >= 2:
                sorted_w = sorted(weights, reverse=True)
                confidences.append(sorted_w[0] - sorted_w[1])
        return np.mean(confidences) if confidences else 0.0

    def summary(self, n_experts):
        return {
            "total_routing_decisions": len(self.records),
            "unique_experts_used": len(set(r["expert_id"] for r in self.records)),
            "load_balancing": self.load_balancing_score(n_experts),
            "top_k_confidence": self.top_k_confidence(),
            "routing_entropy": self.tracer.routing_entropy(),
        }
