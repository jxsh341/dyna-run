import pandas as pd


class RoutingTracer:
    def __init__(self):
        self.records = []

    def reset(self):
        self.records = []

    def record(self, layer, token_pos, expert_id, weight, step=None):
        self.records.append({
            "layer": layer,
            "token_pos": token_pos,
            "expert_id": expert_id,
            "weight": weight,
            "step": step if step is not None else 0,
        })

    def to_dataframe(self):
        return pd.DataFrame(self.records)

    def expert_activation_matrix(self, n_layers, n_experts, n_tokens=None):
        import numpy as np
        mat = np.zeros((n_layers, n_experts))
        for r in self.records:
            mat[r["layer"], r["expert_id"]] += 1
        return mat

    def token_paths(self):
        paths = {}
        for r in self.records:
            key = (r["token_pos"], r["step"])
            if key not in paths:
                paths[key] = {}
            if r["layer"] not in paths[key]:
                paths[key][r["layer"]] = []
            paths[key][r["layer"]].append(r["expert_id"])
        return paths

    def expert_utilization(self, n_experts):
        counts = {i: 0 for i in range(n_experts)}
        for r in self.records:
            counts[r["expert_id"]] = counts.get(r["expert_id"], 0) + 1
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}

    def routing_entropy(self):
        import numpy as np
        from collections import Counter
        if not self.records:
            return 0.0
        expert_counts = Counter(r["expert_id"] for r in self.records)
        total = sum(expert_counts.values())
        probs = np.array([c / total for c in expert_counts.values()])
        return -np.sum(probs * np.log(probs + 1e-10))
