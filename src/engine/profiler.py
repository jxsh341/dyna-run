import time
import psutil
import torch
import numpy as np


class Profiler:
    def __init__(self):
        self.results = {}

    def measure_inference(self, model, input_ids, n_runs=10):
        model.eval()
        model.to(input_ids.device)
        latencies = []
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                logits, gates, indices, aux_loss = model(input_ids)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                latencies.append(time.perf_counter() - start)
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        tokens = input_ids.numel()
        total_params = sum(p.numel() for p in model.parameters())
        active_ratio = self._compute_active_ratio(indices, model.config.moe.n_experts)
        return {
            "mean_latency_ms": np.mean(latencies) * 1000,
            "std_latency_ms": np.std(latencies) * 1000,
            "tokens_per_sec": tokens / np.mean(latencies),
            "ram_mb": mem_after - mem_before,
            "total_params": total_params,
            "active_params_ratio": active_ratio,
            "active_params": int(total_params * active_ratio),
        }

    def _compute_active_ratio(self, indices, n_experts):
        n_layers = len(indices)
        n_experts_per_layer = n_experts
        top_k = indices[0].shape[-1]
        return (n_layers * top_k) / (n_layers * n_experts_per_layer)

    def compare(self, moe_model, dense_model, input_ids, n_runs=10):
        sparse_metrics = self.measure_inference(moe_model, input_ids, n_runs)
        sparse_metrics["mode"] = "sparse (MoE)"
        dense_model.to(input_ids.device)
        dense_model.eval()
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                dense_model(input_ids)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                latencies.append(time.perf_counter() - start)
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        total_params = sum(p.numel() for p in dense_model.parameters())
        dense_metrics = {
            "mode": "dense",
            "mean_latency_ms": np.mean(latencies) * 1000,
            "std_latency_ms": np.std(latencies) * 1000,
            "tokens_per_sec": input_ids.numel() / np.mean(latencies),
            "ram_mb": mem_after - mem_before,
            "total_params": total_params,
            "active_params_ratio": 1.0,
            "active_params": total_params,
        }
        return {
            "sparse": sparse_metrics,
            "dense": dense_metrics,
            "seq_len": input_ids.shape[1],
            "batch_size": input_ids.shape[0],
            "speedup": dense_metrics["mean_latency_ms"] / sparse_metrics["mean_latency_ms"],
            "memory_savings_pct": (1 - sparse_metrics["ram_mb"] / max(dense_metrics["ram_mb"], 0.1)) * 100,
        }

    def benchmark_sequence_lengths(self, moe_model, dense_model, lengths, device="cpu", n_runs=5):
        results = {}
        for L in lengths:
            x = torch.randint(0, moe_model.config.vocab_size, (1, L), device=device)
            results[L] = self.compare(moe_model, dense_model, x, n_runs)
        return results
