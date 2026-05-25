import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.core.config import ModelConfig, MoEConfig
from src.models.moe_transformer import MoETransformer
from src.models.dense_transformer import DenseTransformer
from src.engine.profiler import Profiler
from src.engine.train import Trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    moe_config = MoEConfig(n_experts=8, top_k=2, d_model=128, d_ff=256)
    model_config = ModelConfig(
        vocab_size=512, d_model=128, n_layers=4, n_heads=4,
        max_seq_len=256, moe=moe_config,
    )

    moe_model = MoETransformer(model_config)
    dense_model = DenseTransformer(model_config)

    trainer = Trainer(moe_model, model_config, device)
    if not trainer.load_checkpoint():
        print("No checkpoint found. Run scripts/train_demo.py first.")
        return

    moe_model.to(device)
    dense_model.to(device)
    profiler = Profiler()

    lengths = [64, 128, 256]
    print(f"Benchmarking at sequence lengths: {lengths}")

    for L in lengths:
        x = torch.randint(0, model_config.vocab_size, (1, L), device=device)
        results = profiler.compare(moe_model, dense_model, x, n_runs=10)
        print(f"\nSequence length = {L}:")
        print(f"  Sparse: {results['sparse']['tokens_per_sec']:.1f} tok/s, "
              f"{results['sparse']['mean_latency_ms']:.2f} ms, "
              f"{results['sparse']['ram_mb']:.1f} MB")
        print(f"  Dense:  {results['dense']['tokens_per_sec']:.1f} tok/s, "
              f"{results['dense']['mean_latency_ms']:.2f} ms, "
              f"{results['dense']['ram_mb']:.1f} MB")
        print(f"  Speedup: {results['speedup']:.2f}x, "
              f"Memory savings: {results['memory_savings_pct']:.1f}%")


if __name__ == "__main__":
    main()
