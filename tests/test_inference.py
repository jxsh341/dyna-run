import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import ModelConfig, MoEConfig
from src.models.moe_transformer import MoETransformer
from src.engine.inference import InferenceEngine
from src.trace.metrics import RoutingMetrics


def test_inference_engine_trace():
    config = ModelConfig(vocab_size=256, n_layers=2, max_seq_len=64)
    model = MoETransformer(config)
    engine = InferenceEngine(model)
    x = torch.randint(0, 256, (1, 16))
    logits, tracer = engine.run_sparse(x)
    assert len(tracer.records) > 0, "Tracer should have records"
    df = tracer.to_dataframe()
    assert len(df) > 0, "Dataframe should not be empty"
    assert "layer" in df.columns
    assert "expert_id" in df.columns
    assert "weight" in df.columns


def test_routing_metrics():
    config = ModelConfig(vocab_size=256, n_layers=2, max_seq_len=64)
    model = MoETransformer(config)
    engine = InferenceEngine(model)
    x = torch.randint(0, 256, (1, 16))
    logits, tracer = engine.run_sparse(x)
    metrics = RoutingMetrics(tracer)
    summary = metrics.summary(n_experts=8)
    assert summary["total_routing_decisions"] > 0
    assert summary["unique_experts_used"] > 0
    assert 0 <= summary["load_balancing"] <= 1


def test_model_generate():
    config = ModelConfig(vocab_size=256, n_layers=2, max_seq_len=64)
    model = MoETransformer(config)
    model.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        generated = model.generate(x, max_new_tokens=5, temperature=1.0)
    assert generated.shape[1] == 13, f"Expected 13 tokens, got {generated.shape[1]}"


if __name__ == "__main__":
    test_inference_engine_trace()
    test_routing_metrics()
    test_model_generate()
    print("All inference tests passed!")
