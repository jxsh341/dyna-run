import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import ModelConfig, MoEConfig
from src.models.moe_transformer import MoETransformer
from src.engine.inference import InferenceEngine
from src.engine.train import Trainer
from src.core.pruning import PruningScheduler


def test_pruning_scheduler_creation():
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=2,
                      max_seq_len=64, moe=MoEConfig(
                          n_experts=4, top_k=2, d_model=64, d_ff=128,
                          prune_interval=100, prune_threshold=0.1,
                      ))
    model = MoETransformer(cfg)
    trainer = Trainer(model, cfg)
    assert trainer.pruning_scheduler is not None
    assert trainer.pruning_scheduler.prune_interval == 100
    assert trainer.pruning_scheduler.prune_threshold == 0.1


def test_pruning_scheduler_step():
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=2,
                      max_seq_len=64, moe=MoEConfig(n_experts=4, top_k=2, d_model=64, d_ff=128))
    model = MoETransformer(cfg)
    scheduler = PruningScheduler(model, n_experts=4, prune_interval=2, prune_threshold=0.2)
    indices = torch.randint(0, 4, (1, 4, 2))
    scheduler.step([indices])
    assert scheduler._step == 1
    assert len(scheduler._pruned_experts) == 0


def test_pruning_scheduler_clear():
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=2,
                      max_seq_len=64, moe=MoEConfig(n_experts=4, top_k=2, d_model=64, d_ff=128))
    model = MoETransformer(cfg)
    scheduler = PruningScheduler(model, n_experts=4, prune_interval=1, prune_threshold=0.5)
    indices = torch.zeros(1, 4, 2, dtype=torch.long)
    scheduler.step([indices])
    assert scheduler._step == 1
    scheduler.clear_pruning()
    assert len(scheduler._pruned_experts) == 0
    for block in model.blocks:
        assert block.moe.router.expert_mask is None


def test_serve_app_imports():
    from src.serve.app import app
    assert app.title == "Dyna-Run API"


def test_serve_route_endpoint():
    from src.serve.app import RouteRequest, RouteResponse
    req = RouteRequest(tokens=[1, 2, 3])
    assert len(req.tokens) == 3


def test_serve_generate_endpoint():
    from src.serve.app import GenerateRequest, GenerateResponse
    req = GenerateRequest(prompt_tokens=[1, 2], max_new_tokens=10, temperature=0.8)
    assert req.max_new_tokens == 10
    assert req.temperature == 0.8


def test_serve_profile_endpoint():
    from src.serve.app import ProfileRequest, ProfileResponse
    req = ProfileRequest(seq_len=64, n_runs=3)
    assert req.seq_len == 64
    assert req.n_runs == 3


def test_serve_health_endpoint():
    from src.serve.app import HealthResponse
    resp = HealthResponse(status="ok", device="cpu", model_ready=True, shard_count=0)
    assert resp.status == "ok"
    assert resp.device == "cpu"


if __name__ == "__main__":
    test_pruning_scheduler_creation()
    test_pruning_scheduler_step()
    test_pruning_scheduler_clear()
    test_serve_app_imports()
    test_serve_route_endpoint()
    test_serve_generate_endpoint()
    test_serve_profile_endpoint()
    test_serve_health_endpoint()
    print("All Phases 7-8 tests passed!")
