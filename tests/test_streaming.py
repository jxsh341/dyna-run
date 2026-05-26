import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import ModelConfig, MoEConfig
from src.models.moe_transformer import MoETransformer
from src.streaming.shard_manager import ShardManager
from src.streaming.streaming_scheduler import StreamingScheduler
from src.streaming.memory_controller import MemoryController
from src.streaming.sparse_engine import SparseStreamingEngine


def test_shard_manager_save_load():
    mgr = ShardManager(shard_dir="data/experts")
    state = {"weight": torch.randn(128, 256)}
    mgr.save_expert_shard(state, layer_idx=0, expert_idx=3)
    loaded = mgr.load_expert_shard(0, 3)
    assert "weight" in loaded
    assert torch.allclose(state["weight"], loaded["weight"])
    assert mgr.shard_exists(0, 3)
    assert mgr.shard_size_bytes(0, 3) > 0
    mgr.delete_expert_shard(0, 3)
    assert not mgr.shard_exists(0, 3)


def test_shard_manager_shared():
    mgr = ShardManager(shard_dir="data/experts")
    state = {"embed.weight": torch.randn(128, 64)}
    mgr.save_shared_shard(state, "token_embedding")
    loaded = mgr.load_shared_shard("token_embedding")
    assert "embed.weight" in loaded
    assert torch.allclose(state["embed.weight"], loaded["embed.weight"])
    Path("data/experts/token_embedding.pt").unlink(missing_ok=True)


def test_streaming_scheduler():
    mgr = ShardManager(shard_dir="data/experts")
    state = {"weight": torch.randn(64, 128)}
    mgr.save_expert_shard(state, 0, 1)
    scheduler = StreamingScheduler(mgr, num_workers=1)
    scheduler.prefetch_expert(0, 1)
    loaded = scheduler.get_expert(0, 1)
    assert "weight" in loaded
    scheduler.shutdown()
    mgr.delete_expert_shard(0, 1)


def test_memory_controller():
    mc = MemoryController(device="cpu")
    state = {"w": torch.randn(32, 64)}
    mc.load_to_gpu(state, 0, 2)
    assert mc.is_resident(0, 2)
    assert mc.get_expert_on_gpu(0, 2) is not None
    mc.evict_from_gpu(0, 2)
    assert not mc.is_resident(0, 2)


def test_engine_shard_and_forward():
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=2,
                      max_seq_len=64, moe=MoEConfig(n_experts=4, top_k=2, d_model=64, d_ff=128))
    model = MoETransformer(cfg)
    engine = SparseStreamingEngine(model, cfg, shard_dir="data/experts")
    engine.shard_model()
    assert engine.shard_manager.count_shards() > 0
    x = torch.randint(0, 128, (1, 16))
    logits, tracer, timings, stream_metrics, mem_summary = engine.forward_streaming(x)
    assert logits.shape == (1, 16, 128)
    assert len(tracer.records) > 0
    assert timings["total_ms"] > 0
    assert stream_metrics["cache_hits"] >= 0
    assert mem_summary["n_experts_in_gpu"] == 0
    engine.reset_shards()


def test_streaming_profile():
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=2,
                      max_seq_len=64, moe=MoEConfig(n_experts=4, top_k=2, d_model=64, d_ff=128))
    model = MoETransformer(cfg)
    engine = SparseStreamingEngine(model, cfg, shard_dir="data/experts")
    engine.shard_model()
    from src.engine.profiler import Profiler
    profiler = Profiler()
    x = torch.randint(0, 128, (1, 16))
    result = profiler.measure_streaming(engine, x)
    assert result["total_ms"] > 0
    assert result["cache_hit_rate"] >= 0
    engine.reset_shards()


def test_shard_manager_sharing_reduces_disk():
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=2,
                      max_seq_len=64, moe=MoEConfig(n_experts=4, top_k=2, d_model=64, d_ff=128))
    model = MoETransformer(cfg)
    # Shard without sharing
    mgr_no = ShardManager(shard_dir="data/experts")
    mgr_no.shard_model(model)
    count_no = mgr_no.count_shards()
    size_no = mgr_no.total_shard_size_mb()
    for p in Path("data/experts").glob("*.pt"):
        p.unlink(missing_ok=True)
    for p in Path("data/experts").glob("*.meta"):
        p.unlink(missing_ok=True)
    # Shard with sharing: layer 1 reuses layer 0 experts
    shared_map = {(1, e): (0, e) for e in range(4)}
    mgr_shared = ShardManager(shard_dir="data/experts")
    mgr_shared.shard_model(model, shared_expert_map=shared_map)
    count_shared = mgr_shared.count_shards()
    size_shared = mgr_shared.total_shard_size_mb()
    assert count_shared < count_no, f"Expected fewer shards with sharing ({count_shared} vs {count_no})"
    assert size_shared <= size_no + 0.01, "Shared sharding should not increase disk usage"
    # Cleanup
    for p in Path("data/experts").glob("*"):
        p.unlink(missing_ok=True)


def test_memory_controller_refcounting():
    mc = MemoryController(device="cpu")
    shared_map = {(1, 0): (0, 0), (1, 1): (0, 1)}
    mc.set_sharing_map(shared_map)
    state = {"w": torch.randn(32, 64)}
    # Load (0, 0) — canonical (0, 0)
    mc.load_to_gpu(state, 0, 0)
    assert mc.is_resident(0, 0)
    assert mc._refcount.get((0, 0)) == 1
    # Load (1, 0) — resolves to canonical (0, 0), refcount becomes 2
    mc.load_to_gpu(state, 1, 0)
    assert mc._refcount.get((0, 0)) == 2
    assert mc.is_resident(1, 0)
    # Evict (0, 0) — refcount drops to 1, not evicted
    mc.evict_from_gpu(0, 0)
    assert mc.is_resident(0, 0)
    assert mc._refcount.get((0, 0)) == 1
    # Evict (1, 0) — refcount drops to 0, evicted
    mc.evict_from_gpu(1, 0)
    assert not mc.is_resident(0, 0)
    assert (0, 0) not in mc._refcount


def test_engine_sharing_forward():
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=2,
                      max_seq_len=64, moe=MoEConfig(n_experts=4, top_k=2, d_model=64, d_ff=128))
    model = MoETransformer(cfg)
    engine = SparseStreamingEngine(model, cfg, shard_dir="data/experts")
    shared_map = {(1, e): (0, e) for e in range(4)}
    engine.set_expert_sharing_map(shared_map)
    engine.shard_model()
    total_shards = engine.shard_manager.count_shards()
    assert total_shards == 4, f"Expected 4 shards (only layer 0 experts), got {total_shards}"
    x = torch.randint(0, 128, (1, 16))
    logits, tracer, timings, stream_metrics, mem_summary = engine.forward_streaming(x)
    assert logits.shape == (1, 16, 128)
    assert len(tracer.records) > 0
    assert timings["total_ms"] > 0
    engine.reset_shards()


def test_engine_sharing_numerical_correctness():
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=2,
                      max_seq_len=64, moe=MoEConfig(n_experts=4, top_k=2, d_model=64, d_ff=128))
    model = MoETransformer(cfg)
    # Force layer 1 experts to share layer 0 expert weights in the model itself
    sd = model.state_dict()
    for e in range(4):
        for key in ["net.0.weight", "net.2.weight"]:
            src_key = f"blocks.0.moe.experts.experts.{e}.{key}"
            dst_key = f"blocks.1.moe.experts.experts.{e}.{key}"
            sd[dst_key] = sd[src_key].clone()
    model.load_state_dict(sd)
    vanilla = MoETransformer(cfg)
    vanilla.load_state_dict(model.state_dict())
    engine = SparseStreamingEngine(model, cfg, shard_dir="data/experts")
    shared_map = {(1, e): (0, e) for e in range(4)}
    engine.set_expert_sharing_map(shared_map)
    engine.shard_model()
    x = torch.randint(0, 128, (1, 8))
    logits_shared, tracer, timings, _, _ = engine.forward_streaming(x)
    vanilla.eval()
    with torch.no_grad():
        logits_vanilla, _, _, _ = vanilla(x)
    diff = (logits_shared - logits_vanilla).abs().max().item()
    assert diff < 1e-5, f"Numerical mismatch: max diff = {diff}"
    engine.reset_shards()


def test_distribute_shards():
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=2,
                      max_seq_len=64, moe=MoEConfig(n_experts=4, top_k=2, d_model=64, d_ff=128))
    model = MoETransformer(cfg)
    mgr = ShardManager(shard_dir="data/experts")
    mgr.shard_model(model)
    manifest = mgr.distribute_shards(n_devices=2)
    assert len(manifest) > 0
    for key, dev_id in manifest.items():
        assert dev_id in (0, 1)
    loaded = mgr.load_device_manifest()
    assert len(loaded) == len(manifest)
    for p in Path("data/experts").glob("*.pt"):
        p.unlink(missing_ok=True)
    for p in Path("data/experts").glob("*.meta"):
        p.unlink(missing_ok=True)


def test_memory_controller_device_map():
    mc = MemoryController(device="cpu")
    mc.set_device_map({(0, 0): 0, (0, 1): 1})
    state = {"w": torch.randn(32, 64)}
    mc.load_to_gpu(state, 0, 0)
    assert mc.is_resident(0, 0)
    mc.evict_all()


def test_scheduler_device_assignments():
    mgr = ShardManager(shard_dir="data/experts")
    state = {"w": torch.randn(64, 128)}
    for e in range(4):
        mgr.save_expert_shard(state, 0, e)
    scheduler = StreamingScheduler(mgr, num_workers=1, devices=["cpu:0", "cpu:1"])
    scheduler.set_device_assignments({(0, 0): 0, (0, 1): 1, (0, 2): 0, (0, 3): 1})
    state_dicts = scheduler.resolve_experts(0, [0, 1, 2, 3])
    assert len(state_dicts) == 4
    for eid in range(4):
        assert "w" in state_dicts[eid]
    scheduler.shutdown()
    for e in range(4):
        mgr.delete_expert_shard(0, e)


def test_heterogeneous_shard_save_load():
    path = Path("data/experts")
    path.mkdir(parents=True, exist_ok=True)
    mgr = ShardManager(shard_dir="data/experts")
    small = {"net.0.weight": torch.randn(128, 64), "net.2.weight": torch.randn(64, 64)}
    large = {"net.0.weight": torch.randn(1024, 64), "net.2.weight": torch.randn(512, 64)}
    mgr.save_expert_shard(small, 0, 0)
    mgr.save_expert_shard(large, 0, 1)
    loaded_small = mgr.load_expert_shard(0, 0)
    loaded_large = mgr.load_expert_shard(0, 1)
    assert loaded_small["net.0.weight"].shape == (128, 64)
    assert loaded_large["net.0.weight"].shape == (1024, 64)
    assert mgr.shard_size_bytes(0, 0) < mgr.shard_size_bytes(0, 1)
    mgr.delete_expert_shard(0, 0)
    mgr.delete_expert_shard(0, 1)


def test_streaming_heterogeneous_forward():
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=2,
                      max_seq_len=64, moe=MoEConfig(
                          n_experts=4, top_k=2, d_model=64, d_ff=128,
                          heterogeneous=True, expert_dims=[32, 64, 128, 256],
                      ))
    model = MoETransformer(cfg)
    engine = SparseStreamingEngine(model, cfg, shard_dir="data/experts")
    engine.shard_model()
    x = torch.randint(0, 128, (1, 8))
    logits, tracer, timings, _, _ = engine.forward_streaming(x)
    assert logits.shape == (1, 8, 128)
    engine.reset_shards()


if __name__ == "__main__":
    test_shard_manager_save_load()
    test_shard_manager_shared()
    test_streaming_scheduler()
    test_memory_controller()
    test_engine_shard_and_forward()
    test_streaming_profile()
    test_shard_manager_sharing_reduces_disk()
    test_memory_controller_refcounting()
    test_engine_sharing_forward()
    test_engine_sharing_numerical_correctness()
    test_distribute_shards()
    test_memory_controller_device_map()
    test_scheduler_device_assignments()
    test_heterogeneous_shard_save_load()
    test_streaming_heterogeneous_forward()
    print("All streaming tests passed!")
