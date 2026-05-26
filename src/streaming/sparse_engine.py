import time
import torch
from src.core.config import ModelConfig
from src.models.moe_transformer import MoETransformer
from src.streaming.shard_manager import ShardManager
from src.streaming.streaming_scheduler import StreamingScheduler
from src.streaming.memory_controller import MemoryController
from src.trace.tracer import RoutingTracer


class SparseStreamingEngine:
    def __init__(self, model: MoETransformer, config: ModelConfig,
                 shard_dir="data/experts", num_workers=2):
        self.model = model
        self.config = config
        self.n_layers = config.n_layers
        self.n_experts = config.moe.n_experts
        self.top_k = config.moe.top_k
        self.device = model.device
        self.shard_manager = ShardManager(shard_dir)
        self.scheduler = StreamingScheduler(self.shard_manager, num_workers=num_workers)
        self.memory = MemoryController(str(self.device))
        self.tracer = RoutingTracer()
        self._sharded = False
        self._sharing_map = {}

    def set_expert_sharing_map(self, mapping):
        self._sharing_map = mapping
        self.scheduler.set_key_resolver(lambda l, e: self._sharing_map.get((l, e), (l, e)))
        self.memory.set_sharing_map(mapping)

    def _resolve_key(self, layer_idx, expert_idx):
        return self._sharing_map.get((layer_idx, expert_idx), (layer_idx, expert_idx))

    def shard_model(self):
        self.shard_manager.shard_model(self.model, shared_expert_map=self._sharing_map)
        self._sharded = True

    def shard_distributed(self, n_devices=2):
        self.shard_model()
        manifest = self.shard_manager.distribute_shards(n_devices)
        devices = [f"cuda:{d}" if torch.cuda.is_available() else f"cpu:{d}" for d in range(n_devices)]
        self.scheduler.set_device_assignments(manifest)
        self.memory.set_device_map(manifest)

    def forward_streaming(self, input_ids):
        self.model.eval()
        self.tracer.reset()
        self.scheduler.reset_metrics()
        self.memory.evict_all()
        timings = {"routing_ms": [], "streaming_ms": [], "compute_ms": [], "total_ms": 0}
        t_start = time.perf_counter()
        B, T = input_ids.shape
        x = input_ids.to(self.device)
        with torch.no_grad():
            x = self.model.token_embedding(x)
            for layer_idx in range(self.n_layers):
                block = self.model.blocks[layer_idx]
                t0 = time.perf_counter()
                x = x + block.attn(block.norm1(x))
                t1 = time.perf_counter()
                gates, indices = block.moe.router(block.norm2(x))
                expert_ids = indices[:, :, :].unique().tolist()
                timings["routing_ms"].append((time.perf_counter() - t1) * 1000)
                t2 = time.perf_counter()
                if self._sharded:
                    next_ids = None
                    if layer_idx + 1 < self.n_layers:
                        next_x = x + self.model.blocks[layer_idx + 1].attn(
                            self.model.blocks[layer_idx + 1].norm1(x)
                        )
                        next_gates, next_indices = self.model.blocks[layer_idx + 1].moe.router(
                            self.model.blocks[layer_idx + 1].norm2(next_x)
                        )
                        next_ids = next_indices[:, :, :].unique().tolist()
                    state_dicts = self.scheduler.resolve_experts(
                        layer_idx, expert_ids, next_ids
                    )
                else:
                    state_dicts = {}
                timings["streaming_ms"].append((time.perf_counter() - t2) * 1000)
                t3 = time.perf_counter()
                if self._sharded:
                    for eid in expert_ids:
                        if eid in state_dicts:
                            self.memory.load_to_gpu(state_dicts[eid], layer_idx, eid)
                    x_normed = block.norm2(x)
                    B, T, D = x_normed.shape
                    x_flat = x_normed.view(-1, D)
                    indices_flat = indices[:, :, :].view(-1, indices.shape[-1])
                    moe_flat = torch.zeros_like(x_flat)
                    for eid in expert_ids:
                        expert_weights = self.memory.get_expert_on_gpu(layer_idx, eid)
                        if expert_weights is not None:
                            mask = (indices_flat == eid).any(dim=-1)
                            if mask.any():
                                expert_out = self._apply_expert_weights(
                                    x_flat[mask], expert_weights
                                )
                                moe_flat[mask] = moe_flat[mask] + expert_out
                    x = x + moe_flat.view(B, T, D)
                else:
                    moe_out, gates, indices, aux_loss = block.moe(block.norm2(x))
                    x = x + moe_out
                if self._sharing_map and next_ids is not None:
                    next_canonical = set(self._resolve_key(layer_idx + 1, e) for e in next_ids)
                    for eid in expert_ids:
                        if self._resolve_key(layer_idx, eid) not in next_canonical:
                            self.memory.evict_from_gpu(layer_idx, eid)
                else:
                    for eid in expert_ids:
                        self.memory.evict_from_gpu(layer_idx, eid)
                timings["compute_ms"].append((time.perf_counter() - t3) * 1000)
                self._trace_layer(layer_idx, x, indices, gates)
            x = self.model.norm(x)
            logits = self.model.lm_head(x)
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000
        return logits, self.tracer, timings, self.scheduler.metrics, self.memory.resident_summary()

    def _apply_expert_weights(self, x, state_dict):
        w1 = state_dict["net.0.weight"]
        w2 = state_dict["net.2.weight"]
        hidden = torch.nn.functional.linear(x, w1)
        x_part, gate_part = hidden.chunk(2, dim=-1)
        activated = x_part * torch.nn.functional.silu(gate_part)
        return torch.nn.functional.linear(activated, w2)

    def _trace_layer(self, layer_idx, x, indices, gates):
        for t_idx in range(indices.shape[1]):
            for e_idx in range(indices.shape[2]):
                expert_id = indices[:, t_idx, e_idx].item()
                weight = gates[:, t_idx, expert_id].item() if expert_id < gates.shape[-1] else 0.0
                self.tracer.record(
                    layer=layer_idx,
                    token_pos=t_idx,
                    expert_id=expert_id,
                    weight=weight,
                )

    def get_sharing_map(self):
        return dict(self._sharing_map)

    def reset_shards(self):
        self._sharded = False
        self._sharing_map = {}
        self.scheduler.clear_cache()
        self.memory.evict_all()
