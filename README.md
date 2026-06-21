# Dyna-Run — Dynamic Sparse AI Inference Runtime

A lightweight inference runtime that implements **real Mixture-of-Experts (MoE) routing** from scratch. The default demo activates 2 out of 8 expert sub-networks per token, while the router supports configurable top-k routing. The runtime stores expert weights on disk and **streams only activated experts to GPU at each layer**, enabling massive model sizes on limited hardware.

```
Normal models  = every employee attends every meeting
Sparse models  = only relevant specialists join
```

## Quick Start

```bash
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

Or double-click `run.bat` on Windows.

On first run, Dyna-Run auto-trains a small MoE transformer on Shakespeare (~2 minutes on CPU) then opens the interactive 3-page dashboard.

## Dashboard

| Page | What You See |
|------|-------------|
| **Routing Visualization** | Sankey diagram of token→expert flow, activation heatmaps, expert utilization, routing metrics, pruning status, sharing map |
| **Benchmark Dashboard** | Speed (tok/s), latency (ms), RAM (MB), speedup factor — sparse vs dense at multiple sequence lengths |
| **External Models** | Query Ollama, HuggingFace, or llama.cpp models |

## Default Demo Configuration

| Parameter | Value |
|-----------|-------|
| Experts | 8 |
| Active per token | 2 (top-k) |
| Model dimension | 128 |
| FFN hidden dimension | 256 |
| Layers | 4 |
| Sparsity | 75% of expert params skipped per token |

## Architecture

```
Input Tokens → Embedding → [Attention + MoE Layer] × 4 → Output
                                │
                     ┌──────────┴──────────┐
                  Router                 Experts
              (top-k gating)          (8 × SwiGLU FFN)
                     │                     │
              "token → expert 3"      expert 3 only
              "token → expert 7"      expert 7 only
```

### Core Components

| Module | File | Purpose |
|--------|------|---------|
| **Router** | `src/core/router.py` | Noisy top-k gating with pruning mask support |
| **Experts** | `src/core/experts.py` | Heterogeneous SwiGLU FFNs (each can have different width) |
| **SparseMoE** | `src/core/sparse_moe.py` | Combined router + experts + aux loss + expert_bias scaling |
| **SparseStreamingEngine** | `src/streaming/sparse_engine.py` | Disk-streaming inference: route→load→compute→evict |
| **StreamingScheduler** | `src/streaming/streaming_scheduler.py` | ThreadPoolExecutor prefetch with device-aware scheduling |
| **MemoryController** | `src/streaming/memory_controller.py` | GPU residency tracking, refcounting, device mapping |
| **ShardManager** | `src/streaming/shard_manager.py` | Save/load per-expert shards, sharing maps, distribution manifests |
| **PruningScheduler** | `src/core/pruning.py` | Tracks utilization, soft-prunes underused experts during training |
| **InferenceEngine** | `src/engine/inference.py` | Routing capture and autoregressive generation |
| **FastAPI Server** | `src/serve/app.py` | REST endpoints: health, route, generate, profile, shards |

## What's Built (8 Phases)

### Phase 0: Core MoE
Noisy top-2 gating router, 8 SwiGLU experts, SparseMoE layer with load-balancing auxiliary loss.

### Phase 1: Model Architecture
4-layer RoPE decoder (MoETransformer) + equivalent DenseTransformer baseline. Self-trains on Shakespeare via AdamW.

### Phase 2: Tracing & Visualization
RoutingTracer records every token→expert decision. Plotly Sankey diagrams, activation heatmaps, utilization bar charts. RoutingMetrics computes load balancing, entropy, specialization scores.

### Phase 3: Profiling & External Integrations
Profiler measures latency, tokens/sec, RAM delta, active params ratio. Supports Ollama, HuggingFace, and llama.cpp adapters.

### Phase 4: Expert Disk Streaming (Core Innovation)
**ShardManager** saves each expert as an independent `.pt` shard. **StreamingScheduler** prefetches via ThreadPoolExecutor. **MemoryController** tracks GPU residency. **SparseStreamingEngine** orchestrates per-layer: route → determine needed experts → stream from disk → compute only activated → evict. Prefetches next layer's experts in parallel.

### Phase 5: Cross-Layer Expert Sharing
Experts can be shared across layers (e.g., layer 1 reuses layer 0's experts). Aliased shards aren't duplicated on disk. MemoryController refcounts shared experts (stay resident until all layers evict).

### Phase 6: Multi-GPU & Distributed Sharding
ShardManager round-robins shards across devices and saves a device manifest. Scheduler assigns experts to specific CUDA devices. Falls back cleanly on single-device.

### Phase 7: Dynamic Expert Sizing & Pruning
Each expert supports a different `d_ff` width (heterogeneous). Learned `expert_bias` scales gate weights per-expert. PruningScheduler tracks utilization and applies soft-pruning via router expert_mask during training.

### Phase 8: Production API Server
FastAPI app at `src/serve/app.py` with 5 endpoints. By default it tries to load `data/checkpoints/moe_demo.pt`; set `DYNA_RUN_CHECKPOINT_PATH=/path/to/checkpoint.pt` to load another checkpoint explicitly.
- `GET /health` — device, model status, shard count, GPU memory
- `POST /route` — token routing decisions without full forward
- `POST /generate` — autoregressive sampling with routing traces
- `POST /profile` — benchmark at configurable seq_len / n_runs
- `GET /shards` — list expert shards with sizes

Dockerized via `Dockerfile`.

## Running

```bash
# Train the demo model
python scripts/train_demo.py

# Run benchmarks headless
python scripts/run_benchmark.py

# Launch dashboard
python -m streamlit run dashboard/app.py

# Launch API server
python -m uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

# Run all tests
python -m pip install -e ".[api,dev]"
python -m pytest -q
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Stale checkpoint (shape mismatch) | `Remove-Item data/checkpoints/moe_demo.pt` |
| Stale shard files (format changed) | `Remove-Item data/experts/*.pt` |
| Device manifest stale | `Remove-Item data/experts/device_manifest.pt` |
| Streamlit not recognized | `python -m streamlit run dashboard/app.py` |

## Dependencies

- **Required core**: torch, numpy, psutil, pandas, plotly, matplotlib, streamlit, requests
- **API extra**: `pip install -e ".[api]"` installs fastapi, uvicorn, and pydantic
- **External-model extra**: `pip install -e ".[external]"` installs transformers and llama-cpp-python
- **Development/test extra**: `pip install -e ".[api,dev]"` installs API dependencies and pytest for the full test suite
- **Convenience requirements file**: `pip install -r requirements.txt` installs the project with the API extra

## Tests

The full pytest suite currently covers router behavior, expert dispatch, sparse MoE layers, inference, streaming/sharding, config validation, pruning, and API schemas. Most tests use small dimensions for speed:
`vocab_size=128, d_model=64, n_layers=2, n_heads=2, n_experts=4, top_k=2`.
