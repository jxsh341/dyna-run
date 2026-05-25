# Dyna-Run — Dynamic Sparse AI Inference Runtime

A lightweight inference runtime that implements **real Mixture-of-Experts (MoE) routing** from scratch. Instead of activating all parameters for every token — like every employee attending every meeting — Dyna-Run routes each token to only **2 out of 8 expert sub-networks**, skipping unnecessary computation and visualizing the entire decision process.

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

On first run, Dyna-Run auto-trains a small MoE transformer on Shakespeare (~2 minutes on CPU) then opens the interactive dashboard.

## Architecture

```
Input Tokens → Embedding → [Attention + MoE Layer] × 4 → Output
                                │
                     ┌──────────┴──────────┐
                  Router                 Experts
              (top-2 gating)          (8 × SwiGLU FFN)
                     │                     │
              "token → expert 3"      expert 3 only
              "token → expert 7"      expert 7 only
```

### Core Components

| Module | File | Purpose |
|--------|------|---------|
| **Router** | `src/core/router.py` | Noisy top-k gating — learns to dispatch each token to the most relevant experts |
| **Experts** | `src/core/experts.py` | 8 SwiGLU feed-forward networks — the "specialists" that process tokens |
| **SparseMoE** | `src/core/sparse_moe.py` | Combines router + experts with auxiliary load-balancing loss |
| **DenseBlock** | `src/core/dense_block.py` | Equivalent-capacity dense FFN for fair comparison |
| **MoE Transformer** | `src/models/moe_transformer.py` | 4-layer decoder with RoPE attention + MoE FFN |
| **Dense Transformer** | `src/models/dense_transformer.py` | Same architecture, all parameters always active |

### Default Demo Configuration

| Parameter | Value |
|-----------|-------|
| Experts | 8 |
| Active experts per token | 2 (top-k) |
| Model dimension | 128 |
| FFN hidden dimension | 256 |
| Layers | 4 |
| Attention heads | 4 |
| Max sequence length | 256 |
| Sparsity | 75% of expert params skipped per token |

## Dashboard

Dyna-Run ships with a 3-page Streamlit dashboard.

### 1. Routing Visualization

Enter any text and see exactly which experts activate per token:

- **Sankey diagram** — flow of tokens through experts across layers
- **Token routing paths** — per-layer activation counts
- **Expert activation heatmap** — which experts fire across layers × tokens
- **Expert utilization bar** — distribution of routing decisions
- **Routing metrics** — load balancing score, routing entropy, top-k confidence

### 2. Benchmark Dashboard

Compare performance at multiple sequence lengths (64–512):

- **Speed** (tokens/second) — sparse vs dense
- **Latency** (ms) — with variance and standard deviation
- **Memory** (MB RSS) — RAM usage comparison
- **Speedup factor** — improvement ratio
- **Active parameters** — parameter efficiency ratio

### 3. External Models

Partial integration with local model runtimes:

- **Ollama** — query models via REST API (`localhost:11434`)
- **HuggingFace Transformers** — load models from HuggingFace Hub, probe hidden states
- **llama.cpp** — inference via `llama-cpp-python` with GGUF models

## Project Structure

```
dyna-run/
├── src/
│   ├── core/            # Router, experts, MoE layer, dense block
│   ├── models/          # MoE transformer, dense transformer
│   ├── engine/          # Inference, training, profiling
│   ├── trace/           # Routing tracer and metrics
│   ├── viz/             # Plotly/Matplotlib visualizations
│   └── interfaces/      # Ollama, HuggingFace, llama.cpp clients
├── dashboard/           # Streamlit app (3 pages)
│   ├── app.py           # Entry point with auto-training
│   └── pages/           # Routing viz, benchmark, external models
├── scripts/             # CLI training and benchmark scripts
├── tests/               # 14 unit tests
├── data/                # Sample text, checkpoints
├── run.bat              # Windows launcher
├── requirements.txt
└── setup.py
```

## Running Tests

```bash
python tests/test_router.py
python tests/test_experts.py
python tests/test_sparse_moe.py
python tests/test_inference.py
```

## CLI Usage

```bash
# Train the demo model
python scripts/train_demo.py

# Run benchmarks headless
python scripts/run_benchmark.py
```

## Dependencies

- **Required**: torch, numpy, psutil, pandas, plotly, matplotlib, streamlit, requests
- **Optional**: transformers (HuggingFace), llama-cpp-python

## Troubleshooting

### Stale checkpoint from a different config

If you see `RuntimeError: Error(s) in loading state_dict` with size mismatches, a previous run saved a checkpoint with different model dimensions. Delete it and relaunch:

```bash
Remove-Item data/checkpoints/moe_demo.pt
python -m streamlit run dashboard/app.py
```

The dashboard will retrain a fresh model on next launch.

### Ollama connection errors

The External Models page connects to `http://localhost:11434`. Make sure Ollama is running:

```bash
ollama serve
```

If you see error 500, the model name may be incorrect. Pull it first:

```bash
ollama pull llama3.2:1b
```

### Streamlit not found

If `streamlit` is not recognized, use the Python module syntax:

```bash
python -m streamlit run dashboard/app.py
```

## How It Works

### Routing (Noisy Top-K Gating)

1. Each token is projected through a learned gate to produce scores over 8 experts
2. During training, Gaussian noise (learned variance) is added to the logits to encourage exploration
3. The top-2 experts by score are selected; all others are masked to zero
4. Selected expert outputs are weighted by their gate scores and summed
5. An auxiliary load-balancing loss encourages uniform expert utilization

### Sparse Computation

Only the activated 2 out of 8 expert FFNs compute for each token. For a batch of tokens, each expert processes only the tokens routed to it. This means:

- **75% fewer expert FLOPs** per token compared to a dense model of equal total capacity
- **Same total parameter count** — the model can be much larger while keeping inference cost constant

### Why Dense Is Faster at Small Scale

At toy scale (d_model=128, 4 layers), the routing overhead and non-contiguous memory access mean the dense baseline is faster. The advantage of sparse routing emerges at scale — models with hundreds of experts where activating all of them would be prohibitive. Dyna-Run shows **both** sides transparently, letting you observe the trade-off firsthand.
