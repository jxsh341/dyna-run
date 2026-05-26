import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.core.config import ModelConfig, MoEConfig
from src.models.moe_transformer import MoETransformer
from src.engine.inference import InferenceEngine
from src.engine.profiler import Profiler
from src.streaming.sparse_engine import SparseStreamingEngine


_device = "cuda" if torch.cuda.is_available() else "cpu"
_app_state = {}
_lifespan_done = False


class RouteRequest(BaseModel):
    tokens: list[int]

class RouteResponse(BaseModel):
    routing_decisions: list
    n_layers: int
    n_experts: int

class GenerateRequest(BaseModel):
    prompt_tokens: list[int]
    max_new_tokens: int = 30
    temperature: float = 1.0

class GenerateResponse(BaseModel):
    output_tokens: list[int]
    routing_steps: list

class ProfileRequest(BaseModel):
    seq_len: int = 128
    n_runs: int = 5

class ProfileResponse(BaseModel):
    mean_latency_ms: float
    std_latency_ms: float
    tokens_per_sec: float
    active_params: int

class ShardInfo(BaseModel):
    layer: int
    expert: int
    size_bytes: int

class HealthResponse(BaseModel):
    status: str
    device: str
    model_ready: bool
    shard_count: int
    gpu_memory_mb: float | None = None


def _ensure_loaded():
    if _app_state.get("model"):
        return
    moe_config = MoEConfig(n_experts=8, top_k=2, d_model=128, d_ff=256)
    model_config = ModelConfig(
        vocab_size=512, d_model=128, n_layers=4, n_heads=4,
        max_seq_len=256, moe=moe_config,
    )
    model = MoETransformer(model_config)
    trainer = _app_state.get("trainer")
    if trainer:
        trainer.load_checkpoint()
    model.to(_device)
    model.eval()
    engine = SparseStreamingEngine(model, model_config, shard_dir="data/experts")
    _app_state["model"] = model
    _app_state["config"] = model_config
    _app_state["engine"] = engine
    _app_state["inference"] = InferenceEngine(model)
    _app_state["profiler"] = Profiler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_loaded()
    yield

app = FastAPI(
    title="Dyna-Run API",
    description="Dynamic Sparse AI Inference Runtime — REST API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    shard_count = 0
    engine = _app_state.get("engine")
    if engine:
        shard_count = engine.shard_manager.count_shards()
    return HealthResponse(
        status="ok",
        device=_device,
        model_ready=_app_state.get("model") is not None,
        shard_count=shard_count,
        gpu_memory_mb=torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None,
    )


@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest):
    model = _app_state.get("model")
    if model is None:
        raise HTTPException(503, "Model not loaded")
    engine = _app_state["inference"]
    x = torch.tensor([req.tokens], device=_device)
    logits, tracer = engine.run_sparse(x)
    return RouteResponse(
        routing_decisions=[r.__dict__ if hasattr(r, "__dict__") else r for r in tracer.records],
        n_layers=_app_state["config"].n_layers,
        n_experts=_app_state["config"].moe.n_experts,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    model = _app_state.get("model")
    if model is None:
        raise HTTPException(503, "Model not loaded")
    x = torch.tensor([req.prompt_tokens], device=_device)
    output, tracer = _app_state["inference"].generate(
        x, max_new_tokens=req.max_new_tokens, temperature=req.temperature
    )
    routing_steps = []
    for r in tracer.records:
        if r.get("step", 0) > 0 or len(routing_steps) == 0:
            routing_steps.append(r.__dict__ if hasattr(r, "__dict__") else r)
    return GenerateResponse(
        output_tokens=output[0].tolist(),
        routing_steps=routing_steps,
    )


@app.post("/profile", response_model=ProfileResponse)
async def profile(req: ProfileRequest):
    model = _app_state.get("model")
    if model is None:
        raise HTTPException(503, "Model not loaded")
    model.eval()
    x = torch.randint(0, _app_state["config"].vocab_size, (1, req.seq_len), device=_device)
    profiler = _app_state["profiler"]
    result = profiler.measure_inference(model, x, n_runs=req.n_runs)
    return ProfileResponse(
        mean_latency_ms=result["mean_latency_ms"],
        std_latency_ms=result["std_latency_ms"],
        tokens_per_sec=result["tokens_per_sec"],
        active_params=result["active_params"],
    )


@app.get("/shards", response_model=list[ShardInfo])
async def list_shards():
    engine = _app_state.get("engine")
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    mgr = engine.shard_manager
    shards = mgr.list_expert_shards()
    result = []
    for shard_path in shards:
        fname = shard_path.name
        parts = fname.replace("layer", "").replace("expert", "").replace(".pt", "").split("_")
        result.append(ShardInfo(
            layer=int(parts[0]),
            expert=int(parts[1]),
            size_bytes=shard_path.stat().st_size,
        ))
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
