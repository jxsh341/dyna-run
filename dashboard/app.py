import streamlit as st
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import ModelConfig, MoEConfig
from src.models.moe_transformer import MoETransformer
from src.models.dense_transformer import DenseTransformer
from src.engine.train import Trainer
from src.engine.inference import InferenceEngine
from src.engine.profiler import Profiler

st.set_page_config(
    page_title="Dyna-Run: Dynamic Sparse AI Runtime",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Dyna-Run")
st.markdown("### Dynamic Sparse AI Inference Runtime")

device = "cuda" if torch.cuda.is_available() else "cpu"

moe_config = MoEConfig(
    n_experts=8, top_k=2, d_model=128, d_ff=256,
    noisy_gating=True, aux_loss_coef=0.01,
)
model_config = ModelConfig(
    vocab_size=512, d_model=128, n_layers=4, n_heads=4,
    max_seq_len=256, moe=moe_config, use_moe=True,
)

if "moe_model" not in st.session_state:
    st.session_state.moe_model = None
if "dense_model" not in st.session_state:
    st.session_state.dense_model = None
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "sample_text" not in st.session_state:
    st.session_state.sample_text = ""
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = {}

def load_or_train():
    if st.session_state.model_ready:
        return
    status = st.status("Initializing Dyna-Run...", expanded=True)
    with status:
        st.write(f"Device: {device}")
        st.write("Creating models...")
        moe_model = MoETransformer(model_config)
        dense_model = DenseTransformer(model_config)
        trainer = Trainer(moe_model, model_config, device)
        ckpt_loaded = trainer.load_checkpoint()
        if ckpt_loaded:
            st.write("Checkpoint loaded!")
        else:
            st.write("No checkpoint found. Training demo MoE on Tiny Shakespeare...")
            sample_text_path = Path("data/sample_text.txt")
            if sample_text_path.exists():
                text = sample_text_path.read_text(encoding="utf-8")
            else:
                text = _get_default_text()
            losses = trainer.train(text, max_steps=1000, seq_len=128, batch_size=16)
            st.write(f"Training complete! Final loss: {losses[-1]:.4f}")
        st.session_state.moe_model = moe_model
        st.session_state.dense_model = dense_model
        st.session_state.model_ready = True
        st.session_state.training_complete = True

def _get_default_text():
    return (
        "To be or not to be that is the question "
        "Whether tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune "
        "or to take arms against a sea of troubles "
        "and by opposing end them To die to sleep "
        "no more and by a sleep to say we end "
        "the heartache and the thousand natural shocks "
        "that flesh is heir to tis a consummation "
        "devoutly to be wishd To die to sleep "
        "to sleep perchance to dream ay theres the rub "
        "for in that sleep of death what dreams may come "
    ) * 50

load_or_train()

page = st.sidebar.radio(
    "Navigation",
    ["Routing Visualization", "Benchmark Dashboard", "External Models"],
    index=0,
)

if page == "Routing Visualization":
    from dashboard.pages.routing_viz import show
    show(st.session_state.moe_model, model_config, device)
elif page == "Benchmark Dashboard":
    from dashboard.pages.benchmark import show
    show(st.session_state.moe_model, st.session_state.dense_model, model_config, device, st.session_state.benchmark_results)
elif page == "External Models":
    from dashboard.pages.external_models import show
    show()
