import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.sparse_moe import SparseMoE


def test_sparse_moe_output_shape():
    moe = SparseMoE(d_model=128, n_experts=8, d_ff=256, top_k=2)
    x = torch.randn(2, 10, 128)
    out, gates, indices, aux_loss = moe(x)
    assert out.shape == (2, 10, 128), f"Expected (2,10,128), got {out.shape}"
    assert gates.shape == (2, 10, 8)
    assert indices.shape == (2, 10, 2)
    assert aux_loss.item() >= 0


def test_sparse_moe_dense_mode():
    moe = SparseMoE(d_model=128, n_experts=4, d_ff=256, top_k=2)
    x = torch.randn(1, 5, 128)
    out = moe.forward_dense(x)
    assert out.shape == (1, 5, 128), f"Expected (1,5,128), got {out.shape}"


def test_sparse_moe_returns_gates():
    moe = SparseMoE(d_model=128, n_experts=8, d_ff=256, top_k=2)
    x = torch.randn(1, 8, 128)
    out, gates, indices, aux_loss = moe(x)
    assert (gates >= 0).all(), "All gate weights should be non-negative"
    assert (gates <= 1).all(), "All gate weights should be <= 1"
    assert (indices >= 0).all(), "Indices should be non-negative"
    assert (indices < 8).all(), "Indices should be < n_experts"


def test_sparse_moe_aux_loss():
    moe = SparseMoE(d_model=128, n_experts=8, d_ff=256, top_k=2, aux_loss_coef=0.01)
    x = torch.randn(4, 32, 128)
    out, gates, indices, aux_loss = moe(x)
    assert aux_loss.item() > 0, "Aux loss should be positive"
    assert aux_loss.item() < 1.0, "Aux loss should be small with coefficient 0.01"


if __name__ == "__main__":
    test_sparse_moe_output_shape()
    test_sparse_moe_dense_mode()
    test_sparse_moe_returns_gates()
    test_sparse_moe_aux_loss()
    print("All sparse MoE tests passed!")
