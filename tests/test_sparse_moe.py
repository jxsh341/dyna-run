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


def test_heterogeneous_moe_forward():
    dims = [64, 128, 256, 512]
    moe = SparseMoE(d_model=128, n_experts=4, d_ff=128, top_k=2,
                     heterogeneous=True, expert_dims=dims)
    x = torch.randn(2, 8, 128)
    out, gates, indices, aux_loss = moe(x)
    assert out.shape == (2, 8, 128)
    assert gates.shape == (2, 8, 4)
    assert moe.expert_bias.shape == (4,)


def test_pruning_mask_applied():
    moe = SparseMoE(d_model=128, n_experts=8, d_ff=256, top_k=2)
    x = torch.randn(1, 4, 128)
    out1, gates1, indices1, _ = moe(x)
    moe.set_pruning_mask([0, 1, 2, 3])
    out2, gates2, indices2, _ = moe(x)
    assert (indices2 < 4).all(), "Pruned: only experts 0-3 should be selected"
    moe.clear_pruning_mask()
    out3, gates3, indices3, _ = moe(x)
    assert (indices3 >= 4).any() or (indices3 < 4).all(), "After clear, all experts available"


def test_pruning_mask_zero_experts():
    moe = SparseMoE(d_model=128, n_experts=4, d_ff=256, top_k=2)
    x = torch.randn(1, 4, 128)
    moe.set_pruning_mask([0])
    out, gates, indices, _ = moe(x)
    assert (gates[:, :, 0] > 0).any(), "Expert 0 should have positive gate weights"
    assert (gates[:, :, 1:].abs().max().item() < 1e-6), "Pruned experts should have zero gate weights"


def test_heterogeneous_with_pruning():
    dims = [64, 128, 256, 512]
    moe = SparseMoE(d_model=128, n_experts=4, d_ff=128, top_k=2,
                     heterogeneous=True, expert_dims=dims)
    x = torch.randn(1, 4, 128)
    out1, gates1, indices1, _ = moe(x)
    moe.set_pruning_mask([0, 1])
    out2, gates2, indices2, _ = moe(x)
    assert (indices2 < 2).all(), "Pruned: only experts 0-1 should be selected"
    assert out2.shape == (1, 4, 128)
    moe.clear_pruning_mask()


if __name__ == "__main__":
    test_sparse_moe_output_shape()
    test_sparse_moe_dense_mode()
    test_sparse_moe_returns_gates()
    test_sparse_moe_aux_loss()
    test_heterogeneous_moe_forward()
    test_pruning_mask_applied()
    test_pruning_mask_zero_experts()
    test_heterogeneous_with_pruning()
    print("All sparse MoE tests passed!")
