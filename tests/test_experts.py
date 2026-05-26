import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.experts import Expert, Experts


def test_expert_output_shape():
    expert = Expert(d_model=128, d_ff=256)
    x = torch.randn(4, 16, 128)
    y = expert(x)
    assert y.shape == (4, 16, 128), f"Expected (4,16,128), got {y.shape}"


def test_experts_output_shape():
    experts = Experts(n_experts=8, d_model=128, d_ff=256)
    x = torch.randn(2, 8, 128)
    indices = torch.randint(0, 8, (2, 8, 2))
    y = experts(x, indices)
    assert y.shape == (2, 8, 128), f"Expected (2,8,128), got {y.shape}"


def test_experts_selective_activation():
    experts = Experts(n_experts=8, d_model=128, d_ff=256)
    x = torch.randn(1, 4, 128)
    indices = torch.zeros(1, 4, 2, dtype=torch.long)
    y = experts(x, indices)
    assert y.shape == (1, 4, 128)
    assert not torch.isnan(y).any()


def test_expert_different_inputs():
    expert = Expert(d_model=128, d_ff=256)
    x1 = torch.randn(2, 4, 128)
    x2 = torch.randn(2, 4, 128)
    y1 = expert(x1)
    y2 = expert(x2)
    assert not torch.allclose(y1, y2), "Different inputs should give different outputs"


def test_heterogeneous_experts():
    dims = [64, 128, 256, 512]
    experts = Experts(n_experts=4, d_model=128, d_ff=256, expert_dims=dims)
    assert len(experts.experts) == 4
    for i, expert in enumerate(experts.experts):
        w1 = expert.net[0].weight
        expected_hidden = 2 * dims[i]
        assert w1.shape[0] == expected_hidden, f"Expert {i}: expected {expected_hidden}, got {w1.shape[0]}"
    x = torch.randn(2, 4, 128)
    indices = torch.randint(0, 4, (2, 4, 2))
    y = experts(x, indices)
    assert y.shape == (2, 4, 128)


def test_heterogeneous_varying_sizes():
    dims = [32, 256]
    experts = Experts(n_experts=2, d_model=64, d_ff=128, expert_dims=dims)
    x = torch.randn(1, 2, 64)
    indices = torch.tensor([[[0, 1]], [[0, 1]]])
    y = experts(x, indices)
    assert y.shape == (1, 2, 64)
    assert experts.experts[0].net[0].weight.shape[0] == 64
    assert experts.experts[1].net[0].weight.shape[0] == 512


if __name__ == "__main__":
    test_expert_output_shape()
    test_experts_output_shape()
    test_experts_selective_activation()
    test_expert_different_inputs()
    test_heterogeneous_experts()
    test_heterogeneous_varying_sizes()
    print("All expert tests passed!")
