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


if __name__ == "__main__":
    test_expert_output_shape()
    test_experts_output_shape()
    test_experts_selective_activation()
    test_expert_different_inputs()
    print("All expert tests passed!")
