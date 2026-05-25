import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.router import Router


def test_router_output_shape():
    router = Router(d_model=128, n_experts=8, noisy_gating=True)
    x = torch.randn(2, 16, 128)
    gates, indices = router(x)
    assert gates.shape == (2, 16, 8), f"Expected (2,16,8), got {gates.shape}"
    assert indices.shape == (2, 16, 2), f"Expected (2,16,2), got {indices.shape}"


def test_router_probabilities_sum_to_one():
    router = Router(d_model=128, n_experts=8, noisy_gating=True)
    router.eval()
    x = torch.randn(1, 10, 128)
    gates, indices = router(x)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(1, 10), atol=1e-5)


def test_router_top_k():
    router = Router(d_model=128, n_experts=8, noisy_gating=False)
    router.eval()
    x = torch.randn(1, 5, 128)
    gates, indices = router(x)
    for i in range(indices.shape[1]):
        token_gates = gates[0, i]
        nonzero = (token_gates > 0).sum().item()
        assert nonzero <= 2, f"Token {i} has {nonzero} non-zero gates (expected <= 2)"


def test_router_load_balancing_loss():
    router = Router(d_model=128, n_experts=8, noisy_gating=True)
    x = torch.randn(4, 32, 128)
    gates, indices = router(x)
    loss = router.load_balancing_loss(gates, indices)
    assert loss.item() > 0, "Load balancing loss should be positive"


if __name__ == "__main__":
    test_router_output_shape()
    test_router_probabilities_sum_to_one()
    test_router_top_k()
    test_router_load_balancing_loss()
    print("All router tests passed!")
