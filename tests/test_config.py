import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import ModelConfig, MoEConfig


def test_moe_config_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        MoEConfig(n_experts=4, top_k=5)


def test_moe_config_validates_expert_dims_length():
    with pytest.raises(ValueError, match="expert_dims"):
        MoEConfig(n_experts=4, expert_dims=[64, 128])


def test_model_config_requires_attention_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=130, n_heads=8, moe=MoEConfig(d_model=130))


def test_model_config_requires_matching_moe_dimension():
    with pytest.raises(ValueError, match="moe.d_model"):
        ModelConfig(d_model=64, moe=MoEConfig(d_model=128))
