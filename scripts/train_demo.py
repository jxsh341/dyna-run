import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.core.config import ModelConfig, MoEConfig
from src.models.moe_transformer import MoETransformer
from src.engine.train import Trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    moe_config = MoEConfig(n_experts=8, top_k=2, d_model=128, d_ff=256)
    model_config = ModelConfig(
        vocab_size=512, d_model=128, n_layers=4, n_heads=4,
        max_seq_len=256, moe=moe_config,
    )

    model = MoETransformer(model_config)
    trainer = Trainer(model, model_config, device=device)

    sample_path = Path("data/sample_text.txt")
    if sample_path.exists():
        text = sample_path.read_text(encoding="utf-8")
    else:
        text = (
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

    print(f"Training on {len(text)} characters...")
    losses = trainer.train(text, max_steps=1000, seq_len=128, batch_size=16)
    print(f"Done! Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
