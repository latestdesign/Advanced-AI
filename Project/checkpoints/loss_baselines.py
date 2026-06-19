"""Baseline cross-entropy losses to contextualise the trained VLM.

Three references on the validation split, all in nats (PyTorch CE uses natural
log); perplexity = exp(CE):

  1. step-0 VLM  - pretrained SigLIP2 + SmolLM2 backbones with a randomly
                   initialised modality projector, no training. This is where
                   the loss curve starts; the random projector injects noise
                   visual tokens, so it can sit above the text-only number.
  2. text-only   - same model, image ablated (zeroed pixels): the projector
                   emits a constant, information-free visual token and the
                   decoder must answer from the question text alone. The gap
                   down to the trained loss is what the vision path contributes.
  3. uniform     - ln(vocab_size): the zero-knowledge ceiling, a model that
                   spreads probability equally over every token.

Run on Turpan inside run_apptainer_gpu, from Project/:
    uv run --no-sync python loss_baselines.py \
        --shuffled_path /tmpdir/$USER/cauldron_shuffled --n_batches 64

Results on Turpan (2024-06-05) with 64 batches of 16 examples each:
baseline                   CE (nats)    perplexity
step-0 VLM                    1.6716          5.32
text-only (ablated)           1.6877          5.41
uniform ln(49153)            10.8027      49153.00
"""
import argparse
import math
from contextlib import nullcontext

import torch

from models.config import TrainConfig, VLMConfig
from models.vision_language_model import VisionLanguageModel
from train import get_dataloaders


def measure(model, val_loader, device, n_batches, ablate_image):
    autocast_ctx = (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )
    losses = []
    val_iter = iter(val_loader)
    for _ in range(n_batches):
        batch = next(val_iter, None)
        if batch is None:
            break
        pixel_values = batch["pixel_values"].to(device)
        if ablate_image:
            pixel_values = torch.zeros_like(pixel_values)
        with torch.no_grad(), autocast_ctx:
            _, loss = model(
                batch["input_ids"].to(device),
                pixel_values,
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )
        if loss is not None:
            losses.append(loss.item())
    return sum(losses) / len(losses) if losses else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shuffled_path", default="")
    p.add_argument("--dataset_local_path", default="/work/shared/TPIRT")
    p.add_argument("--dataset_type", default="cauldron")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=1536)
    p.add_argument("--val_size", type=int, default=1024)
    p.add_argument("--n_batches", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_cfg = TrainConfig(
        dataset_type=args.dataset_type,
        dataset_local_path=args.dataset_local_path,
        shuffled_path=args.shuffled_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_size=args.val_size,
        num_workers=0,
    )
    vlm_cfg = VLMConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, val_loader = get_dataloaders(train_cfg, vlm_cfg, seed=args.seed)
    model = VisionLanguageModel(vlm_cfg, load_backbone=True).to(device)
    model.eval()

    step0 = measure(model, val_loader, device, args.n_batches, ablate_image=False)
    text_only = measure(model, val_loader, device, args.n_batches, ablate_image=True)
    uniform = math.log(vlm_cfg.lm.vocab_size)

    print(f"\n{'baseline':<24}{'CE (nats)':>12}{'perplexity':>14}")
    for name, ce in [
        ("step-0 VLM", step0),
        ("text-only (ablated)", text_only),
        (f"uniform ln({vlm_cfg.lm.vocab_size})", uniform),
    ]:
        print(f"{name:<24}{ce:>12.4f}{math.exp(ce):>14.2f}")


if __name__ == "__main__":
    main()
