"""Training script for the VLM project.

Usage:
    python train.py --dataset_local_path /shared/datasets/the_cauldron/ai2d
    python train.py --dataset_type flickr \
        --dataset_local_path /shared/datasets/flickr30k
    python train.py --batch_size 1 --max_steps 100  # quick smoke test

The PROVIDED sections handle:
  * argument parsing and config override
  * data loading (CauldronDataset / FlickrDataset + DataLoader)
  * model construction
  * optimizer setup (3 parameter groups with different learning rates)
  * cosine LR schedule with warmup
  * evaluation loop
  * checkpoint saving

The STUDENT SECTION (clearly marked below) is the inner training loop body.
"""

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import fields

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.config import VLMConfig, TrainConfig
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from data.collator import VQACollator


# ── Cosine LR schedule with linear warmup ────────────────────────────────────
def get_lr(step: int, max_lr: float, max_steps: int, warmup_fraction: float) -> float:
    """Return the learning rate for a given step.

    Phase 1: linear ramp from 0 → max_lr over the first 3% of steps.
    Phase 2: cosine decay from max_lr → max_lr/10 over the remaining steps.
    """
    min_lr = max_lr * 0.1
    warmup_steps = max(1, int(max_steps * warmup_fraction))
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (
        1.0 + math.cos(math.pi * decay)
    )


# ── Data loading (PROVIDED) ───────────────────────────────────────────────────
def get_dataloaders(train_cfg: TrainConfig, vlm_cfg: VLMConfig, seed: int = 42):
    from datasets import load_from_disk, concatenate_datasets

    if not train_cfg.dataset_local_path:
        raise ValueError(
            "dataset_local_path is required. "
            "Run prepare_datasets.py first, then set --dataset_local_path."
        )

    # Resolve the dataset folder under the base path (idempotent; runs here so it
    # survives CLI overrides of dataset_type, unlike a dataclass __post_init__).
    dtype = train_cfg.dataset_type
    if dtype in ('flickr', 'flickr30k') and not train_cfg.dataset_local_path.rstrip('/').endswith('flickr30k'):
        train_cfg.dataset_local_path = os.path.join(train_cfg.dataset_local_path, 'flickr30k')
    elif dtype == 'cauldron' and 'the_cauldron' not in train_cfg.dataset_local_path:
        train_cfg.dataset_local_path = os.path.join(train_cfg.dataset_local_path, 'the_cauldron')

    tokenizer = get_tokenizer(vlm_cfg.lm.tokenizer, vlm_cfg.image_token)
    image_processor = get_image_processor(vlm_cfg.vit.img_size)

    if train_cfg.dataset_type == 'flickr30k' or train_cfg.dataset_type == 'flickr':
        print(f"Loading dataset from disk: {train_cfg.dataset_local_path}")
        raw = load_from_disk(train_cfg.dataset_local_path)
        ds = raw["train"] if "train" in raw else raw
        # contiguous split: leaves _indices=None so iteration stays on the fast
        # sequential Arrow path (a shuffled select forces random per-row seeks).
        n_val = int(0.1 * len(ds))
        val_ds = ds.select(range(n_val))
        train_ds = ds.select(range(n_val, len(ds)))

        # fix: shuffle via an iterable view so batches mix the split (reads stay
        # sequential). cap shards at set size for tiny datasets.
        train_ds = train_ds.to_iterable_dataset(num_shards=min(8, len(train_ds))).shuffle(buffer_size=10000, seed=seed)
        val_ds = val_ds.to_iterable_dataset(num_shards=min(8, len(val_ds))).shuffle(buffer_size=2000, seed=42)

        from data.dataset import FlickrDataset
        train_dataset = FlickrDataset(
            train_ds, tokenizer, image_processor, vlm_cfg
        )
        val_dataset = FlickrDataset(
            val_ds, tokenizer, image_processor, vlm_cfg
        )
    else:
        # Load each subset, split it contiguously (keeps _indices=None so the
        # concatenated train set iterates on the fast sequential Arrow path),
        # then concatenate. Per-subset split also stratifies val across subsets.
        train_splits, val_splits = [], []
        base_path = train_cfg.dataset_local_path
        for subset in train_cfg.dataset_subsets:
            subset_path = os.path.join(base_path, subset)
            if not os.path.exists(subset_path):
                print(f"  [skip] {subset} not found at {subset_path}")
                continue
            print(f"  Loading {subset}...")
            raw = load_from_disk(subset_path)
            ds = raw["train"] if "train" in raw else raw
            n_val = int(0.1 * len(ds))
            val_splits.append(ds.select(range(n_val)))
            train_splits.append(ds.select(range(n_val, len(ds))))

        if not train_splits:
            raise ValueError(
                f"No cauldron subsets found under {base_path}/. "
                "Run prepare_datasets.py first."
            )

        train_ds = concatenate_datasets(train_splits)
        val_ds = concatenate_datasets(val_splits)
        print(
            f"Concatenated {len(train_splits)} subsets → "
            f"{len(train_ds)} train | {len(val_ds)} val samples"
        )

        # fix: buffered shuffle on an iterable view so batches mix the 47 subsets;
        # reads stay sequential (was serving subsets contiguously in alpha order ->
        # a short run only saw ai2d and loss stayed stuck). cap shards at set size.
        train_ds = train_ds.to_iterable_dataset(num_shards=min(8, len(train_ds))).shuffle(buffer_size=10000, seed=seed)
        val_ds = val_ds.to_iterable_dataset(num_shards=min(8, len(val_ds))).shuffle(buffer_size=2000, seed=42)

        from data.dataset import CauldronDataset
        train_dataset = CauldronDataset(
            train_ds, tokenizer, image_processor, vlm_cfg
        )
        val_dataset = CauldronDataset(
            val_ds, tokenizer, image_processor, vlm_cfg
        )

    collator = VQACollator(tokenizer, max_length=train_cfg.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True,
    )
    return train_loader, val_loader


# ── Resume checkpointing ──────────────────────────────────────────────────────
def _ckpt_step(path):
    # extract the step number from a ckpt_step<N>.pt filename, for sorting
    return int(re.search(r"ckpt_step(\d+)\.pt$", path).group(1))


def find_resume_checkpoint(path):
    # accept either an explicit .pt file or a directory (pick the newest ckpt there)
    if os.path.isdir(path):
        # skip 0-byte files: a truncated write (e.g. disk full) must not be picked
        # as the newest and then fail to load.
        ckpts = [p for p in glob.glob(os.path.join(path, "ckpt_step*.pt"))
                 if os.path.getsize(p) > 0]
        return max(ckpts, key=_ckpt_step) if ckpts else None
    return path if os.path.exists(path) else None


def save_checkpoint(checkpoint_dir, model, optimizer, global_step, best_val_loss,
                    best_mmstar_acc, resume_count, device, keep_last):
    # everything needed to continue training byte-for-byte: weights, AdamW moments
    # (without these the first resumed step overshoots), the step counter (the LR
    # schedule is a pure function of it), best-metric trackers, and RNG so dropout
    # etc. line up. data cursor is intentionally not saved (see train()).
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_mmstar_acc": best_mmstar_acc,
        "resume_count": resume_count,
        "rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    # write to a temp file then os.replace: the swap is atomic, so a crash mid-write
    # can never leave a half-written checkpoint that fails to load.
    path = os.path.join(checkpoint_dir, f"ckpt_step{global_step}.pt")
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    # rotate: keep only the newest keep_last checkpoints (each is ~weights+optimizer,
    # several GB) so we always have a few rollback points without filling scratch.
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt_step*.pt")),
                   key=_ckpt_step)
    for old in ckpts[:-keep_last]:
        os.remove(old)


def prune_best(checkpoint_dir, prefix, keep_path):
    # we only write a best-folder on an improvement, so the newest one IS the best
    # of all time; drop the older ones (each holds full ~1.7GB weights).
    for old in glob.glob(os.path.join(checkpoint_dir, f"{prefix}*")):
        if old != keep_path:
            shutil.rmtree(old, ignore_errors=True)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    random.setstate(ckpt["rng"])
    # map_location moved these onto the GPU; set_rng_state needs CPU ByteTensors
    torch.set_rng_state(ckpt["torch_rng"].cpu())
    if device.type == "cuda" and ckpt["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
    print(f"Resumed from {path} at step {ckpt['global_step']}")
    return (ckpt["global_step"], ckpt["best_val_loss"],
            ckpt["best_mmstar_acc"], ckpt["resume_count"])


# ── Main training function ────────────────────────────────────────────────────
def train(train_cfg: TrainConfig, vlm_cfg: VLMConfig):
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.backends.mps.enable_fallback_to_cpu = True
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VisionLanguageModel(
        vlm_cfg, load_backbone=vlm_cfg.load_backbone_weights
    )
    model.to(device)
    if train_cfg.compile:
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # ── Optimizer — three learning-rate groups ────────────────────────────────
    # Why three groups?
    #   * The modality projector (MP) is randomly initialised → high LR
    #   * The ViT and LM are pretrained → low LR to preserve knowledge
    param_groups = [
        {
            "params": list(model.MP.parameters()),
            "lr": train_cfg.lr_mp,
            "name": "MP",
        },
        {
            "params": list(model.vision_encoder.parameters()),
            "lr": train_cfg.lr_vit,
            "name": "ViT",
        },
        {
            "params": list(model.decoder.parameters()),
            "lr": train_cfg.lr_lm,
            "name": "LM",
        },
    ]
    # max_lrs: the initial (maximum) LR per group, used inside get_lr().
    # Students reference this in TODO 5.
    max_lrs = [  # noqa: F841
        train_cfg.lr_mp, train_cfg.lr_vit, train_cfg.lr_lm
    ]
    optimizer = optim.AdamW(param_groups)
    # all_params: flat list of parameters for gradient clipping.
    # Students reference this in TODO 5.
    all_params = [  # noqa: F841
        p for g in optimizer.param_groups for p in g["params"]
    ]

    # ── Resume (load before building data so we can bump the shuffle seed) ──────
    global_step = 0
    best_val_loss = float("inf")
    best_mmstar_acc = -1.0
    resume_count = 0
    if train_cfg.resume_from:
        ckpt_path = find_resume_checkpoint(train_cfg.resume_from)
        if ckpt_path is not None:
            global_step, best_val_loss, best_mmstar_acc, resume_count = load_checkpoint(
                ckpt_path, model, optimizer, device
            )
            resume_count += 1   # count this resume so the seed below changes
        else:
            print(f"No checkpoint found at {train_cfg.resume_from}; starting fresh")

    # ── Data ──────────────────────────────────────────────────────────────────
    # The data cursor isn't checkpointed: with num_workers=1 it lives in the worker
    # process (unreachable from here), and a full run sees <10% of one epoch, so a
    # little replay is harmless. Instead we bump the shuffle seed per resume -> a
    # fresh order each time -> better coverage rather than replaying the same head.
    train_loader, val_loader = get_dataloaders(
        train_cfg, vlm_cfg, seed=42 + resume_count
    )
    iter_train = iter(train_loader)

    # ── AMP context ───────────────────────────────────────────────────────────
    autocast_dtype = (
        torch.bfloat16 if device.type in ("cuda", "cpu") else torch.float16
    )
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=autocast_dtype
    )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    # ── Training state (global_step/best_* set above so resume can restore them) ─
    batch_loss = 0.0   # set by the student section each micro-step
    accum_loss_sum = 0.0   # fix: running sum to log the accumulation-window mean
    optimizer.zero_grad()

    print(
        f"Training for {train_cfg.max_steps} optimiser steps "
        f"(gradient_accumulation={train_cfg.gradient_accumulation_steps})"
    )
    t0 = time.time()

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    accum_step = 0   # counts micro-steps within one accumulation cycle
    while global_step < train_cfg.max_steps:
        model.train()

        # ── Get next batch (skip None batches from the collator) ──────────────
        batch = None
        while batch is None:
            try:
                batch = next(iter_train)
            except StopIteration:
                iter_train = iter(train_loader)
                batch = next(iter_train)

        is_update_step = (
            (accum_step + 1) % train_cfg.gradient_accumulation_steps == 0
        )

        # ══════════════════════════════════════════════════════════════════════
        # STUDENT SECTION — implement the training step
        #
        # TODO 1 — Move tensors to device:
        input_ids      = batch["input_ids"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # TODO 2 — Forward pass (inside autocast_ctx for mixed precision):
        with autocast_ctx:
            _, loss = model(
                input_ids, pixel_values, attention_mask, labels
            )

        # TODO 3 — Scale loss for gradient accumulation:
        loss = loss / train_cfg.gradient_accumulation_steps

        # TODO 4 — Backward pass:
        loss.backward()

        # TODO 5 — Optimiser step (only on update steps):
        if is_update_step:
            torch.nn.utils.clip_grad_norm_(
                all_params, train_cfg.max_grad_norm
            )
            for g, max_lr in zip(optimizer.param_groups, max_lrs):
                g["lr"] = get_lr(
                    global_step, max_lr, train_cfg.max_steps, train_cfg.warmup_fraction
                )
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # TODO 6 — Store the unscaled loss for logging:
        # fix: average over the accumulation window instead of logging only the
        # last micro-batch (one batch_size=N sample made the curve very noisy)
        accum_loss_sum += loss.item() * train_cfg.gradient_accumulation_steps
        if is_update_step:
            batch_loss = accum_loss_sum / train_cfg.gradient_accumulation_steps
            accum_loss_sum = 0.0
        # ══════════════════════════════════════════════════════════════════════

        accum_step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if is_update_step and global_step % train_cfg.log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"step {global_step:6d} | loss {batch_loss:.6f}"
                f" | {elapsed:.1f}s"
            )

        # ── Periodic resume checkpoint (step-tagged, keeps newest keep_checkpoints) ─
        if is_update_step and global_step % train_cfg.save_interval == 0:
            try:
                save_checkpoint(
                    train_cfg.checkpoint_dir, model, optimizer, global_step,
                    best_val_loss, best_mmstar_acc, resume_count, device,
                    train_cfg.keep_checkpoints,
                )
            except Exception as e:
                # a failed save (e.g. full disk) must not kill a healthy run
                print(f"  [warn] checkpoint save failed: {e}")

        # ── Evaluation ────────────────────────────────────────────────────────
        if is_update_step and global_step % train_cfg.eval_interval == 0:
            model.eval()
            val_losses = []
            val_iter = iter(val_loader)
            n_val = min(64, train_cfg.val_size // train_cfg.batch_size)
            for _ in range(n_val):
                vbatch = next(val_iter, None)
                if vbatch is None:
                    break
                with torch.no_grad(), autocast_ctx:
                    _, vloss = model(
                        vbatch["input_ids"].to(device),
                        vbatch["pixel_values"].to(device),
                        vbatch["attention_mask"].to(device),
                        vbatch["labels"].to(device),
                    )
                if vloss is not None:
                    val_losses.append(vloss.item())

            avg_val = (
                sum(val_losses) / len(val_losses)
                if val_losses else float("nan")
            )
            print(f"step {global_step:6d} | val_loss {avg_val:.6f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                ckpt = os.path.join(
                    train_cfg.checkpoint_dir, f"best_step{global_step}"
                )
                model.save_pretrained(ckpt)
                print(f"  → new best checkpoint saved to {ckpt}")
                prune_best(train_cfg.checkpoint_dir, "best_step", ckpt)

            if (
                train_cfg.mmstar_val_path
                and train_cfg.mmstar_eval_interval > 0
                and global_step % train_cfg.mmstar_eval_interval == 0
            ):
                from datasets import load_from_disk

                from eval_mmstar import evaluate_mmstar

                tokenizer = get_tokenizer(vlm_cfg.lm.tokenizer, vlm_cfg.image_token)
                image_processor = get_image_processor(vlm_cfg.vit.img_size)
                raw_mmstar = load_from_disk(train_cfg.mmstar_val_path)
                mmstar_val = raw_mmstar["val"] if "val" in raw_mmstar else raw_mmstar
                mmstar_metrics = evaluate_mmstar(
                    model=model,
                    dataset=mmstar_val,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    device=device,
                    limit=train_cfg.mmstar_eval_limit,
                    show_progress=False,
                )
                mmstar_acc = mmstar_metrics["accuracy"]
                print(
                    f"step {global_step:6d} | mmstar_val_acc "
                    f"{mmstar_acc:.6f}"
                )

                os.makedirs(train_cfg.mmstar_output_dir, exist_ok=True)
                mmstar_path = os.path.join(
                    train_cfg.mmstar_output_dir,
                    f"mmstar_step{global_step}.json",
                )
                with open(mmstar_path, "w") as f:
                    json.dump(
                        {
                            "global_step": global_step,
                            "checkpoint_dir": train_cfg.checkpoint_dir,
                            "mmstar_val_path": train_cfg.mmstar_val_path,
                            "metrics": mmstar_metrics,
                        },
                        f,
                        indent=2,
                    )

                if mmstar_acc > best_mmstar_acc:
                    best_mmstar_acc = mmstar_acc
                    ckpt = os.path.join(
                        train_cfg.checkpoint_dir,
                        f"best_mmstar_step{global_step}",
                    )
                    model.save_pretrained(ckpt)
                    print(f"  → new best MMStar checkpoint saved to {ckpt}")
                    prune_best(train_cfg.checkpoint_dir, "best_mmstar_step", ckpt)

            model.train()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the VLM project model"
    )
    for f in fields(TrainConfig):
        if f.type in (int, float, bool, str):
            parser.add_argument(
                f"--{f.name}",
                type=f.type,
                default=None,
                help=f"TrainConfig.{f.name} (default: {f.default})",
            )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()
    for f in fields(TrainConfig):
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(train_cfg, f.name, val)
    train(train_cfg, vlm_cfg)
