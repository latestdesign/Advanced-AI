"""One-time global shuffle of the Cauldron subsets into a single on-disk dataset.

The 47 subsets sit contiguously on disk, so shuffling at read time means random
seeks across the network filesystem on every batch (slow). Instead we pay that
cost once here: concatenate all subsets, shuffle globally, and save the result in
mixed order. Training then reads it sequentially (fast) while still seeing every
subset mixed together.

Run on the login node (has network, not GPU-billed), inside the container:
    uv run python shuffle_dataset.py \
        --src /work/shared/TPIRT/the_cauldron \
        --out /tmpdir/$USER/cauldron_shuffled
Then train with --shuffled_path /tmpdir/$USER/cauldron_shuffled
"""
import argparse
import os

from datasets import DatasetDict, concatenate_datasets, load_from_disk

from models.config import TrainConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/work/shared/TPIRT/the_cauldron",
                    help="dir holding the per-subset Arrow datasets")
    ap.add_argument("--out", default=f"/tmpdir/{os.environ.get('USER', '')}/cauldron_shuffled",
                    help="where to write the shuffled DatasetDict (use scratch)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_proc", type=int, default=8,
                    help="parallel shard writers (match cpus-per-task)")
    args = ap.parse_args()

    cfg = TrainConfig()
    train_splits, val_splits = [], []
    for subset in cfg.dataset_subsets:
        path = os.path.join(args.src, subset)
        if not os.path.exists(path):
            print(f"  [skip] {subset} not found")
            continue
        print(f"  loading {subset}...")
        raw = load_from_disk(path)
        ds = raw["train"] if "train" in raw else raw
        # same contiguous 10% val split as train.py's fallback path
        n_val = int(0.1 * len(ds))
        val_splits.append(ds.select(range(n_val)))
        train_splits.append(ds.select(range(n_val, len(ds))))

    if not train_splits:
        raise ValueError(f"No cauldron subsets found under {args.src}/")

    # shuffle() only sets a lazy index permutation; save_to_disk then materialises
    # the rows in that shuffled order -> a contiguous, globally mixed copy on disk.
    # keep_in_memory: the source lives on read-only /work, so the shuffle index
    # mapping can't be cached next to it — hold it in RAM instead (just an int array).
    train = concatenate_datasets(train_splits).shuffle(seed=args.seed, keep_in_memory=True)
    val = concatenate_datasets(val_splits).shuffle(seed=args.seed, keep_in_memory=True)
    print(f"shuffling {len(train)} train | {len(val)} val -> {args.out}")

    DatasetDict({"train": train, "val": val}).save_to_disk(args.out, num_proc=args.num_proc)
    print("done")


if __name__ == "__main__":
    main()
