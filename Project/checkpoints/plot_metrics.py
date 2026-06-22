#!/usr/bin/env python
"""Plot train/val/mmstar curves from the long-form metrics.csv.

CSV layout (written by train.py): step,metric,value with metric in
{train_loss, val_loss, mmstar_acc}. Run after fetch_checkpoints.sh has
pulled metrics.csv into Project/checkpoints/.

    python plot_metrics.py                       # checkpoints/metrics.csv -> metrics.png
    python plot_metrics.py path/to/metrics.csv   # custom file
    python plot_metrics.py --ema 0.9 --show
"""
import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REMOTE_METRICS = "turpan:/tmpdir/tpirtgstrt/checkpoints/metrics.csv"


def ema(series, alpha):
    """Exponential moving average for a noisy per-step series."""
    return series.ewm(alpha=1 - alpha).mean()


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default=str(here / "metrics.csv"))
    p.add_argument("--out", default="", help="output image (default: <csv>.png)")
    p.add_argument("--ema", type=float, default=0.9,
                   help="smoothing for raw train loss in [0,1); higher = smoother")
    p.add_argument("--show", action="store_true")
    p.add_argument("--pull", action="store_true",
                   help=f"scp {REMOTE_METRICS} into the csv path first")
    args = p.parse_args()

    csv = Path(args.csv)
    if args.pull:
        csv.parent.mkdir(parents=True, exist_ok=True)
        if subprocess.run(["scp", "-q", REMOTE_METRICS, str(csv)]).returncode:
            print("pull failed — using existing local file")
    df = pd.read_csv(csv)
    wide = df.pivot_table(index="step", columns="metric", values="value")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    if "train_loss" in wide:
        tr = wide["train_loss"].dropna()
        ax.plot(tr.index, tr.values, color="#9ecae1", lw=1.0, alpha=0.9,
                label="train (raw)")
        sm = ema(tr, args.ema)
        ax.plot(sm.index, sm.values, color="#08519c", lw=2.0,
                label=f"train (ema {args.ema})")
    if "val_loss" in wide:
        vl = wide["val_loss"].dropna()
        ax.plot(vl.index, vl.values, color="#e6550d", lw=2.0, marker="o",
                ms=4, label="val")

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"VLM training — {csv.parent.name}/{csv.name}")

    handles, labels = ax.get_legend_handles_labels()

    if "mmstar_acc" in wide:
        mm = wide["mmstar_acc"].dropna()
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.plot(mm.index, mm.values, color="#31a354", lw=2.0, marker="s",
                 ms=4, label="mmstar acc")
        ax2.set_ylabel("MMStar accuracy")
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    ax.legend(handles, labels, loc="upper right", framealpha=0.9)

    out = Path(args.out) if args.out else csv.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
