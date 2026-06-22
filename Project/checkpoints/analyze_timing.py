#!/usr/bin/env python
# Parse per-50-step wall time from chain job logs, plot the throughput trend + ETA.
# --pull rsyncs the latest leg logs from turpan first; --min-job sets the first job
# id of the chain to include (defaults to the current 100k run).
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--pull", action="store_true",
                help="rsync the latest leg logs from turpan first")
ap.add_argument("--min-job", type=int, default=92405,
                help="first job id of the chain to include")
args = ap.parse_args()

TARGET = 100000
MIN_JOB = args.min_job
REMOTE_LOGS = "turpan:job_results/out/job_*.out"
line = re.compile(r"step\s+(\d+)\s+\|\s+loss\s+[\d.]+\s+\|\s+([\d.]+)s")
jobid = re.compile(r"job_(\d+)\.out$")

if args.pull:
    os.makedirs("joblogs", exist_ok=True)
    if subprocess.run(["rsync", "-az", REMOTE_LOGS, "joblogs/"]).returncode:
        print("pull failed — using existing local logs")


def is_current_leg(f):
    """A leg of this run: job id >= MIN_JOB and header says max_steps == TARGET."""
    m = jobid.search(f)
    if not m or int(m.group(1)) < MIN_JOB:
        return False
    head = open(f).read(2000)
    return f"max_steps {TARGET}" in head


rows = []  # (step, dt_per_50, leg)
legs = [f for f in sorted(glob.glob("joblogs/job_*.out")) if is_current_leg(f)]
for f in legs:
    prev_s = prev_t = None
    for ln in open(f):
        m = line.search(ln)
        if not m:
            continue
        s, t = int(m.group(1)), float(m.group(2))
        if prev_t is not None and s - prev_s == 50 and t > prev_t:
            rows.append((s, t - prev_t, f))
        prev_s, prev_t = s, t
print(f"using {len(legs)} chain legs: {[jobid.search(f).group(1) for f in legs]}")

if not rows:
    sys.exit("no step lines parsed")

steps = np.array([r[0] for r in rows])
dt = np.array([r[1] for r in rows])
order = steps.argsort()
steps, dt = steps[order], dt[order]
cur = steps.max()

# robust rate: median over the steady interior 50-step blocks (eval/save spikes excluded by median)
med50 = np.median(dt)
sps = med50 / 50.0
remaining = TARGET - cur
compute_h = remaining * sps / 3600.0
# realized end-to-end incl. eval/save overhead: mean of all 50-blocks
mean50 = dt.mean()
e2e_h = remaining * (mean50 / 50.0) / 3600.0
# per-leg restart overhead: ~13k steps/leg, ~8 min gap each
legs_left = remaining / 13000.0
overhead_h = legs_left * 8 / 60.0
eta_h = e2e_h + overhead_h

print(f"current step      : {cur}")
print(f"median /50 steps  : {med50:.1f}s  ({sps:.3f} s/step)")
print(f"mean   /50 steps  : {mean50:.1f}s  (incl. eval+save spikes)")
print(f"remaining steps   : {remaining}")
print(f"compute-only ETA  : {compute_h:.1f} h")
print(f"end-to-end ETA    : {e2e_h:.1f} h + ~{overhead_h:.1f} h restarts = {eta_h:.1f} h")

# rolling mean for the trend line
w = 15
roll = np.convolve(dt, np.ones(w) / w, mode="valid")
roll_x = steps[w - 1:]

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
ax.scatter(steps, dt, s=10, color="#9ecae1", alpha=0.6, label="per 50 steps")
ax.plot(roll_x, roll, color="#08519c", lw=2.0, label=f"rolling mean ({w})")
ax.axhline(med50, color="#e6550d", ls="--", lw=1.5, label=f"median {med50:.0f}s")
ax.set_xlabel("optimizer step")
ax.set_ylabel("wall time per 50 steps (s)")
ax.set_title(f"VLM chain throughput — step {cur}/{TARGET}, ETA ~{eta_h:.0f}h")
ax.legend(loc="upper right")
fig.savefig("timing.png", dpi=150)
print("saved timing.png")
