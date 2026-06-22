# Project: Build a Vision-Language Model

In this project you will implement a full Vision-Language Model (VLM) from a skeleton codebase.
The goal is to produce a model that can accept an image and a text prompt, and generate a descriptive answer — exactly like small open-source models such as SmolVLM.

You will implement the two backbone encoders (a Vision Transformer and a Language Model), a lightweight modality projector that bridges them, and the training loop. A complete test suite lets you validate each component independently before wiring everything together.

**Groups.** The project is done in groups. Choose one person's GitHub repository as the shared workspace; that person must invite the other group members as collaborators so everyone can push and contribute.

**Infrastructure.** Training will be conducted on Turpan.

Full assignment: https://paulnovello.github.io/Advanced-AI/project/

---

# Out of scope training details

The assignment skeleton leaves the model internals and the training step as `#TODO`
slots. Everything documented below is the **infrastructure built around those TODOs** —
the data pipeline, checkpoint/resume system, training-loop architecture, in-training
evaluation, HPC orchestration, and the correctness/compatibility fixes to the *provided*
scaffolding. This stack is what took the model from "validation loss stuck at 1–2" to a
full ~1-epoch (100k-step) Cauldron run on an A100. Code is named so it can be found, not
quoted verbatim.

## 1. Data pipeline — the convergence fix

The single most important fix. The Cauldron is 47 sub-datasets written back-to-back on
disk. Read in order, **each batch sees a single sub-dataset**: the model specializes on
the current block and forgets the previous ones — train loss falls while validation
**rises**, the "stuck at 1–2" symptom.

- **Global one-time on-disk shuffle** — `shuffle_dataset.py` concatenates all subsets,
  shuffles once with a fixed seed, and `save_to_disk` materializes the rows in mixed
  order. Training then reads **sequentially** (fast, ~0.3 s/step) while every batch still
  mixes dozens of subsets. The discarded alternative — read-time buffer shuffling — either
  stays locally blocky or forces random seeks on the network filesystem (~50× slower,
  13 s/step).
- **Disjoint train/val splits** — `get_dataloaders` builds non-overlapping splits on all
  three data paths (the pre-shuffled Cauldron uses `dsd["val"]`; the flickr and fallback
  paths `select` disjoint ranges), closing a val==train leak.
- **Parallel decode without breaking sequential reads** — `_ShardedCauldron` hands each
  DataLoader worker a *contiguous* shard (`num_workers=16`), so JPEG decode parallelizes
  with no duplication and reads stay sequential.
- **Iterable buffered shuffle** on the flickr/fallback paths (`to_iterable_dataset().shuffle()`)
  where no on-disk pre-shuffle exists.
- **Memory fit** — `max_length=1536` capping in the collator plus
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` keep the run inside A100 80 GB
  (~73 GB plateau).

## 2. Checkpoint & exact-resume system

Built so a 4 h wall-clock kill loses minimal progress and a restart is *exact*.

- **Atomic writes** — `_atomic_save` writes a `.tmp` then `os.replace`, so a crash
  mid-write can never leave a half-written checkpoint that fails to load.
- **Full resume state** — `_ckpt_state` saves weights, AdamW moments (without them the
  first resumed step overshoots), `global_step` (the LR schedule is a pure function of
  it), best-metric trackers, `resume_count`, and python/torch/cuda RNG.
- **Exact data-stream resume** — on restart, `skip_samples = global_step × batch_size ×
  grad_accum` and `get_dataloaders` selects `range(start, n)`, so training *advances
  through the data* instead of replaying the head (confirmed by a `resuming data stream
  at sample N` log).
- **Rotation & two "best" tracks** — `save_checkpoint` keeps the newest
  `keep_checkpoints` (default 3) resume points; `best_step*` (lowest val loss) and
  `best_mmstar_step*` (highest MMStar accuracy) are written via `save_pretrained` and
  pruned to the single best (`prune_best`).
- **Permanent milestones** — `save_milestone` writes `ckpt_milestone*.pt` snapshots that
  rotation never prunes and resume discovery ignores, for later experiments.
- **Robustness** — `find_resume_checkpoint` skips 0-byte/truncated files; a failed save
  (e.g. full disk) is caught and warned, never killing a healthy run; checkpoints go to
  `/tmpdir` scratch because `HOME` has a hard 10 GB quota.

## 3. Metrics logging & analysis

- **Long-form CSV** — `append_metrics` writes `step,metric,value`, so `train_loss`,
  `val_loss`, and `mmstar_acc` — logged at different cadences — append cleanly into one
  plottable file. `truncate_metrics` drops post-checkpoint rows on resume so the curve
  stays monotonic.
- **`checkpoints/plot_metrics.py`** renders train (raw + EMA) / val / MMStar (twin axis).
- **`checkpoints/loss_baselines.py`** contextualizes the loss with three reference CEs:
  step-0 VLM, text-only (image-ablated) and uniform `ln(vocab)`.
- **`checkpoints/analyze_timing.py`** parses per-50-step wall time from job logs into a
  throughput trend and an ETA.

## 4. Training-loop architecture

- **Three LR groups** — the randomly-initialized modality projector trains at a high LR
  (`5e-3`); the pretrained ViT and LM at a low LR (`5e-5`) to preserve their knowledge.
- **Schedule** — `get_lr` does a linear warmup over `warmup_fraction` of steps
  (default 3%) then cosine decay to `max_lr/10`, driven entirely by config.
- **Gradient accumulation** — effective batch = `batch_size × gradient_accumulation_steps`;
  the logged loss is the **accumulation-window mean**, not the last micro-batch, so the
  curve isn't dominated by single-batch noise.
- **Mixed precision** — bf16 `autocast` on CUDA; gradient clipping at `max_grad_norm=1.0`;
  `None` batches from the collator are skipped.

## 5. In-training MMStar evaluation

A periodic `evaluate_mmstar` hook (`--mmstar_eval_interval`) runs inside the eval block:
it scores the MMStar val set, logs `mmstar_acc` to the CSV, dumps a per-step JSON, and
keeps a `best_mmstar_step*` checkpoint. (The 10k checkpoint scored 0.148 — early
letter-choice bias, as expected at that stage.)

## 6. HPC orchestration

- **`run_job.sbatch`** — Apptainer wrapper that runs `uv run python "$@"` with the
  offline HF flags and the scratch-based env the compute nodes require.
- **`chain_train.sh`** — submits a SLURM `afterany` dependency chain of 4 h jobs for runs
  longer than the wall limit (~13k steps/leg → 100k). It lives in the queue, survives
  disconnects with no tmux babysitting, is extendable mid-run via a seed job id, and is
  aborted with `scancel -n vlm`.
- **`fetch_checkpoints.sh`** — location-aware rsync that pulls `best_*` weights and
  `metrics.csv` back into `Project/checkpoints/`.

## 7. Correctness & compatibility fixes to the provided scaffolding

- **`transformers` 4.x/5.x RoPE compatibility** — `LanguageModel.from_pretrained` reads
  `hf.rope_parameters` when present, else falls back to `getattr(hf, "rope_theta",
  100000)`, so it loads on the Turpan container's `transformers` 4.57 *and* on 5.x.
- **HF cache permission workaround** — the shared cache is read-only and missing the
  SmolLM2-360M weights, so the run points `HF_HOME` at `/tmpdir/$USER`.

## 8. Deliberate scope

The connector is a pixel-shuffle×4 + linear projection (SmolVLM-style, 1024→64 visual
tokens); a 2-layer GELU MLP variant (LLaVA-1.5) is planned as an A/B. RoPE scaling keys
on the call sequence length rather than the stricter max-position-id reference —
irrelevant while sequences (≤1536) stay far below `max_position_embeddings=8192`.
