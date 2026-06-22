#!/usr/bin/env bash
# chain_train.sh — submit a dependency chain of 4h SLURM jobs that resume the VLM
# training until --max_steps is reached. Each job resumes from the latest checkpoint;
# the afterany dependency fires the next one when the previous ends (timeout/done/fail),
# so they run back-to-back. No tmux/login-node babysitting: it lives in the SLURM queue
# and survives disconnects. The first job (empty checkpoint dir) starts fresh.
#   ./chain_train.sh [n_jobs] [max_steps] [save_interval] [milestone_interval] [mmstar_interval] [after_jobid]
# after_jobid: seed the first job on an external running job (afterany), to extend a
# live chain without disturbing it. Abort the whole chain:  scancel -n vlm
set -euo pipefail
N="${1:-12}"
MAX_STEPS="${2:-11000}"
SAVE="${3:-100}"
MILE="${4:-1000}"
MMSTAR="${5:-500}"
AFTER="${6:-}"
PROJ="$HOME/Project-Advanced-AI/Project"
CKPT="/tmpdir/$USER/checkpoints"
SHUF="/tmpdir/$USER/cauldron_shuffled"

cd "$PROJ"
dep="$AFTER"
for i in $(seq 1 "$N"); do
  args=(run_job.sbatch train.py
    --shuffled_path "$SHUF" --checkpoint_dir "$CKPT" --resume_from "$CKPT"
    --metrics_file "$CKPT/metrics.csv" --max_steps "$MAX_STEPS"
    --save_interval "$SAVE" --milestone_interval "$MILE"
    --mmstar_eval_interval "$MMSTAR")
  if [ -z "$dep" ]; then
    jid=$(sbatch --parsable "${args[@]}")
  else
    jid=$(sbatch --parsable --dependency=afterany:"$dep" "${args[@]}")
  fi
  echo "job $i: $jid (after ${dep:-none})"
  dep="$jid"
done
echo "chain of $N jobs submitted; abort all with: scancel -n vlm"
