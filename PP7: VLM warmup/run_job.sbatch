#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p shared
#SBATCH --output=/users/formation/tpirtlkpl/job_results/out/job_%j.out
#SBATCH --error=/users/formation/tpirtlkpl/job_results/err/job_%j.err

PY_SCRIPT=$1

echo "Job $SLURM_JOB_NAME started (JobID: $SLURM_JOB_ID)"
echo "Running: $PY_SCRIPT $PY_ARGS"

apptainer exec \
    --env PATH=$HOME/.local/bin:$PATH \
    --env UV_PROJECT_ENVIRONMENT=/tmpdir/tpirtlkpl/envs/aai \
    --env UV_NO_SYNC=true \
    --env HF_HOME=/work/shared/TPIRT/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env HF_DATASETS_CACHE=/tmpdir/tpirtlkpl/huggingface/datasets \
    --env TRANSFORMERS_OFFLINE=1 \
    --bind /tmpdir,/work \
    --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
    uv run python "$PY_SCRIPT" $PY_ARGS