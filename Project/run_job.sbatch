#!/bin/bash
#SBATCH -J vlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p shared
#SBATCH --time=4:00:00
#SBATCH --output=/users/formation/tpirtlkpl/job_results/out/job_%j.out
#SBATCH --error=/users/formation/tpirtlkpl/job_results/err/job_%j.err

# Run from Project/. Pass the script + its args:
#   sbatch run_job.sbatch train.py --dataset_type cauldron --max_steps 10000
#   sbatch --reservation=tpirtNN run_job.sbatch train.py ...   # priority during a window

echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) -- running: $*"

apptainer exec \
    --env PATH=$HOME/.local/bin:$PATH \
    --env UV_PROJECT_ENVIRONMENT=/tmpdir/tpirtlkpl/envs/aai \
    --env UV_NO_SYNC=true \
    --env HF_HOME=/users/formation/tpirtlkpl/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env HF_DATASETS_CACHE=/tmpdir/tpirtlkpl/huggingface/datasets \
    --env TRANSFORMERS_OFFLINE=1 \
    --bind /tmpdir,/work \
    --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
    uv run python "$@"
