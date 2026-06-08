#!/bin/bash
#SBATCH -J mon_job
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p shared
<<<<<<< HEAD
#SBATCH --time=00:15:00 
#SBATCH --output=/users/formation/YOUR_USERNAME/job_results/out/job_%j.out
#SBATCH --error=/users/formation/YOUR_USERNAME/job_results/err/job_%j.err
=======
#SBATCH --time=00:15:00
#SBATCH --output=/users/formation/tpirtlkpl/job_results/out/job_%j.out
#SBATCH --error=/users/formation/tpirtlkpl/job_results/err/job_%j.err
>>>>>>> main


apptainer exec \
--env PATH=$HOME/.local/bin:$PATH \
<<<<<<< HEAD
--env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai \
=======
--env UV_PROJECT_ENVIRONMENT=/tmpdir/tpirtlkpl/envs/aai \
>>>>>>> main
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
uv run --no-sync python mon_script.py