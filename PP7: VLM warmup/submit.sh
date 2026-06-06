#!/bin/bash
if [ "$#" -lt 2 ]; then
    echo "Usage: ./submit.sh <RES_NUM> <script.py> [job_name] [time] [cpus] [-- python args...]"
    exit 1
fi

RES_NUM=$1
PY_SCRIPT=$2
JOB_NAME=${3:-vlm_train}
TIME_LIMIT=${4:-00:15:00}
CPUS=${5:-8}

PY_ARGS=""
found_separator=false
for arg in "$@"; do
    if $found_separator; then
        PY_ARGS="$PY_ARGS $arg"
    elif [ "$arg" = "--" ]; then
        found_separator=true
    fi
done

export PY_ARGS

sbatch \
    --job-name="$JOB_NAME" \
    --time="$TIME_LIMIT" \
    --cpus-per-task="$CPUS" \
    --reservation="tpirt$RES_NUM" \
    run_job.sbatch "$PY_SCRIPT"