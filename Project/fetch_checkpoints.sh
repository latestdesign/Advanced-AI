#!/usr/bin/env bash
# Fetch the VLM checkpoints (/tmpdir scratch, purgeable) into Project/checkpoints/,
# regardless of the directory the script is launched from.
#   - run on Turpan       -> local copy from /tmpdir/<account>/checkpoints
#   - run on another host  -> pulled over ssh from Turpan (account resolved from Turpan)
#   best_step*        = best val-loss weights for inference (save_pretrained)
#   best_mmstar_step* = best MMStar weights for inference (save_pretrained)
# The .pt files (ckpt_step* resume points AND ckpt_milestone* snapshots) carry the
# AdamW + RNG state (~5 GB each, ~3x the weights alone): useless for inference/metrics,
# so they are intentionally skipped here. Pull them by hand if you need to resume.
#
# No argument required. Usage: ./fetch_checkpoints.sh [ssh_host]   (default: turpan)
set -uo pipefail

HOST="${1:-turpan}"
# destination = Project/checkpoints (relative to the script, not the current dir)
DEST="$(cd "$(dirname "$0")" && pwd)/checkpoints"
ME="$(whoami)"

if [ -d "/tmpdir/$ME/checkpoints" ]; then
    # already on Turpan: the checkpoints are local, no ssh needed
    SRC="/tmpdir/$ME/checkpoints"
    echo "Local source (Turpan): $SRC -> $DEST"
else
    # remote machine: the Turpan account is resolved from Turpan itself -> nothing to type
    REMOTE_USER="$(ssh "$HOST" whoami)" || true
    if [ -z "$REMOTE_USER" ]; then
        echo "Cannot reach '$HOST' (account not resolved). Check your ssh config." >&2
        exit 1
    fi
    SRC="$HOST:/tmpdir/$REMOTE_USER/checkpoints"
    echo "Remote source: $SRC -> $DEST"
fi

mkdir -p "$DEST"
# patterns are expanded on the source side (no match -> message, no abort)
rsync -avz --progress "$SRC/best_step"*        "$DEST/" || echo "  (no best_step* yet)"
rsync -avz --progress "$SRC/best_mmstar_step"* "$DEST/" || echo "  (no best_mmstar_step* yet)"
rsync -avz --progress "$SRC/metrics.csv"        "$DEST/" || echo "  (no metrics.csv)"

echo "--- local checkpoints ---"
ls -lh "$DEST"
