#!/usr/bin/env bash
# Récupère les checkpoints VLM (/tmpdir scratch, purgeable) vers Project/checkpoints/,
# peu importe le dossier d'où on lance le script.
#   - lancé sur Turpan        -> copie locale depuis /tmpdir/<compte>/checkpoints
#   - lancé sur autre machine -> tire par ssh depuis Turpan (compte demandé à Turpan)
#   best_step*    = meilleurs poids pour l'inférence (format save_pretrained)
#   ckpt_step*.pt = état complet de reprise (poids + AdamW + step)
#
# Aucun argument requis. Usage : ./fetch_checkpoints.sh [hote_ssh]   (défaut: turpan)
set -uo pipefail

HOST="${1:-turpan}"
# destination = Project/checkpoints (relatif au script, pas au répertoire courant)
DEST="$(cd "$(dirname "$0")" && pwd)/checkpoints"
ME="$(whoami)"

if [ -d "/tmpdir/$ME/checkpoints" ]; then
    # on est déjà sur Turpan : les checkpoints sont locaux, pas de ssh
    SRC="/tmpdir/$ME/checkpoints"
    echo "Source locale (Turpan) : $SRC -> $DEST"
else
    # machine distante : le compte Turpan vient de Turpan lui-même -> rien à saisir
    REMOTE_USER="$(ssh "$HOST" whoami)" || true
    if [ -z "$REMOTE_USER" ]; then
        echo "Impossible de joindre '$HOST' (compte non résolu). Vérifiez votre config ssh." >&2
        exit 1
    fi
    SRC="$HOST:/tmpdir/$REMOTE_USER/checkpoints"
    echo "Source distante : $SRC -> $DEST"
fi

mkdir -p "$DEST"
# les motifs sont développés côté source (pas de correspondance -> message, pas d'arrêt)
rsync -avz --progress "$SRC/best_step"*   "$DEST/" || echo "  (pas encore de best_step*)"
rsync -avz --progress "$SRC/ckpt_step"*.pt "$DEST/" || echo "  (pas encore de ckpt_step*.pt)"
rsync -avz --progress "$SRC/ckpt_milestone"*.pt "$DEST/" || echo "  (pas de ckpt_milestone*.pt)"
rsync -avz --progress "$SRC/metrics.csv"   "$DEST/" || echo "  (pas de metrics.csv)"

echo "--- checkpoints locaux ---"
ls -lh "$DEST"
