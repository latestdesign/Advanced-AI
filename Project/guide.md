# Guide — entraînement du VLM sur Turpan

Workflow pour mélanger le dataset, entraîner, reprendre après coupure et récupérer
les poids. Les scripts sont génériques (`$USER` / `%u` / `$HOME`) : rien à adapter
par compte, on les lance tels quels. Les alias/fonctions cités sont en **annexe**.

## L'idée clé : pré-mélanger une fois

The Cauldron, ce sont 47 sous-ensembles écrits les uns après les autres sur disque.

**Ce qui ne marche pas :**
- *Concaténer et lire dans l'ordre* (approche initiale) : chaque batch ne contient
  qu'un seul sous-ensemble à la fois. Le modèle se spécialise sur le bloc courant
  puis « oublie » les précédents → la loss d'entraînement baisse mais la validation
  **monte**. Mauvaise convergence.
- *Mélanger à la lecture* (buffer shuffle, `interleave_datasets`, beaucoup de
  shards) : soit le mélange reste local (encore des blocs), soit on force des
  lectures aléatoires sur le système de fichiers réseau — **~50x plus lent**
  (13 s/step au lieu de 0,3).

**Ce qui marche :** mélanger globalement **une seule fois** et réécrire le dataset
dans cet ordre (`shuffle_dataset.py`). L'entraînement lit ensuite **séquentiellement**
(rapide) un ordre déjà mélangé, donc chaque batch voit des dizaines de sous-ensembles.

**Correctif côté `train.py` (résumé, pour qui voudrait relancer)** : le dataset
pré-mélangé est chargé en *map-style* et itéré dans l'ordre ; à la reprise, on **saute
les échantillons déjà consommés** (`global_step × batch_size`, via `select(range(...))`)
au lieu de repartir du début. Le mélange assure la convergence, le saut assure une
reprise exacte — les deux sont nécessaires.

## 0. Prérequis (une fois, par personne)

Tout passe par le conteneur Apptainer
`/work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif`. Le scratch
`/tmpdir` est purgeable mais sans quota ; le `HOME` a un quota dur de 10 Go, donc
**toujours écrire dataset et checkpoints dans `/tmpdir/$USER/`**.

1. Accès ssh à Turpan configuré (voir **Annexe — accès ssh**).
2. Cloner le repo sur Turpan (`~/Advanced-AI`).
3. Copier les alias de l'**annexe** dans `~/.bashrc`, puis `source ~/.bashrc`.
4. Créer les dossiers de logs SLURM (sinon le job échoue à écrire) :
   ```bash
   mkdir -p ~/job_results/out ~/job_results/err
   ```
5. Installer `uv` s'il est absent. Il s'installe dans `~/.local/bin`, **absent du
   `PATH` du shell courant** : il faut le rafraîchir tout de suite, sinon `uv` reste
   introuvable (y compris dans `log_apptainer`, qui hérite de l'environnement courant) :
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env       # met uv sur le PATH sans rouvrir de shell
   ```
   (Si vous aviez déjà lancé `log_apptainer` avant ce `source`, sortez-en puis
   rentrez-y : le conteneur fige le `PATH` au moment où on l'ouvre.)
6. Entrer dans le conteneur du nœud de login avec `log_apptainer`, puis construire
   l'environnement sur le scratch :
   ```bash
   uv venv --python=/usr/bin/python --system-site-packages /tmpdir/$USER/envs/aai
   uv sync --only-group turpan
   ```
7. Toujours dans le conteneur login (réseau disponible), télécharger les backbones
   dans le cache HF — les nœuds de calcul sont hors-ligne, il faut donc les avoir en
   cache **avant** l'entraînement :
   ```bash
   HF_HUB_OFFLINE=0 uv run --no-sync python -c "from transformers import AutoModel; \
     AutoModel.from_pretrained('HuggingFaceTB/SmolLM2-360M-Instruct'); \
     AutoModel.from_pretrained('google/siglip2-base-patch16-512')"
   ```

## 1. Pré-mélange du dataset (une seule fois)

Dans le conteneur login (`log_apptainer`), depuis `Project/` :
```bash
uv run --no-sync python shuffle_dataset.py \
  --src /work/shared/TPIRT/the_cauldron \
  --out /tmpdir/$USER/cauldron_shuffled
```
L'opération relit tout le dataset dans l'ordre mélangé pour le réécrire : c'est long
mais ne se fait qu'une fois. (Voir « L'idée clé » ci-dessus pour le pourquoi.)

## 2. Entraînement

Depuis `Project/` :
```bash
sbatch run_job.sbatch train.py \
  --shuffled_path /tmpdir/$USER/cauldron_shuffled \
  --checkpoint_dir /tmpdir/$USER/checkpoints \
  --resume_from /tmpdir/$USER/checkpoints
```
`--dataset_type cauldron` et `--max_steps 10000` sont déjà les valeurs par défaut.
Un checkpoint de reprise complet est écrit tous les 2500 steps, et un dossier
`best_step*` à chaque amélioration de la validation.

Raccourci équivalent (alias `train_vlm` de l'annexe) : `train_vlm`, qui accepte aussi
des surcharges, ex. `train_vlm --max_steps 20000`.

## 3. Suivi

Avec les alias de l'annexe :
```bash
sq            # mes jobs (squeue -u $USER -l)
jlog          # suit le stdout du dernier job
jerr          # suit le stderr du dernier job
stats         # usage GPU échantillonné ~30 s
mon           # tmux : GPU (haut) + log live (bas), détacher avec Ctrl-b d
```

## 4. Reprise après coupure

Relancez exactement la même commande qu'en section 2. `--resume_from` reprend le
dernier `ckpt_step*.pt`, restaure poids + optimiseur + step, et **saute les
échantillons déjà consommés** : on continue dans le flux de données au lieu de
réapprendre le début. Une ligne `resuming data stream at sample N/...` le confirme.

## 5. Récupérer les checkpoints en local

Aucun argument requis, le script s'adapte tout seul :
```bash
./fetch_checkpoints.sh
```
- Lancé **sur Turpan** : copie locale depuis `/tmpdir/<compte>/checkpoints` (vers
  `Project/checkpoints/`, qui est sur le HOME — attention au quota de 10 Go).
- Lancé **sur votre machine** : le compte est demandé à Turpan puis les poids sont
  tirés par ssh.

Comme `/tmpdir` est purgeable, récupérez vos poids dès qu'un entraînement intéressant
est fini.

## 6. Charger un checkpoint

Pour l'inférence ou l'API, à partir d'un dossier `best_step*` (format
`save_pretrained`) :
```python
from models.vision_language_model import VisionLanguageModel
model = VisionLanguageModel.from_pretrained("checkpoints/best_stepXXXX")
```

## Annexe — alias et fonctions `~/.bashrc`

À copier dans `~/.bashrc` puis `source ~/.bashrc`. Tout dérive de `$USER`/`$HOME`,
donc valable pour n'importe quel compte de la formation.

```bash
# Cache Hugging Face global (hors conteneur)
export HF_HOME=$HOME/.cache/huggingface

# Conteneur interactif sur le nœud de login (CPU) : env build, tests, pré-mélange
alias log_apptainer="apptainer shell \
  --env PATH=$HOME/.local/bin:$PATH \
  --env UV_PROJECT_ENVIRONMENT=/tmpdir/$USER/envs/aai \
  --env HF_HOME=$HOME/.cache/huggingface \
  --bind /tmpdir,/work \
  --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif"

# Conteneur interactif avec GPU (nœud de calcul, hors-ligne) : smoke tests, debug
alias run_apptainer_gpu="srun -p shared -n1 --gres=gpu:1 --pty apptainer shell \
  --env PATH=$HOME/.local/bin:$PATH \
  --env UV_PROJECT_ENVIRONMENT=/tmpdir/$USER/envs/aai \
  --env UV_NO_SYNC=true \
  --env HF_HOME=$HOME/.cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --bind /tmpdir,/work \
  --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif"

# Soumettre un job depuis Project/ (sans changer de répertoire courant)
subjob() { ( cd ~/Advanced-AI/Project && sbatch "$@" ); }

alias sq='squeue -u $USER -l'                       # mes jobs

# Suivre le stdout / stderr du dernier job
jlog() { tail -n "${1:-40}" -f "$(ls -t ~/job_results/out/job_*.out 2>/dev/null | head -1)"; }
jerr() { tail -n "${1:-40}" -f "$(ls -t ~/job_results/err/job_*.err 2>/dev/null | head -1)"; }

# Usage GPU du job en cours (échantillonne N s, défaut 30)
stats() {
  local jid="${JOBID:-$(squeue -u $USER -h -t R -o "%i" | head -1)}"
  [ -z "$jid" ] && { echo "no running job"; return 1; }
  timeout "${1:-30}" srun --jobid="$jid" --overlap \
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv -l 1
}

# tmux : GPU (haut) + log live (bas). détacher : Ctrl-b puis d
mon() {
  local jid="${JOBID:-$(squeue -u $USER -h -t R -o '%i' | head -1)}"
  [ -z "$jid" ] && { echo "no running job"; return 1; }
  tmux new-session -d -s mon "watch -n 2 'srun --jobid=$jid --overlap nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv'"
  tmux split-window -v -t mon "tail -f \$(ls -t ~/job_results/out/job_*.out | head -1)"
  tmux attach -t mon
}

# Raccourci entraînement (chemins de pré-mélange + checkpoints)
train_vlm() {
  subjob run_job.sbatch train.py \
    --shuffled_path /tmpdir/$USER/cauldron_shuffled \
    --checkpoint_dir /tmpdir/$USER/checkpoints \
    --resume_from /tmpdir/$USER/checkpoints "$@"
}
```

## Annexe — accès ssh à Turpan

Turpan se rejoint via un rebond (machine de l'école). Sur **votre machine locale**,
`~/.ssh/config` (remplacez les valeurs entre `<...>`) :
```
Host turpan
    HostName <hote_turpan>
    User <compte_calmip>
    ProxyJump <hote_rebond>
    IdentityFile ~/.ssh/<votre_cle>
    IdentitiesOnly yes
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

Piège CALMIP : Turpan n'utilise **pas** `~/.ssh/authorized_keys`, mais un dossier
géré `~/.ssh_calmip/`. La clé publique doit être ajoutée dans le seul fichier
éditable, `authorized_keys.user` (vous en êtes propriétaire, mode 400) — sinon la clé
est refusée et on retombe sur une demande de mot de passe en boucle :
```bash
# sur Turpan :
chmod u+w ~/.ssh_calmip/authorized_keys.user
echo 'ssh-ed25519 AAAA...votre_cle_publique... vous@machine' >> ~/.ssh_calmip/authorized_keys.user
chmod 400 ~/.ssh_calmip/authorized_keys.user
```
Ne pas toucher `authorized_keys.admin` / `.pi` / `.internal` (gérés par CALMIP). Une
fois la clé en place, elle est permanente côté Turpan.

