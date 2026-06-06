# Finetune a pretrained GPT-2 (124M) on the French Philosophy 10K dataset
# Designed to run on CPU (macbook) - use 'mps' or 'cuda' if available
#
# Usage:
#   cd "PP5: Pretraining GPT2"
#   python data/french_philosophy/prepare.py
#   python train.py config/finetune_french_philosophy.py

out_dir = "out-french-philosophy-ft"
eval_interval = 50
eval_iters = 20
log_interval = 10

always_save_checkpoint = False

dataset = "french_philosophy"
init_from = "gpt2"  # start from pretrained GPT-2 124M weights

# the number of examples per iter:
# 8 batch_size * 1 grad_accum * 1024 tokens = 8,192 tokens/iter
batch_size = 8
gradient_accumulation_steps = 1
block_size = 1024  # GPT-2 native context length

# finetuning hyperparameters
learning_rate = 3e-5  # lower LR for finetuning
max_iters = 500
dropout = 0.1

# short warmup then constant LR (good for finetuning)
decay_lr = False
warmup_iters = 20

# CPU settings (change to 'cuda' or 'mps' if available)
compile = False
dtype = "float32"
