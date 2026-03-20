# Train a small GPT-2 on the French Philosophy 10K dataset
# Designed to run on CPU (macbook)
#
# Usage:
#   cd "PP5: Pretraining GPT2"
#   python data/french_philosophy/prepare.py
#   python train.py config/train_french_philosophy.py

out_dir = "out-french-philosophy"
eval_interval = 250
eval_iters = 20
log_interval = 10

always_save_checkpoint = False

dataset = "french_philosophy"
gradient_accumulation_steps = 1
batch_size = 8
block_size = 256

# small GPT model suitable for CPU training
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# CPU settings
compile = False
dtype = "float32"
