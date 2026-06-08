"""
Evaluate CLIP image features on CIFAR-100 with linear probing.

Linear probing setup:
- Extract frozen CLIP image embeddings
- Fit a multinomial Logistic Regression classifier on train embeddings
- Evaluate top-1 accuracy on the CIFAR-100 test set

By default, only 10,000 train images are used (instead of full 50,000)
to keep runtime reasonable.

Usage:
  uv run python eval_cifar100_linear_probe.py
  uv run python eval_cifar100_linear_probe.py --model_name="ViT-L/14" --device="cuda"
  uv run python eval_cifar100_linear_probe.py --train_subset=5000 --batch_size=64
"""

import ssl

# School/corporate networks often have a self-signed cert in the chain;
# patch the global SSL context before any download is triggered.
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from clip_api import load

import torchvision.datasets as datasets

# -----------------------------------------------------------------------------
model_name = "ViT-B/32"  # CLIP model variant
device = "cuda"  # 'cpu' or 'cuda'
batch_size = 64  # batch size for feature extraction
data_root = "./data"  # where to download CIFAR-100
seed = 1337
train_subset = 10000  # number of train images used for linear probe (<= 50000)
max_iter = 1000  # sklearn LogisticRegression iterations
# -----------------------------------------------------------------------------
exec(open(f"{os.getcwd()}/configurator.py").read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
np.random.seed(seed)

# =============================================================================
# Load model
# =============================================================================

print(f"Loading CLIP model '{model_name}' on {device}...")
model, preprocess = load(model_name, device=device)
model.eval()

# =============================================================================
# Load CIFAR-100 train/test sets
# =============================================================================

print("Loading CIFAR-100 train/test sets...")
cifar100_train = datasets.CIFAR100(
    root=data_root, train=True, download=True, transform=preprocess
)
cifar100_test = datasets.CIFAR100(
    root=data_root, train=False, download=True, transform=preprocess
)

if train_subset is None:
    train_subset = len(cifar100_train)

if train_subset <= 0:
    raise ValueError(f"train_subset must be > 0, got {train_subset}")

train_subset = min(train_subset, len(cifar100_train))
if train_subset < len(cifar100_train):
    indices = torch.randperm(len(cifar100_train))[:train_subset].tolist()
    cifar100_train = torch.utils.data.Subset(cifar100_train, indices)

print(f"Using {train_subset} / 50000 CIFAR-100 train images for linear probing.")

train_loader = torch.utils.data.DataLoader(
    cifar100_train, batch_size=batch_size, shuffle=False, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    cifar100_test, batch_size=batch_size, shuffle=False, num_workers=0
)


def extract_image_features(data_loader, split_name):
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Extract {split_name} feats"):
            images = images.to(device)
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=1, keepdim=True)  # CLIP-style L2 norm

            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    x = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return x, y


# =============================================================================
# 1. Linear Probing (Logistic Regression)
# =============================================================================

print("\n" + "=" * 60)
print("1. LINEAR PROBING ON CIFAR-100")
print("=" * 60)

x_train, y_train = extract_image_features(train_loader, "train")
x_test, y_test = extract_image_features(test_loader, "test")

print(f"Train feature shape: {x_train.shape}")
print(f"Test  feature shape: {x_test.shape}")
print("Fitting Logistic Regression...")

clf = LogisticRegression(
    max_iter=max_iter,
    solver="lbfgs",
    multi_class="multinomial",
    random_state=seed,
)
clf.fit(x_train, y_train)

preds = clf.predict(x_test)
acc = accuracy_score(y_test, preds) * 100

print(f"\nLinear probe on CIFAR-100 test set ({len(y_test)} images):")
print(f"  Top-1 accuracy: {acc:.2f}%")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Linear probe top-1 : {acc:.2f}%")
print()