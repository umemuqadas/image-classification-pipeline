#!/usr/bin/env python3
"""Builds inference_demo.ipynb (a runnable companion notebook for Task 4).

The notebook itself does NOT retrain anything: it loads the checkpoint exported by
fashion_mnist_pipeline.py, re-applies the exact stored normalization, runs inference
on Fashion-MNIST test images and shows predicted label + confidence next to the
ground truth.
"""

from pathlib import Path

import nbformat as nbf

BASE = Path(__file__).resolve().parent

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                             "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

cells = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


md("""
# TASK 4 — Inference Demo Notebook

**Fashion-MNIST image classification with classical / pre-trained ML**

This notebook is the *inference* companion of `fashion_mnist_pipeline.py`:
it loads the checkpoint written by that script, re-applies the exact same
normalization, and classifies test images — displaying each prediction next to
its **ground-truth label** and the model's **confidence**.

1. Load dataset + stored normalization constants
2. Rebuild the CNN architecture and load the trained weights
3. Classify a batch of test images
4. Plot: image → `True` vs `Pred (confidence %)`

> Run `python fashion_mnist_pipeline.py` first (or use the pre-exported
> `models/cnn_fashion_mnist.pt`).
""")

code("""
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST

BASE = Path.cwd()
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
CKPT = BASE / "models" / "cnn_fashion_mnist.pt"
N_SAMPLES = 16
SEED = 42
print("checkpoint:", CKPT, "exists:", CKPT.exists())
""")

md("""
## 1. Dataset loading & normalization

Raw pixels are `uint8 [0,255]`. As in the training script:

* **stage 1** — pixel scaling: `x / 255 → [0,1]`
* **stage 2** — standardization: `(x − mean) / std`

The `mean`/`std` come from the checkpoint, so inference uses *exactly* the
statistics the model was trained with (no train/test mismatch).
""")

code("""
train_ds = FashionMNIST(root=str(BASE / "data"), train=True, download=True)
test_ds  = FashionMNIST(root=str(BASE / "data"), train=False, download=True)

x_test = test_ds.data.numpy().astype(np.float32) / 255.0   # stage 1: pixel scaling
y_test = test_ds.targets.numpy()

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
mean, std = ckpt["norm_mean"], ckpt["norm_std"]            # stage 2 constants (from training)
x_test = (x_test - mean) / std

print(f"raw pixels  : uint8 [{test_ds.data.min()}, {test_ds.data.max()}]")
print(f"after norm  : mean={x_test.mean():.6f}  std={x_test.std():.6f}")
print(f"train mean/std used: {mean:.6f} / {std:.6f}")
""")

md("""
## 2. Rebuild the model and load the trained weights

The architecture is the lightweight CNN from the pipeline
(2 conv blocks → dense head). We rebuild it from scratch and load the exported state dict.
""")

code("""
class CNN(nn.Module):
    def __init__(self, n_classes=10, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


model = CNN()
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"loaded '{ckpt['model']}' checkpoint, "
      f"{sum(p.numel() for p in model.parameters()):,} parameters")
""")

md("""
## 3. Classify sample test images → prediction + confidence

Softmax over the logits gives a calibrated-ish confidence per class; we keep the
top prediction and its probability for every image.
""")

code("""
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(y_test), size=N_SAMPLES, replace=False))

x = torch.from_numpy(x_test[idx][:, None, :, :])
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1).numpy()
pred = probs.argmax(axis=1)
conf = probs.max(axis=1)

print(f"{'idx':>5} {'true':>12} {'predicted':>12} {'confidence':>10}  result")
print("-" * 56)
for i, k in enumerate(idx):
    ok = pred[i] == y_test[k]
    print(f"{k:>5} {CLASS_NAMES[y_test[k]]:>12} {CLASS_NAMES[pred[i]]:>12} "
          f"{conf[i]:>10.4f}  {'OK' if ok else 'MISS'}")
print(f"\\nbatch accuracy: {(pred == y_test[idx]).mean():.2%}")
""")

md("""
## 4. Visual demo — sample test images with predicted class & confidence

Green = correct prediction, red = wrong prediction. The **ground truth** is always
shown on the first line of each title, the **prediction + confidence** on the second.
""")

code("""
cols, rows = 4, N_SAMPLES // 4
fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
for ax, i, k in zip(np.ravel(axes), range(N_SAMPLES), idx):
    ok = pred[i] == y_test[k]
    ax.imshow(x_test[k], cmap="gray")
    ax.set_title(f"True : {CLASS_NAMES[y_test[k]]}\\n"
                 f"Pred : {CLASS_NAMES[pred[i]]} ({conf[i] * 100:.1f}%)",
                 fontsize=9, color="#1a7f37" if ok else "#c1121f")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Inference demo — prediction + confidence vs ground truth", fontsize=13)
fig.tight_layout()
plt.show()
""")

md("""
### Notes

* Training, curves and confusion matrices are produced by `python fashion_mnist_pipeline.py`
  (see `outputs/training_curves.png`, `outputs/confusion_matrix_*.png`).
* The checkpoint stores `norm_mean` / `norm_std`, so this notebook never re-derives
  statistics from the test set.
* Everything runs locally on CPU with free open-source packages — no cloud services.
""")

nb.cells = cells
path = BASE / "inference_demo.ipynb"
nbf.write(nb, str(path))
print("wrote", path, "cells:", len(cells))
