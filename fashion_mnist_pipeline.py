#!/usr/bin/env python3
"""
TASK 4: Image Classification with Classical / Pre-trained ML
=============================================================

Fashion-MNIST image classification pipeline (100% local, free open-source tools).

Pipeline steps
--------------
1.  Load the Fashion-MNIST benchmark dataset (60,000 train / 10,000 test, 28x28 grayscale).
2.  Normalize the image arrays: pixel scaling  [0,255] -> [0,1]  -> zero-mean / unit-std.
3.  Stratified train / validation partitioning (sklearn) on top of the official test split.
4.  Train three classifiers:
        * MLP  - Multi-Layer Perceptron (PyTorch)
        * CNN  - lightweight 2-block Convolutional Neural Network (PyTorch)
        * SVM  - linear SVM with hinge loss trained by SGD (scikit-learn)
5.  Evaluate every model across all 10 target classes: per-class precision / recall /
    F1, accuracy, and a confusion matrix chart (raw counts + row-normalized).
6.  Inference demo: sample test images shown with predicted label + confidence,
    paired with the ground-truth label (correct = green, wrong = red).

Outputs (written to ./outputs)
------------------------------
    training_curves.png           loss & accuracy progression (MLP + CNN)
    confusion_matrix_<model>.png  10x10 confusion matrix charts (3 models)
    inference_demo.png            sample test images with predicted class + confidence
    model_comparison.png          test-accuracy bar chart of all models
    metrics.json                  hyperparameters, normalization stats, all metrics
    classification_report_*.txt   sklearn per-class classification reports
    training_history.csv          per-epoch train/val loss & accuracy
    prediction_log.csv            inference demo predictions & confidences

Usage
-----
    python fashion_mnist_pipeline.py                 # full training run
    python fashion_mnist_pipeline.py --smoke         # fast end-to-end sanity run
    python fashion_mnist_pipeline.py --epochs-cnn 12 # tune epochs
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: figures are saved to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"


@dataclass
class Config:
    """All hyper-parameters of the run (exported to metrics.json / README)."""

    seed: int = 42
    batch_size: int = 128
    epochs_mlp: int = 8
    epochs_cnn: int = 8
    lr: float = 1e-3                 # Adam learning rate (both deep models)
    weight_decay: float = 0.0
    dropout: float = 0.25            # MLP + CNN head dropout
    val_fraction: float = 0.1        # 10% of the training set held out for validation
    num_workers: int = 0             # Windows-friendly: no multiprocessing loaders
    # classical linear-SVM baseline (scikit-learn)
    svm_alpha: float = 1e-4          # 1/ (C * n_samples) style regularisation
    svm_max_iter: int = 30
    n_inference_samples: int = 16    # images shown in the inference demo
    smoke: bool = False              # tiny run used to validate the pipeline
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Reproducibility & data loading
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(cfg: Config):
    """Download (first run only) and load Fashion-MNIST as raw uint8 arrays."""
    print("=" * 72)
    print("STEP 1 - Loading Fashion-MNIST dataset")
    print("=" * 72)
    t0 = time.time()
    train_ds = FashionMNIST(root=str(DATA_DIR), train=True, download=True)
    test_ds = FashionMNIST(root=str(DATA_DIR), train=False, download=True)

    # .numpy() on the underlying torch tensors (no copies of the label data)
    x_train_raw = train_ds.data.numpy()          # uint8 (60000, 28, 28)
    y_train = train_ds.targets.numpy()           # int64 (60000,)
    x_test_raw = test_ds.data.numpy()            # uint8 (10000, 28, 28)
    y_test = test_ds.targets.numpy()             # int64 (10000,)

    if cfg.smoke:  # tiny subset for a fast end-to-end sanity check
        x_train_raw, y_train = x_train_raw[:5000], y_train[:5000]
        x_test_raw, y_test = x_test_raw[:1000], y_test[:1000]

    print(f"  train images : {x_train_raw.shape}  dtype={x_train_raw.dtype}")
    print(f"  test images  : {x_test_raw.shape}  dtype={x_test_raw.dtype}")
    print(f"  raw pixel range: [{x_train_raw.min()}, {x_train_raw.max()}] (uint8)")
    print(f"  classes      : {len(CLASS_NAMES)} -> {CLASS_NAMES}")
    print(f"  loaded in {time.time() - t0:.1f}s  (dataset cached in {DATA_DIR})\n")
    return x_train_raw, y_train, x_test_raw, y_test


def normalize_pixels(x_raw: np.ndarray):
    """
    STEP 2 - Normalization / pixel scaling.

        stage 1: uint8 [0,255]  -> float32 [0,1]     (divide by 255)
        stage 2: [0,1]          -> zero-mean unit-std ((x - mean) / std)

    mean/std are computed on the TRAINING partition only (no test-set leakage)
    and are reused for the validation and test partitions.
    """
    x01 = x_raw.astype(np.float32) / 255.0
    return x01


def compute_stats(x01: np.ndarray) -> tuple[float, float]:
    mean = float(x01.mean())
    std = float(x01.std())
    return mean, std


def standardize(x01: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((x01 - mean) / std).astype(np.float32)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class MLP(nn.Module):
    """Multi-Layer Perceptron: 784 -> 512 -> 256 -> 10."""

    def __init__(self, n_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    """Lightweight CNN: 2 conv blocks (32 -> 64) + dense head 128 -> 10."""

    def __init__(self, n_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 14x14 -> 7x7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Training / evaluation loops
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loader(model, loader, device, criterion=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if criterion is not None:
            total_loss += criterion(logits, yb).item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)
    loss = total_loss / total if criterion is not None else float("nan")
    return loss, correct / total


def train_model(name, model, train_loader, val_loader, epochs, device, cfg, history):
    """Full-batch-loop training with per-epoch train/val loss & accuracy logging."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    print(f"  {name}: {count_parameters(model):,} trainable parameters, "
          f"{epochs} epoch(s), lr={cfg.lr}")
    print(f"  {'epoch':>5} {'train_loss':>11} {'train_acc':>10} "
          f"{'val_loss':>10} {'val_acc':>9} {'time':>7}")
    print("  " + "-" * 58)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

        train_loss, train_acc = run_loss / total, correct / total
        val_loss, val_acc = evaluate_loader(model, val_loader, device, criterion)
        elapsed = time.time() - t0

        history.append(dict(model=name, epoch=epoch, train_loss=train_loss,
                            train_acc=train_acc, val_loss=val_loss,
                            val_acc=val_acc, seconds=round(elapsed, 1)))
        print(f"  {epoch:>5} {train_loss:>11.4f} {train_acc:>10.4f} "
              f"{val_loss:>10.4f} {val_acc:>9.4f} {elapsed:>6.1f}s")
    print()
    return history


@torch.no_grad()
def predict_torch(model, loader, device):
    """Return (true labels, predicted labels, softmax probabilities)."""
    model.eval()
    ys, preds, probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1)
        ys.append(yb.numpy())
        preds.append(p.argmax(dim=1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------

def plot_training_curves(history, path: Path):
    """Loss and accuracy progression for every deep model trained."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"MLP": "#1f77b4", "CNN": "#d62728"}
    for name in sorted({h["model"] for h in history}):
        rows = [h for h in history if h["model"] == name]
        ep = [r["epoch"] for r in rows]
        c = colors.get(name, "#2ca02c")
        axes[0].plot(ep, [r["train_loss"] for r in rows], "o-", color=c,
                     label=f"{name} train")
        axes[0].plot(ep, [r["val_loss"] for r in rows], "s--", color=c,
                     alpha=0.7, label=f"{name} val")
        axes[1].plot(ep, [r["train_acc"] for r in rows], "o-", color=c,
                     label=f"{name} train")
        axes[1].plot(ep, [r["val_acc"] for r in rows], "s--", color=c,
                     alpha=0.7, label=f"{name} val")

    axes[0].set_title("Training curves - Loss (cross-entropy)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Training curves - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.75, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Fashion-MNIST: loss & accuracy progression", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_confusion(y_true, y_pred, path: Path, title: str):
    """10x10 confusion matrix chart: raw counts + row-normalized percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0],
                linewidths=0.5, annot_kws={"size": 8})
    axes[0].set_title(f"{title}\nconfusion matrix (counts)")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1],
                vmin=0.0, vmax=1.0, linewidths=0.5, annot_kws={"size": 8})
    axes[1].set_title(f"{title}\nrow-normalized (recall per true class)")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_model_comparison(results: dict, path: Path):
    names = list(results.keys())
    accs = [results[n]["test_accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=["#1f77b4", "#d62728", "#2ca02c"][:len(names)])
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=11)
    ax.set_ylim(min(accs) - 0.05, 1.0)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Fashion-MNIST - test accuracy by model")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_inference_demo(x_test01, y_true, y_pred, conf, path: Path,
                        n: int = 16, seed: int = 42):
    """
    Inference demo: sample test images displayed next to their predicted class
    label and confidence, paired with the ground-truth label.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=min(n, len(y_true)), replace=False)
    idx = np.sort(idx)

    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes, idx):
        ok = y_pred[i] == y_true[i]
        ax.imshow(x_test01[i], cmap="gray")
        ax.set_title(
            f"True : {CLASS_NAMES[y_true[i]]}\n"
            f"Pred : {CLASS_NAMES[y_pred[i]]} ({conf[i] * 100:.1f}%)",
            fontsize=9,
            color="#1a7f37" if ok else "#c1121f",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(idx):]:
        ax.axis("off")

    fig.suptitle("Inference demo - test images: prediction + confidence vs ground truth",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")
    return idx


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Fashion-MNIST image classification pipeline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs-mlp", type=int, default=8)
    p.add_argument("--epochs-cnn", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--no-svm", action="store_true", help="skip the classical SVM baseline")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run (5k train / 1k test, 1 epoch)")
    a = p.parse_args()

    cfg = Config(seed=a.seed, batch_size=a.batch_size, epochs_mlp=a.epochs_mlp,
                 epochs_cnn=a.epochs_cnn, lr=a.lr, dropout=a.dropout,
                 val_fraction=a.val_fraction, smoke=a.smoke)
    if a.smoke:
        cfg.epochs_mlp = cfg.epochs_cnn = 1
    cfg.extra["skip_svm"] = a.no_svm
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    # smoke runs write to separate folders so they can never clobber real results
    out_dir = OUT_DIR if not cfg.smoke else BASE_DIR / "outputs_smoke"
    model_dir = MODEL_DIR if not cfg.smoke else BASE_DIR / "models_smoke"
    out_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))

    print(f"device: {device} | torch {torch.__version__} | seed {cfg.seed}")
    if cfg.smoke:
        print("*** SMOKE MODE: small subset, 1 epoch (pipeline validation only) ***\n")

    # ---------------- STEP 1: dataset -------------------------------------
    x_train_raw, y_train, x_test_raw, y_test = load_dataset(cfg)

    # ---------------- STEP 2: normalization / pixel scaling ---------------
    print("=" * 72)
    print("STEP 2 - Normalization (pixel scaling)")
    print("=" * 72)
    x_train01 = normalize_pixels(x_train_raw)   # [0,255] -> [0,1]
    x_test01 = normalize_pixels(x_test_raw)
    mean, std = compute_stats(x_train01)        # stats from TRAIN split only
    x_train_n = standardize(x_train01, mean, std)
    x_test_n = standardize(x_test01, mean, std)
    print(f"  stage 1 - pixel scaling   : uint8 [0,255] -> float32 [0,1]")
    print(f"  stage 2 - standardization : (x - {mean:.4f}) / {std:.4f}")
    print(f"  check train after norm    : mean={x_train_n.mean():.6f} "
          f"std={x_train_n.std():.6f}  (should be ~0.0 / ~1.0)")
    print(f"  check test  after norm    : mean={x_test_n.mean():.6f} "
          f"std={x_test_n.std():.6f}\n")

    # ---------------- STEP 3: train / validation partitioning --------------
    print("=" * 72)
    print("STEP 3 - Train / validation / test partitioning")
    print("=" * 72)
    idx_all = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        idx_all, test_size=cfg.val_fraction, random_state=cfg.seed,
        stratify=y_train, shuffle=True,
    )
    print(f"  stratified {int(cfg.val_fraction * 100)}% hold-out from the training set")
    print(f"  train: {len(train_idx):>6} images")
    print(f"  val  : {len(val_idx):>6} images")
    print(f"  test : {len(y_test):>6} images (official Fashion-MNIST test split)\n")

    # flattened views (classical ML) and channel-first views (deep learning)
    x_train_flat = x_train_n.reshape(len(x_train_n), -1)
    x_test_flat = x_test_n.reshape(len(x_test_n), -1)
    x_train_img = x_train_n[:, None, :, :]
    x_test_img = x_test_n[:, None, :, :]

    def make_loader(x, y, shuffle):
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, drop_last=False)

    train_loader = make_loader(x_train_img[train_idx], y_train[train_idx], True)
    val_loader = make_loader(x_train_img[val_idx], y_train[val_idx], False)
    test_loader = make_loader(x_test_img, y_test, False)

    results: dict[str, dict] = {}
    histories: dict[str, list] = {}

    # ---------------- STEP 4: train the deep models ------------------------
    print("=" * 72)
    print("STEP 4 - Training classifiers")
    print("=" * 72)
    for name, model_cls, epochs in (("MLP", MLP, cfg.epochs_mlp),
                                    ("CNN", CNN, cfg.epochs_cnn)):
        print(f"[{name}]")
        model = model_cls(dropout=cfg.dropout).to(device)
        hist = []
        t0 = time.time()
        train_model(name, model, train_loader, val_loader, epochs, device, cfg, hist)
        histories[name] = hist
        torch.save({"model_state": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "norm_mean": mean, "norm_std": std,
                    "model": name}, model_dir / f"{name.lower()}_fashion_mnist.pt")
        y_true, y_pred, y_prob = predict_torch(model, test_loader, device)
        acc = accuracy_score(y_true, y_pred)
        results[name] = dict(model=model, test_accuracy=float(acc),
                             y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                             seconds=round(time.time() - t0, 1))
        print(f"  -> {name} test accuracy: {acc:.4f}  "
              f"({results[name]['seconds']}s total)\n")

    # ---------------- STEP 4b: classical linear-SVM baseline ---------------
    if not cfg.extra.get("skip_svm"):
        print("[SVM] linear SVM (hinge loss) trained with SGD - scikit-learn")
        t0 = time.time()
        svm = SGDClassifier(loss="hinge", alpha=cfg.svm_alpha,
                            max_iter=cfg.svm_max_iter, tol=1e-3,
                            random_state=cfg.seed)
        svm.fit(x_train_flat[train_idx], y_train[train_idx])
        # Platt (sigmoid) scaling on the held-out validation split so that the
        # SVM reports calibrated class probabilities instead of raw margins.
        svm_cal = CalibratedClassifierCV(FrozenEstimator(svm), method="sigmoid")
        svm_cal.fit(x_train_flat[val_idx], y_train[val_idx])
        svm_prob = svm_cal.predict_proba(x_test_flat)
        svm_pred = svm_prob.argmax(axis=1)
        acc = accuracy_score(y_test, svm_pred)
        results["SVM"] = dict(model=svm_cal, test_accuracy=float(acc),
                              y_true=y_test, y_pred=svm_pred, y_prob=svm_prob,
                              seconds=round(time.time() - t0, 1))
        print(f"  -> SVM test accuracy: {acc:.4f}  ({results['SVM']['seconds']}s)\n")

    # ---------------- STEP 5: evaluation ----------------------------------
    print("=" * 72)
    print("STEP 5 - Evaluation across all 10 target classes")
    print("=" * 72)

    history_rows = [r for rows in histories.values() for r in rows]
    with open(out_dir / "training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "epoch", "train_loss",
                                          "train_acc", "val_loss", "val_acc",
                                          "seconds"])
        w.writeheader()
        w.writerows(history_rows)

    if history_rows:
        plot_training_curves(history_rows, out_dir / "training_curves.png")

    for name, res in results.items():
        report = classification_report(res["y_true"], res["y_pred"],
                                       target_names=CLASS_NAMES, digits=4,
                                       zero_division=0)
        (out_dir / f"classification_report_{name.lower()}.txt").write_text(
            f"{name} - Fashion-MNIST classification report\n"
            f"test accuracy: {res['test_accuracy']:.4f}\n\n{report}")
        print(f"\n[{name}] test accuracy = {res['test_accuracy']:.4f}")
        print(report)
        plot_confusion(res["y_true"], res["y_pred"],
                       out_dir / f"confusion_matrix_{name.lower()}.png",
                       f"{name} (acc={res['test_accuracy']:.4f})")

    plot_model_comparison({k: v for k, v in results.items()},
                          out_dir / "model_comparison.png")

    # ---------------- STEP 6: inference demo ------------------------------
    print("=" * 72)
    print("STEP 6 - Inference demo (sample test images + confidence)")
    print("=" * 72)
    best_name = max(results, key=lambda k: results[k]["test_accuracy"])
    best = results[best_name]
    print(f"  best model: {best_name} "
          f"(test accuracy {best['test_accuracy']:.4f})\n")

    demo_idx = plot_inference_demo(
        x_test01, best["y_true"], best["y_pred"],
        best["y_prob"].max(axis=1),
        out_dir / "inference_demo.png",
        n=cfg.n_inference_samples, seed=cfg.seed)

    header = (f"{'idx':>5} {'true':>12} {'predicted':>12} {'confidence':>10} "
              f"{'result':>7}")
    print("  " + header)
    print("  " + "-" * len(header))
    log_rows = []
    for i in demo_idx:
        ok = best["y_pred"][i] == best["y_true"][i]
        row = dict(
            index=int(i),
            true_label=CLASS_NAMES[int(best["y_true"][i])],
            predicted_label=CLASS_NAMES[int(best["y_pred"][i])],
            confidence=round(float(best["y_prob"][i].max()), 4),
            correct=bool(ok),
        )
        log_rows.append(row)
        print(f"  {i:>5} {row['true_label']:>12} {row['predicted_label']:>12} "
              f"{row['confidence']:>10.4f} {'OK' if ok else 'MISS':>7}")

    with open(out_dir / "prediction_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "true_label", "predicted_label",
                                          "confidence", "correct"])
        w.writeheader()
        w.writerows(log_rows)
    n_ok = sum(r["correct"] for r in log_rows)
    print(f"\n  demo accuracy on {len(log_rows)} samples: "
          f"{n_ok}/{len(log_rows)} ({n_ok / len(log_rows):.2%})")
    print(f"  saved prediction_log.csv, inference_demo.png")

    # ---------------- STEP 7: metrics export ------------------------------
    per_class = {}
    for name, res in results.items():
        rep = classification_report(res["y_true"], res["y_pred"],
                                    target_names=CLASS_NAMES, output_dict=True,
                                    zero_division=0)
        per_class[name] = {
            c: {k: round(rep[c][k], 4) for k in ("precision", "recall", "f1-score")}
            for c in CLASS_NAMES
        }

    metrics = {
        "dataset": "Fashion-MNIST (Zalando), 28x28 grayscale, 10 classes",
        "splits": {"train": int(len(train_idx)), "val": int(len(val_idx)),
                   "test": int(len(y_test))},
        "normalization": {
            "stage1": "pixel scaling uint8 [0,255] -> float32 [0,1]",
            "stage2": f"standardization (x - {mean:.6f}) / {std:.6f}",
            "train_mean_after": round(float(x_train_n.mean()), 6),
            "train_std_after": round(float(x_train_n.std()), 6),
        },
        "hyperparameters": {k: v for k, v in asdict(cfg).items() if k != "extra"},
        "test_accuracy": {k: round(v["test_accuracy"], 4)
                          for k, v in results.items()},
        "train_seconds": {k: v["seconds"] for k, v in results.items()},
        "best_model": best_name,
        "per_class_metrics": per_class,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  saved metrics.json")

    # ---------------- summary ---------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, res in results.items():
        print(f"  {name:<5} test accuracy = {res['test_accuracy']:.4f}")
    print(f"  best model : {best_name}")
    print(f"  artifacts  : {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
