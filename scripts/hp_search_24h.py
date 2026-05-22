#!/usr/bin/env python3
"""
24-hour Hyperparameter Search for Gate Model (anti-overfitting focus)
=====================================================================
Runs a grid of experiments with different regularization, data splits,
and cross-version evaluation to find a gate model that genuinely generalises.

Outputs:
  reports/hp_search/  — per-experiment training curves + metrics
  reports/hp_search/summary.csv — all experiments in one table
  reports/hp_search/comparison_*.png — comparison charts

Usage:
  python scripts/hp_search_24h.py                    # full 24h search
  python scripts/hp_search_24h.py --max-hours 2      # quick 2h test
  python scripts/hp_search_24h.py --device mps        # Apple Silicon
  python scripts/hp_search_24h.py --experiments 1,2,5  # run specific ones
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import random
import time
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "splits"
SPLITS_EVAL_DIR = PROJECT_ROOT / "splits_eval"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "hp_search"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GateDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {"normal": 0, "anomaly": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.label_map.get(row["label"], 0)
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def labels(self):
        return self.df["label"].map(self.label_map).values

    def class_counts(self):
        labels = self.labels
        return int((labels == 0).sum()), int((labels == 1).sum())


# ---------------------------------------------------------------------------
# Transforms (with configurable augmentation strength)
# ---------------------------------------------------------------------------
def get_train_transforms(input_size=224, aug_strength="normal"):
    base = [T.Resize((input_size, input_size))]

    if aug_strength == "light":
        base += [
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    elif aug_strength == "normal":
        base += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    elif aug_strength == "heavy":
        base += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    return T.Compose(base)


def get_eval_transforms(input_size=224):
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(gate_name: str, dropout: float = 0.3, pretrained: bool = True):
    if gate_name == "mnv3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 1),
        )
    elif gate_name == "effnetb0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 1),
        )
    elif gate_name == "mnv3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, 1),
        )
    else:
        raise ValueError(f"Unknown gate: {gate_name}")
    return model


def freeze_backbone(model, gate_name: str):
    if gate_name == "mnv3_small":
        for param in model.features.parameters():
            param.requires_grad = False
    elif gate_name == "effnetb0":
        for param in model.features.parameters():
            param.requires_grad = False
    elif gate_name == "mnv3_large":
        for param in model.features.parameters():
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Label smoothing BCE
# ---------------------------------------------------------------------------
class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets_smooth)


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.float().to(device)

        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(imgs.size(0)).to(device)
            imgs = lam * imgs + (1 - lam) * imgs[idx]
            labels = lam * labels + (1 - lam) * labels[idx]

        optimizer.zero_grad()
        logits = model(imgs).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        if mixup_alpha > 0:
            correct += (preds == labels.round().long()).sum().item()
        else:
            correct += (preds == labels.long()).sum().item()
        total += imgs.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels_f = labels.float().to(device)
        logits = model(imgs).squeeze(-1)
        loss = criterion(logits, labels_f)
        running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        correct += (preds == labels.to(device).long()).sum().item()
        total += imgs.size(0)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    return (
        running_loss / max(total, 1),
        correct / max(total, 1),
        np.array(all_probs),
        np.array(all_labels),
    )


def compute_metrics(y_true, probs):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {}
    y_pred = (probs >= 0.5).astype(int)
    return {
        "auroc": roc_auc_score(y_true, probs),
        "auprc": average_precision_score(y_true, probs),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "accuracy": (y_pred == y_true).mean(),
    }


# ---------------------------------------------------------------------------
# Cross-version evaluation
# ---------------------------------------------------------------------------
def check_csv_paths_exist(csv_path: Path, sample_n: int = 5) -> bool:
    """Check if image paths in a CSV actually exist on disk."""
    try:
        df = pd.read_csv(csv_path, nrows=sample_n)
        for _, row in df.iterrows():
            if not os.path.exists(row["path"]):
                return False
        return True
    except Exception:
        return False


def cross_version_eval(model, device, input_size, versions=("v1", "v2", "v3")):
    """Evaluate trained model on ALL version test sets + balanced test sets."""
    results = {}
    eval_tf = get_eval_transforms(input_size)
    criterion = nn.BCEWithLogitsLoss()

    for v in versions:
        for split_name, split_dir in [
            ("test_gate", SPLITS_DIR),
            ("balanced_test", SPLITS_EVAL_DIR),
        ]:
            csv_path = split_dir / f"{v}_{split_name}.csv"
            if not csv_path.exists():
                continue
            if not check_csv_paths_exist(csv_path):
                print(f"    [skip] {v}_{split_name}: image paths not found locally")
                continue
            ds = GateDataset(csv_path, transform=eval_tf)
            dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
            _, acc, probs, labels = evaluate(model, dl, criterion, device)
            metrics = compute_metrics(labels, probs)
            metrics["accuracy"] = acc
            key = f"{v}_{split_name}"
            results[key] = metrics

    # Also eval on round-based test sets (may use Docker paths)
    for rnd in ("round1", "round2", "round3"):
        csv_path = SPLITS_DIR / f"{rnd}_test_gate.csv"
        if not csv_path.exists():
            continue
        if not check_csv_paths_exist(csv_path):
            print(f"    [skip] {rnd}_test_gate: image paths not found locally (Docker paths?)")
            continue
        ds = GateDataset(csv_path, transform=eval_tf)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        _, acc, probs, labels = evaluate(model, dl, criterion, device)
        metrics = compute_metrics(labels, probs)
        metrics["accuracy"] = acc
        results[f"{rnd}_test_gate"] = metrics

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_experiment_curves(history: dict, exp_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Experiment: {exp_name}", fontsize=14, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "b-", label="train", linewidth=1.5)
    ax.plot(epochs, history["val_loss"], "r-", label="val", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], "b-", label="train", linewidth=1.5)
    ax.plot(epochs, history["val_acc"], "r-", label="val", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.02])

    # Train-Val gap (overfitting indicator)
    ax = axes[1, 0]
    gap_loss = [t - v for t, v in zip(history["train_loss"], history["val_loss"])]
    gap_acc = [t - v for t, v in zip(history["train_acc"], history["val_acc"])]
    ax.plot(epochs, gap_loss, "g-", label="loss gap (train-val)", linewidth=1.5)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gap"); ax.set_title("Overfitting Gap (Loss)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # LR schedule
    ax = axes[1, 1]
    if "lr" in history:
        ax.plot(epochs, history["lr"], "m-", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("Learning Rate")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{exp_name}_curves.png", dpi=150)
    plt.close(fig)


def plot_cross_eval_heatmap(cross_results: dict, exp_name: str, out_dir: Path):
    """Heatmap of F1 scores across evaluation sets."""
    if not cross_results:
        return
    eval_sets = sorted(cross_results.keys())
    f1_scores = [cross_results[k].get("f1", 0) for k in eval_sets]

    fig, ax = plt.subplots(figsize=(max(10, len(eval_sets) * 1.2), 3))
    colors = ["#d32f2f" if f < 0.8 else "#ff9800" if f < 0.95 else "#4caf50" for f in f1_scores]
    bars = ax.barh(eval_sets, f1_scores, color=colors)
    ax.set_xlim([0, 1.05])
    ax.set_xlabel("F1 Score")
    ax.set_title(f"{exp_name} — Cross-evaluation F1")
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.axvline(x=0.95, color="gray", linestyle="--", linewidth=0.8, label="0.95 threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{exp_name}_cross_eval.png", dpi=150)
    plt.close(fig)


def plot_summary_comparison(summary_df: pd.DataFrame, out_dir: Path):
    """Big comparison chart across all experiments."""
    if summary_df.empty:
        return

    # F1 comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(summary_df) * 0.4)))

    # Chart 1: Same-split test F1
    ax = axes[0]
    names = summary_df["name"].values
    test_f1 = summary_df["test_f1"].values
    colors = ["#d32f2f" if f < 0.8 else "#ff9800" if f < 0.95 else "#4caf50" for f in test_f1]
    bars = ax.barh(names, test_f1, color=colors)
    ax.set_xlim([0, 1.05])
    ax.set_xlabel("F1 Score")
    ax.set_title("Same-split Test F1")
    for bar, val in zip(bars, test_f1):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")

    # Chart 2: Cross-version mean F1 (generalization)
    ax = axes[1]
    cross_f1 = summary_df["cross_mean_f1"].values
    colors2 = ["#d32f2f" if f < 0.8 else "#ff9800" if f < 0.95 else "#4caf50" for f in cross_f1]
    bars2 = ax.barh(names, cross_f1, color=colors2)
    ax.set_xlim([0, 1.05])
    ax.set_xlabel("F1 Score")
    ax.set_title("Cross-version Mean F1 (Generalisation)")
    for bar, val in zip(bars2, cross_f1):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")

    fig.tight_layout()
    fig.savefig(out_dir / "comparison_f1.png", dpi=150)
    plt.close(fig)

    # Overfitting gap chart
    fig, ax = plt.subplots(figsize=(12, max(5, len(summary_df) * 0.35)))
    gap = (summary_df["test_f1"] - summary_df["cross_mean_f1"]).values
    colors3 = ["#d32f2f" if g > 0.1 else "#ff9800" if g > 0.03 else "#4caf50" for g in gap]
    bars3 = ax.barh(names, gap, color=colors3)
    ax.set_xlabel("F1 Gap (same-split - cross-version)")
    ax.set_title("Generalisation Gap (lower = better)")
    for bar, val in zip(bars3, gap):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=8)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_gap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
def get_experiments() -> list[dict]:
    """Each dict defines one experiment config."""
    experiments = [
        # --- Baseline reproduction (current config) ---
        {
            "name": "baseline_v1_mnv3s",
            "gate": "mnv3_small",
            "train_tag": "v1",
            "pretrained": True,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 30,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "normal",
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "freeze_epochs": 3,
            "description": "Current config (baseline reproduction)",
        },
        # --- Heavy regularisation ---
        {
            "name": "heavy_reg_v1",
            "gate": "mnv3_small",
            "train_tag": "v1",
            "pretrained": True,
            "dropout": 0.6,
            "lr": 5e-4,
            "weight_decay": 1e-2,
            "epochs": 50,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.3,
            "freeze_epochs": 5,
            "description": "Dropout=0.6, WD=1e-2, label smooth=0.1, mixup=0.3, heavy aug",
        },
        {
            "name": "heavy_reg_v2",
            "gate": "mnv3_small",
            "train_tag": "v2",
            "pretrained": True,
            "dropout": 0.6,
            "lr": 5e-4,
            "weight_decay": 1e-2,
            "epochs": 50,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.3,
            "freeze_epochs": 5,
            "description": "Same heavy reg on v2",
        },
        {
            "name": "heavy_reg_v3",
            "gate": "mnv3_small",
            "train_tag": "v3",
            "pretrained": True,
            "dropout": 0.6,
            "lr": 5e-4,
            "weight_decay": 1e-2,
            "epochs": 50,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.3,
            "freeze_epochs": 5,
            "description": "Same heavy reg on v3",
        },
        # --- From scratch (no pretrained) ---
        {
            "name": "scratch_v1",
            "gate": "mnv3_small",
            "train_tag": "v1",
            "pretrained": False,
            "dropout": 0.5,
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "epochs": 60,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.05,
            "mixup_alpha": 0.2,
            "freeze_epochs": 0,
            "description": "No pretrained weights — tests if ImageNet features cause overfit",
        },
        {
            "name": "scratch_v3",
            "gate": "mnv3_small",
            "train_tag": "v3",
            "pretrained": False,
            "dropout": 0.5,
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "epochs": 60,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.05,
            "mixup_alpha": 0.2,
            "freeze_epochs": 0,
            "description": "From scratch on v3 (most augmented)",
        },
        # --- Medium regularisation (sweet spot search) ---
        {
            "name": "med_reg_v1",
            "gate": "mnv3_small",
            "train_tag": "v1",
            "pretrained": True,
            "dropout": 0.5,
            "lr": 5e-4,
            "weight_decay": 1e-3,
            "epochs": 40,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.05,
            "mixup_alpha": 0.1,
            "freeze_epochs": 5,
            "description": "Medium reg: dropout=0.5, WD=1e-3, LS=0.05, mixup=0.1",
        },
        {
            "name": "med_reg_v2",
            "gate": "mnv3_small",
            "train_tag": "v2",
            "pretrained": True,
            "dropout": 0.5,
            "lr": 5e-4,
            "weight_decay": 1e-3,
            "epochs": 40,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.05,
            "mixup_alpha": 0.1,
            "freeze_epochs": 5,
            "description": "Medium reg on v2",
        },
        {
            "name": "med_reg_v3",
            "gate": "mnv3_small",
            "train_tag": "v3",
            "pretrained": True,
            "dropout": 0.5,
            "lr": 5e-4,
            "weight_decay": 1e-3,
            "epochs": 40,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.05,
            "mixup_alpha": 0.1,
            "freeze_epochs": 5,
            "description": "Medium reg on v3",
        },
        # --- Small batch + medium reg ---
        {
            "name": "small_batch_v1",
            "gate": "mnv3_small",
            "train_tag": "v1",
            "pretrained": True,
            "dropout": 0.5,
            "lr": 3e-4,
            "weight_decay": 1e-3,
            "epochs": 40,
            "batch_size": 16,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.05,
            "mixup_alpha": 0.15,
            "freeze_epochs": 5,
            "description": "Batch=16 + medium reg (more gradient noise)",
        },
        # --- Frozen backbone (head-only training) ---
        {
            "name": "frozen_v1",
            "gate": "mnv3_small",
            "train_tag": "v1",
            "pretrained": True,
            "dropout": 0.5,
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "epochs": 30,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "normal",
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "freeze_epochs": 999,
            "description": "Backbone fully frozen — only classifier head trains",
        },
        {
            "name": "frozen_v3",
            "gate": "mnv3_small",
            "train_tag": "v3",
            "pretrained": True,
            "dropout": 0.5,
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "epochs": 30,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "normal",
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "freeze_epochs": 999,
            "description": "Backbone fully frozen on v3",
        },
        # --- Low LR + long training ---
        {
            "name": "low_lr_v1",
            "gate": "mnv3_small",
            "train_tag": "v1",
            "pretrained": True,
            "dropout": 0.5,
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "epochs": 80,
            "batch_size": 16,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.05,
            "mixup_alpha": 0.1,
            "freeze_epochs": 10,
            "description": "Low LR=1e-4, bs=16, heavy aug, long training",
        },
        # --- EfficientNet comparison ---
        {
            "name": "effnetb0_v1_heavy",
            "gate": "effnetb0",
            "train_tag": "v1",
            "pretrained": True,
            "dropout": 0.5,
            "lr": 5e-4,
            "weight_decay": 5e-3,
            "epochs": 40,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.2,
            "freeze_epochs": 5,
            "description": "EfficientNet-B0 with heavy regularisation (capacity comparison)",
        },
        # --- Extreme regularisation ---
        {
            "name": "extreme_reg_v1",
            "gate": "mnv3_small",
            "train_tag": "v1",
            "pretrained": True,
            "dropout": 0.7,
            "lr": 3e-4,
            "weight_decay": 5e-2,
            "epochs": 50,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.2,
            "mixup_alpha": 0.4,
            "freeze_epochs": 10,
            "description": "Extreme: dropout=0.7, WD=5e-2, LS=0.2, mixup=0.4",
        },
        {
            "name": "extreme_reg_v3",
            "gate": "mnv3_small",
            "train_tag": "v3",
            "pretrained": True,
            "dropout": 0.7,
            "lr": 3e-4,
            "weight_decay": 5e-2,
            "epochs": 50,
            "batch_size": 32,
            "scheduler": "cosine",
            "aug_strength": "heavy",
            "label_smoothing": 0.2,
            "mixup_alpha": 0.4,
            "freeze_epochs": 10,
            "description": "Extreme reg on v3",
        },
    ]
    return experiments


# ---------------------------------------------------------------------------
# Run single experiment
# ---------------------------------------------------------------------------
def run_experiment(exp: dict, device: str, out_dir: Path) -> dict:
    name = exp["name"]
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {name}")
    print(f"  {exp['description']}")
    print(f"{'='*70}")

    set_seed(42)
    tag = exp["train_tag"]
    input_size = 224

    train_csv = SPLITS_DIR / f"{tag}_train_gate_mix.csv"
    val_csv = SPLITS_DIR / f"{tag}_val_gate.csv"
    test_csv = SPLITS_DIR / f"{tag}_test_gate.csv"
    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    train_tf = get_train_transforms(input_size, exp["aug_strength"])
    eval_tf = get_eval_transforms(input_size)

    train_ds = GateDataset(train_csv, transform=train_tf)
    val_ds = GateDataset(val_csv, transform=eval_tf)
    test_ds = GateDataset(test_csv, transform=eval_tf)

    bs = exp["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    n_neg, n_pos = train_ds.class_counts()
    print(f"  Train: {len(train_ds)} (normal={n_neg}, anomaly={n_pos})")
    print(f"  Val:   {len(val_ds)}")
    print(f"  Test:  {len(test_ds)}")

    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    if exp["label_smoothing"] > 0:
        criterion = LabelSmoothingBCELoss(smoothing=exp["label_smoothing"], pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    model = build_model(exp["gate"], dropout=exp["dropout"], pretrained=exp["pretrained"])
    if exp["freeze_epochs"] > 0:
        freeze_backbone(model, exp["gate"])
    model = model.to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=exp["lr"],
        weight_decay=exp["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp["epochs"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    patience = 10

    start_time = time.time()
    for epoch in range(1, exp["epochs"] + 1):
        if epoch == exp["freeze_epochs"] + 1 and exp["freeze_epochs"] < 999:
            print(f"  >> Unfreezing backbone at epoch {epoch}")
            unfreeze_all(model)
            optimizer = optim.AdamW(model.parameters(), lr=exp["lr"] * 0.1, weight_decay=exp["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp["epochs"] - epoch + 1)

        t_loss, t_acc = train_one_epoch(
            model, train_dl, criterion, optimizer, device, mixup_alpha=exp["mixup_alpha"]
        )
        v_loss, v_acc, _, _ = evaluate(model, val_dl, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["lr"].append(current_lr)

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            gap = t_acc - v_acc
            print(f"  Epoch {epoch:3d}/{exp['epochs']}  "
                  f"t_loss={t_loss:.4f} t_acc={t_acc:.4f}  "
                  f"v_loss={v_loss:.4f} v_acc={v_acc:.4f}  "
                  f"gap={gap:+.4f}  lr={current_lr:.6f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Test eval (same split)
    _, test_acc, test_probs, test_labels = evaluate(model, test_dl, criterion, device)
    test_metrics = compute_metrics(test_labels, test_probs)
    print(f"\n  Test results (same-split {tag}):")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")

    # Cross-version evaluation
    print(f"\n  Cross-version evaluation...")
    cross_results = cross_version_eval(model, device, input_size)
    cross_f1_vals = [m.get("f1", 0) for m in cross_results.values()]
    cross_mean_f1 = np.mean(cross_f1_vals) if cross_f1_vals else 0
    print(f"  Cross-eval mean F1: {cross_mean_f1:.4f}")
    for eval_name, metrics in sorted(cross_results.items()):
        print(f"    {eval_name}: F1={metrics.get('f1', 0):.4f}  "
              f"acc={metrics.get('accuracy', 0):.4f}")

    # Plot
    exp_dir = out_dir / name
    plot_experiment_curves(history, name, exp_dir)
    plot_cross_eval_heatmap(cross_results, name, exp_dir)

    # Save model
    model_path = exp_dir / f"{name}_gate.pt"
    torch.save({
        "gate_name": exp["gate"],
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "experiment": exp,
    }, model_path)

    result = {
        "name": name,
        "gate": exp["gate"],
        "train_tag": tag,
        "description": exp["description"],
        "pretrained": exp["pretrained"],
        "dropout": exp["dropout"],
        "lr": exp["lr"],
        "weight_decay": exp["weight_decay"],
        "label_smoothing": exp["label_smoothing"],
        "mixup_alpha": exp["mixup_alpha"],
        "aug_strength": exp["aug_strength"],
        "actual_epochs": len(history["train_loss"]),
        "best_val_loss": float(best_val_loss),
        "test_acc": test_metrics.get("accuracy", 0),
        "test_f1": test_metrics.get("f1", 0),
        "test_auroc": test_metrics.get("auroc", 0),
        "test_precision": test_metrics.get("precision", 0),
        "test_recall": test_metrics.get("recall", 0),
        "cross_mean_f1": cross_mean_f1,
        "generalisation_gap": test_metrics.get("f1", 0) - cross_mean_f1,
        "elapsed_sec": elapsed,
        "cross_detail": cross_results,
    }

    # Save per-experiment JSON
    json_path = exp_dir / f"{name}_results.json"
    serialisable = {k: v for k, v in result.items()}
    serialisable["cross_detail"] = {
        k: {kk: float(vv) for kk, vv in v.items()} for k, v in cross_results.items()
    }
    json_path.write_text(json.dumps(serialisable, indent=2, default=str))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated experiment indices (0-based) to run, e.g. '0,1,5'")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    deadline = datetime.now() + timedelta(hours=args.max_hours)

    experiments = get_experiments()

    if args.experiments:
        indices = [int(x.strip()) for x in args.experiments.split(",")]
        experiments = [experiments[i] for i in indices if i < len(experiments)]

    print(f"{'='*70}")
    print(f"  24-HOUR HYPERPARAMETER SEARCH")
    print(f"  Device: {args.device}")
    print(f"  Experiments: {len(experiments)}")
    print(f"  Deadline: {deadline.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}")

    for i, exp in enumerate(experiments):
        print(f"\n  [{i}] {exp['name']}: {exp['description']}")

    all_results = []
    for i, exp in enumerate(experiments):
        if datetime.now() >= deadline:
            print(f"\n⏰ Time limit reached. Completed {i}/{len(experiments)} experiments.")
            break

        remaining = (deadline - datetime.now()).total_seconds() / 3600
        print(f"\n[{i+1}/{len(experiments)}] Starting '{exp['name']}' "
              f"(~{remaining:.1f}h remaining)")

        try:
            result = run_experiment(exp, args.device, OUTPUT_DIR)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ❌ FAILED: {exp['name']}: {e}")
            traceback.print_exc()
            all_results.append({
                "name": exp["name"],
                "description": exp["description"],
                "error": str(e),
                "test_f1": 0,
                "cross_mean_f1": 0,
                "generalisation_gap": 0,
            })

        # Write incremental summary after each experiment
        summary_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "cross_detail"}
            for r in all_results
        ])
        summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
        plot_summary_comparison(summary_df, OUTPUT_DIR)

        print(f"\n  ✓ Summary updated: {OUTPUT_DIR / 'summary.csv'}")

    # Final summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "cross_detail"}
        for r in all_results
    ])

    if not summary_df.empty:
        summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
        plot_summary_comparison(summary_df, OUTPUT_DIR)

        print(summary_df[[
            "name", "test_f1", "cross_mean_f1", "generalisation_gap"
        ]].to_string(index=False))

        best_idx = summary_df["cross_mean_f1"].idxmax()
        best = summary_df.iloc[best_idx]
        print(f"\n  🏆 Best generalising model: {best['name']}")
        print(f"     Test F1: {best['test_f1']:.4f}")
        print(f"     Cross-version mean F1: {best['cross_mean_f1']:.4f}")
        print(f"     Gap: {best['generalisation_gap']:.4f}")

    print(f"\n  Results: {OUTPUT_DIR}")
    print(f"  Summary CSV: {OUTPUT_DIR / 'summary.csv'}")
    print(f"  Comparison charts: {OUTPUT_DIR / 'comparison_f1.png'}")


if __name__ == "__main__":
    main()
