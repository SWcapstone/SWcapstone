"""
Gate Model – 5-Fold Stratified CV + Focal Loss + TTA
=====================================================
Splits at the **original image** level so no augmentation of the same base
image leaks across train / val / test within any fold.

Improvements over single-split train_gate.py:
  1. Stratified 5-Fold CV  → uses all 277 anomaly originals for evaluation
  2. Focal Loss (γ=2)      → focuses on hard examples in imbalanced data
  3. Test-Time Augmentation → 5× augmented inference, averaged per sample

Usage
-----
    python scripts/train_gate_kfold.py --tag v1 --gate mnv3_small \
        --config configs/noleak.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
import yaml
from PIL import Image
from sklearn.isotonic import IsotonicRegression
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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "assets"

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
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Binary focal loss for imbalanced classification.

    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce
        return loss.mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {"normal": 0, "anomaly": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label_map.get(row["label"], 0)
        return img, label

    @property
    def labels(self):
        return self.df["label"].map(self.label_map).values

    def class_counts(self):
        labels = self.labels
        return int((labels == 0).sum()), int((labels == 1).sum())


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transforms(input_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(input_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms(input_size: int = 224) -> List[T.Compose]:
    """5 TTA variants: original + 4 augmented views."""
    base_resize = T.Resize((input_size, input_size))
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return [
        T.Compose([base_resize, T.ToTensor(), norm]),
        T.Compose([base_resize, T.RandomHorizontalFlip(p=1.0), T.ToTensor(), norm]),
        T.Compose([base_resize, T.RandomVerticalFlip(p=1.0), T.ToTensor(), norm]),
        T.Compose([base_resize, T.RandomRotation((90, 90)), T.ToTensor(), norm]),
        T.Compose([base_resize, T.RandomRotation((270, 270)), T.ToTensor(), norm]),
    ]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_gate_model(gate_name: str, pretrained: bool = True) -> nn.Module:
    if gate_name == "effnetb0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.2, inplace=True), nn.Linear(in_f, 1))
    elif gate_name == "mnv3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_f = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_f, 1280), nn.Hardswish(inplace=True),
            nn.Dropout(0.2, inplace=True), nn.Linear(1280, 1))
    elif gate_name == "mnv3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_f = model.classifier[0].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3, inplace=True), nn.Linear(in_f, 1))
    else:
        raise ValueError(f"Unsupported: {gate_name}")
    return model


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += imgs.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels_f = imgs.to(device), labels.float().to(device)
        logits = model(imgs).squeeze(-1)
        loss = criterion(logits, labels_f)
        running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits)
        correct += ((probs >= 0.5).long() == labels.to(device).long()).sum().item()
        total += imgs.size(0)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
    return running_loss / max(total, 1), correct / max(total, 1), np.array(all_probs), np.array(all_labels)


@torch.no_grad()
def predict_tta(model, df: pd.DataFrame, tta_transforms: list, device: str, batch_size: int = 64) -> np.ndarray:
    """Run TTA: for each image, run N transform variants and average probabilities."""
    model.eval()
    n_tta = len(tta_transforms)
    paths = df["path"].tolist()
    all_probs = np.zeros((len(paths), n_tta))

    for t_idx, tfm in enumerate(tta_transforms):
        probs_this = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
            logits = model(imgs).squeeze(-1)
            probs_this.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_probs[:, t_idx] = probs_this

    return all_probs.mean(axis=1)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class IsotonicCalibrator:
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip")
        self.method = "isotonic"

    def fit(self, probs, labels):
        self.ir.fit(probs, labels)

    def predict(self, x):
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        return np.asarray(self.ir.predict(arr)).reshape(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_base_id(path: str) -> str:
    fname = path.split("/")[-1]
    return re.sub(r"_(v[0-9]+_aug[0-9]+|orig)\.jpg$", "", fname)


def compute_optimal_threshold(y_true, probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[min(best_idx, len(thresholds) - 1)])


def bootstrap_ci(y_true, y_prob, y_pred, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    metrics = {"recall": [], "f1": [], "auprc": [], "auroc": []}
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        bt, bp, bpred = y_true[idx], y_prob[idx], y_pred[idx]
        if bt.sum() == 0 or (1 - bt).sum() == 0:
            continue
        metrics["recall"].append(recall_score(bt, bpred))
        metrics["f1"].append(f1_score(bt, bpred))
        metrics["auroc"].append(roc_auc_score(bt, bp))
        p, r, _ = precision_recall_curve(bt, bp)
        metrics["auprc"].append(auc(r, p))
    ci = {}
    for k, vals in metrics.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ci[k] = (float(lo), float(hi))
    return ci


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_aggregate_roc(fold_results: list, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    for i, fr in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(fr["y_true"], fr["y_prob"])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, alpha=0.3, label=f"Fold {i+1} (AUC={a:.3f})")
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std([auc(*roc_curve(fr["y_true"], fr["y_prob"])[:2]) for fr in fold_results])
    ax.plot(mean_fpr, mean_tpr, "b-", lw=2,
            label=f"Mean (AUC={mean_auc:.3f}±{std_auc:.3f})")
    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1),
                     np.clip(mean_tpr + std_tpr, 0, 1), alpha=0.15, color="b")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("5-Fold CV ROC (with ±1 SD band)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_aggregate_pr(fold_results: list, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, fr in enumerate(fold_results):
        p, r, _ = precision_recall_curve(fr["y_true"], fr["y_prob"])
        ap = average_precision_score(fr["y_true"], fr["y_prob"])
        ax.plot(r, p, alpha=0.3, label=f"Fold {i+1} (AP={ap:.3f})")
    y_all = np.concatenate([fr["y_true"] for fr in fold_results])
    p_all = np.concatenate([fr["y_prob"] for fr in fold_results])
    p_agg, r_agg, _ = precision_recall_curve(y_all, p_all)
    ap_agg = average_precision_score(y_all, p_all)
    ax.plot(r_agg, p_agg, "b-", lw=2, label=f"Aggregated (AP={ap_agg:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("5-Fold CV PR Curve")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_aggregate_cm(y_true, y_pred, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_title("5-Fold Aggregated Confusion Matrix")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "anomaly"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "anomaly"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_fold_curves(all_curves: list, save_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, curves in enumerate(all_curves):
        row = i // 3
        col = i % 3
        if i >= 5:
            break
        ax = axes[row][col]
        ep = range(1, len(curves["train_loss"]) + 1)
        ax.plot(ep, curves["train_loss"], label="train loss")
        ax.plot(ep, curves["val_loss"], label="val loss")
        ax.set_title(f"Fold {i+1} (stop: ep {len(curves['train_loss'])})")
        ax.set_xlabel("Epoch"); ax.legend(fontsize=7)
    if len(all_curves) < 6:
        axes[1][2].axis("off")
    fig.suptitle("Per-Fold Training Curves", fontsize=14)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="5-Fold CV gate training with Focal Loss + TTA")
    p.add_argument("--tag", required=True, help="v1, v2, or v3")
    p.add_argument("--gate", required=True, choices=["effnetb0", "mnv3_large", "mnv3_small"])
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "noleak.yaml"))
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--tta", action="store_true", default=True, help="Enable TTA (default: on)")
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    use_tta = not args.no_tta
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = yaml.safe_load(open(args.config))
    seed = cfg.get("seed", 42)
    set_seed(seed)

    gate_cfg = cfg["gate"]["models"][args.gate]
    input_size = gate_cfg["input_size"]
    batch_size = args.batch_size or gate_cfg["batch_size"]
    epochs = args.epochs or gate_cfg["epochs"]
    lr = args.lr or gate_cfg["lr"]
    weight_decay = gate_cfg["weight_decay"]
    patience = gate_cfg.get("early_stopping_patience", 7)
    device = args.device or cfg.get("training", {}).get("device", "cpu")

    run_tag = f"{args.tag}_{args.gate}_kfold"
    n_folds = args.n_folds

    print("=" * 65)
    print("5-Fold Stratified CV + Focal Loss + TTA")
    print(f"  tag        : {args.tag}")
    print(f"  model      : {args.gate}")
    print(f"  device     : {device}")
    print(f"  folds      : {n_folds}")
    print(f"  focal_loss : α={args.focal_alpha}, γ={args.focal_gamma}")
    print(f"  TTA        : {'on (5 views)' if use_tta else 'off'}")
    print(f"  epochs     : {epochs}")
    print(f"  lr         : {lr}")
    print(f"  batch_size : {batch_size}")
    print(f"  seed       : {seed}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load all data from noleak splits
    # ------------------------------------------------------------------
    splits_dir_cfg = cfg.get("data", {}).get("splits_dir", "splits_noleak")
    splits_dir = Path(splits_dir_cfg)
    if not splits_dir.is_absolute():
        splits_dir = PROJECT_ROOT / splits_dir
    if not splits_dir.exists():
        splits_dir = PROJECT_ROOT / "splits_noleak"

    tag = args.tag
    dfs = []
    for split_name in ["train_gate_mix", "val_gate", "test_gate"]:
        csv_path = splits_dir / f"{tag}_{split_name}.csv"
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))
    if not dfs:
        raise FileNotFoundError(f"No split CSVs found in {splits_dir} for tag={tag}")
    all_data = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["path"])

    all_data["base_id"] = all_data["path"].apply(extract_base_id)
    all_data["y"] = (all_data["label"] == "anomaly").astype(int)

    originals = all_data.groupby("base_id")["y"].first().reset_index()
    n_total = len(originals)
    n_anom = originals["y"].sum()
    n_norm = n_total - n_anom

    print(f"\n전체 원본: {n_total} (normal={n_norm}, anomaly={n_anom})")
    print(f"전체 이미지: {len(all_data)}")

    # ------------------------------------------------------------------
    # Stratified K-Fold on originals
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_splits = list(skf.split(originals["base_id"], originals["y"]))

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    tta_transforms = get_tta_transforms(input_size) if use_tta else None

    fold_results = []
    fold_curves = []

    for fold_idx, (train_orig_idx, test_orig_idx) in enumerate(fold_splits):
        set_seed(seed + fold_idx)
        print(f"\n{'='*65}")
        print(f"  FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*65}")

        train_base_ids = set(originals.iloc[train_orig_idx]["base_id"])
        test_base_ids = set(originals.iloc[test_orig_idx]["base_id"])

        train_mask = all_data["base_id"].isin(train_base_ids)
        test_mask = all_data["base_id"].isin(test_base_ids)

        train_df_full = all_data[train_mask].copy()
        test_df = all_data[test_mask].copy()

        # Split train into train (85%) and val (15%) — also at original level
        train_base_list = list(train_base_ids)
        train_labels = [originals.set_index("base_id").loc[b, "y"] for b in train_base_list]
        n_val_orig = max(1, int(len(train_base_list) * 0.15))

        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_val_orig, random_state=seed + fold_idx)
        tr_idx, va_idx = next(sss.split(train_base_list, train_labels))
        val_base_ids = set(np.array(train_base_list)[va_idx])
        real_train_base_ids = set(np.array(train_base_list)[tr_idx])

        train_df = all_data[all_data["base_id"].isin(real_train_base_ids)].copy()
        val_df = all_data[all_data["base_id"].isin(val_base_ids)].copy()

        n_tr_anom = train_df["y"].sum()
        n_tr_norm = len(train_df) - n_tr_anom
        n_va = len(val_df)
        n_te = len(test_df)
        n_te_orig = len(test_base_ids)
        n_te_anom_orig = originals.iloc[test_orig_idx]["y"].sum()

        print(f"  Train: {len(train_df)} imgs (normal={n_tr_norm}, anomaly={n_tr_anom})")
        print(f"  Val  : {n_va} imgs")
        print(f"  Test : {n_te} imgs ({n_te_orig} originals, {n_te_anom_orig} anomaly originals)")

        train_ds = GateDataset(train_df, transform=get_train_transforms(input_size))
        val_ds = GateDataset(val_df, transform=get_eval_transforms(input_size))

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = build_gate_model(args.gate, pretrained=True).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        t_losses, v_losses = [], []

        for epoch in range(1, epochs + 1):
            t_loss, t_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
            v_loss, v_acc, _, _ = evaluate(model, val_dl, criterion, device)
            t_losses.append(t_loss)
            v_losses.append(v_loss)
            scheduler.step()
            cur_lr = optimizer.param_groups[0]["lr"]

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d}/{epochs}  "
                      f"t_loss={t_loss:.4f} t_acc={t_acc:.4f}  "
                      f"v_loss={v_loss:.4f} v_acc={v_acc:.4f}  lr={cur_lr:.6f}")

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stop at epoch {epoch} (patience={patience})")
                    break

        fold_curves.append({"train_loss": t_losses, "val_loss": v_losses})

        if best_state:
            model.load_state_dict(best_state)

        # Calibration on val
        _, _, val_probs, val_labels = evaluate(model, val_dl, criterion, device)
        calibrator = IsotonicCalibrator()
        if len(np.unique(val_labels)) >= 2:
            calibrator.fit(val_probs, val_labels)
        else:
            calibrator = None

        # Test inference (with TTA if enabled)
        if use_tta:
            test_probs_raw = predict_tta(model, test_df, tta_transforms, device, batch_size)
        else:
            test_ds = GateDataset(test_df, transform=get_eval_transforms(input_size))
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            _, _, test_probs_raw, _ = evaluate(model, test_dl, criterion, device)

        if calibrator:
            test_probs = calibrator.predict(test_probs_raw)
        else:
            test_probs = test_probs_raw

        test_labels = test_df["y"].values
        test_base_ids_arr = test_df["base_id"].values

        fold_results.append({
            "y_true": test_labels,
            "y_prob": test_probs,
            "base_ids": test_base_ids_arr,
            "paths": test_df["path"].values,
        })

        # Per-fold quick stats
        if len(np.unique(test_labels)) >= 2:
            thr = compute_optimal_threshold(test_labels, test_probs)
            y_pred = (test_probs >= thr).astype(int)
            f1 = f1_score(test_labels, y_pred)
            rec = recall_score(test_labels, y_pred)
            prec = precision_score(test_labels, y_pred)
            auroc_val = roc_auc_score(test_labels, test_probs)
            print(f"  → Fold {fold_idx+1} test: AUROC={auroc_val:.4f} F1={f1:.4f} "
                  f"Recall={rec:.4f} Precision={prec:.4f} (thr={thr:.3f})")

    # ------------------------------------------------------------------
    # Aggregate across folds
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  AGGREGATED RESULTS (all folds)")
    print("=" * 65)

    all_y = np.concatenate([fr["y_true"] for fr in fold_results])
    all_p = np.concatenate([fr["y_prob"] for fr in fold_results])
    all_base = np.concatenate([fr["base_ids"] for fr in fold_results])

    # Image-level metrics
    opt_thr = compute_optimal_threshold(all_y, all_p)
    all_pred = (all_p >= opt_thr).astype(int)

    img_auroc = roc_auc_score(all_y, all_p)
    img_auprc = average_precision_score(all_y, all_p)
    img_f1 = f1_score(all_y, all_pred)
    img_recall = recall_score(all_y, all_pred)
    img_precision = precision_score(all_y, all_pred)

    # Original-level metrics
    result_df = pd.DataFrame({"base_id": all_base, "y_true": all_y, "y_prob": all_p})
    orig_agg = result_df.groupby("base_id").agg(
        y_true=("y_true", "first"),
        mean_prob=("y_prob", "mean")
    ).reset_index()
    orig_y = orig_agg["y_true"].values
    orig_prob = orig_agg["mean_prob"].values
    orig_pred = (orig_prob >= opt_thr).astype(int)

    orig_auroc = roc_auc_score(orig_y, orig_prob)
    orig_auprc_val = average_precision_score(orig_y, orig_prob)
    orig_f1 = f1_score(orig_y, orig_pred)
    orig_recall = recall_score(orig_y, orig_pred)
    orig_prec = precision_score(orig_y, orig_pred) if orig_pred.sum() > 0 else 0.0

    # Bootstrap CI (original-level)
    ci = bootstrap_ci(orig_y, orig_prob, orig_pred, n_boot=2000, seed=seed)

    print(f"\n  Optimal threshold: {opt_thr:.4f}")
    print(f"  TTA: {'5-view' if use_tta else 'off'}")
    print(f"  Loss: FocalLoss(α={args.focal_alpha}, γ={args.focal_gamma})")
    print(f"\n  원본 수: {len(orig_agg)} (anomaly={int(orig_y.sum())})")
    print(f"\n  {'지표':<12} {'이미지단위':>10} {'원본단위':>10} {'95% CI':>20}")
    print(f"  {'AUROC':<12} {img_auroc:>10.4f} {orig_auroc:>10.4f} [{ci['auroc'][0]:.3f}, {ci['auroc'][1]:.3f}]")
    print(f"  {'AUPRC':<12} {img_auprc:>10.4f} {orig_auprc_val:>10.4f} [{ci['auprc'][0]:.3f}, {ci['auprc'][1]:.3f}]")
    print(f"  {'F1':<12} {img_f1:>10.4f} {orig_f1:>10.4f} [{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]")
    print(f"  {'Recall':<12} {img_recall:>10.4f} {orig_recall:>10.4f} [{ci['recall'][0]:.3f}, {ci['recall'][1]:.3f}]")
    print(f"  {'Precision':<12} {img_precision:>10.4f} {orig_prec:>10.4f}")

    # ------------------------------------------------------------------
    # Comparison with single-split (noleak) results if available
    # ------------------------------------------------------------------
    noleak_cfg_path = MODELS_DIR / f"{args.tag}_mnv3_small_noleak_config.json"
    if noleak_cfg_path.exists():
        with open(noleak_cfg_path) as f:
            prev = json.load(f)
        prev_m = prev.get("test_metrics", {})
        if prev_m:
            print(f"\n  단일 split (noleak) 대비:")
            print(f"    F1:     {prev_m.get('f1',0):.4f} → {orig_f1:.4f} (원본단위)")
            print(f"    Recall: {prev_m.get('recall',0):.4f} → {orig_recall:.4f} (원본단위)")
            print(f"    AUROC:  {prev_m.get('auroc',0):.4f} → {orig_auroc:.4f} (원본단위)")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    prefix = f"{run_tag}_focal_tta" if use_tta else f"{run_tag}_focal"
    plot_aggregate_roc(fold_results, REPORTS_DIR / f"{prefix}_roc.png")
    plot_aggregate_pr(fold_results, REPORTS_DIR / f"{prefix}_pr.png")
    plot_aggregate_cm(all_y, all_pred, REPORTS_DIR / f"{prefix}_cm.png")
    plot_fold_curves(fold_curves, REPORTS_DIR / f"{prefix}_fold_curves.png")
    print(f"\n  Plots saved to: {REPORTS_DIR}/{prefix}_*.png")

    # ------------------------------------------------------------------
    # Save aggregated config
    # ------------------------------------------------------------------
    out_config = {
        "tag": args.tag,
        "run_tag": run_tag,
        "gate": args.gate,
        "method": "5-fold-stratified-cv",
        "loss": f"FocalLoss(alpha={args.focal_alpha}, gamma={args.focal_gamma})",
        "tta": use_tta,
        "n_folds": n_folds,
        "seed": seed,
        "epochs_max": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "optimal_threshold": float(opt_thr),
        "image_level": {
            "auroc": float(img_auroc), "auprc": float(img_auprc),
            "f1": float(img_f1), "recall": float(img_recall),
            "precision": float(img_precision),
        },
        "original_level": {
            "auroc": float(orig_auroc), "auprc": float(orig_auprc_val),
            "f1": float(orig_f1), "recall": float(orig_recall),
            "precision": float(orig_prec),
            "n_originals": int(len(orig_agg)),
            "n_anomaly_originals": int(orig_y.sum()),
        },
        "bootstrap_95ci": {k: {"lo": v[0], "hi": v[1]} for k, v in ci.items()},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path = MODELS_DIR / f"{prefix}_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(out_config, indent=2))
    print(f"  Config saved: {config_path}")

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
