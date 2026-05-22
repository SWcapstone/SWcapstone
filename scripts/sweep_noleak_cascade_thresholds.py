#!/usr/bin/env python3
"""
Sweep cascade thresholds on the noleak splits.

For each version:
1. Load noleak gate + PatchCore models.
2. Score val_gate and test_gate once each.
3. Search (T_low, T_high) on val_gate.
4. For each threshold pair, fit the PatchCore decision threshold only on the
   uncertain val subset.
5. Apply the selected (T_low, T_high, patchcore_threshold) to test_gate.

This avoids selecting thresholds directly on the final test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.gate_model import GateModel
from backend.src.heatmap_model import PatchCoreModel


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.method: Optional[str] = None
        self.ir = None
        self.calibrator = None

    def predict(self, x):
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        model = self.ir if self.ir is not None else self.calibrator
        if model is None:
            return arr
        out = model.predict(arr)
        return np.asarray(out).reshape(-1)


sys.modules["__main__"].IsotonicCalibrator = IsotonicCalibrator

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform(input_size: int = 224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_split(csv_path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["path"]
            if not os.path.exists(path):
                continue
            samples.append({
                "path": path,
                "label": 1 if row["label"] == "anomaly" else 0,
            })
    return samples


def find_best_patchcore_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0:
        return 0.0
    candidates = np.unique(np.concatenate([
        np.linspace(scores.min(), scores.max(), 101),
        np.quantile(scores, np.linspace(0.0, 1.0, 21)),
    ]))
    best_t = float(candidates[0])
    best_f1 = -1.0
    for t in candidates:
        preds = (scores >= t).astype(np.int32)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def score_samples(
    samples: List[Dict[str, Any]],
    gate: GateModel,
    patchcore: PatchCoreModel,
    transform,
    split_name: str,
    log_every: int = 200,
) -> Dict[str, np.ndarray]:
    labels: List[int] = []
    gate_scores: List[float] = []
    patchcore_scores: List[float] = []
    total = len(samples)
    start = time.perf_counter()
    for idx, s in enumerate(samples, start=1):
        image = Image.open(s["path"]).convert("RGB")
        tensor = transform(image)
        gate_scores.append(float(gate.predict(tensor)))
        patchcore_scores.append(float(patchcore.predict(tensor)["anomaly_score"]))
        labels.append(int(s["label"]))
        if idx % log_every == 0 or idx == total:
            elapsed = time.perf_counter() - start
            rate = idx / max(elapsed, 1e-8)
            print(
                f"    {split_name}: {idx}/{total} "
                f"({elapsed:.1f}s, {rate:.1f} img/s)"
            )
    return {
        "labels": np.asarray(labels, dtype=np.int32),
        "gate_scores": np.asarray(gate_scores, dtype=np.float64),
        "patchcore_scores": np.asarray(patchcore_scores, dtype=np.float64),
    }


def load_cached_scores(cache_path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not cache_path.exists():
        return None
    cached = np.load(cache_path)
    return {
        "labels": cached["labels"],
        "gate_scores": cached["gate_scores"],
        "patchcore_scores": cached["patchcore_scores"],
    }


def save_cached_scores(cache_path: Path, scored: Dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        labels=scored["labels"],
        gate_scores=scored["gate_scores"],
        patchcore_scores=scored["patchcore_scores"],
    )


def load_or_score_samples(
    samples: List[Dict[str, Any]],
    gate: GateModel,
    patchcore: PatchCoreModel,
    transform,
    cache_path: Path,
    split_name: str,
    force_rescore: bool = False,
) -> Dict[str, np.ndarray]:
    if not force_rescore:
        cached = load_cached_scores(cache_path)
        if cached is not None:
            print(f"    {split_name}: loaded cached scores from {cache_path}")
            return cached

    scored = score_samples(samples, gate, patchcore, transform, split_name=split_name)
    save_cached_scores(cache_path, scored)
    print(f"    {split_name}: saved scores to {cache_path}")
    return scored


def evaluate_with_thresholds(scored: Dict[str, np.ndarray], t_low: float, t_high: float, pc_thr: float) -> Dict[str, float]:
    labels = scored["labels"]
    gate_scores = scored["gate_scores"]
    patchcore_scores = scored["patchcore_scores"]
    preds = np.zeros_like(labels)
    normal_mask = gate_scores < t_low
    anomaly_mask = gate_scores > t_high
    uncertain_mask = ~(normal_mask | anomaly_mask)
    preds[normal_mask] = 0
    preds[anomaly_mask] = 1
    preds[uncertain_mask] = (patchcore_scores[uncertain_mask] >= pc_thr).astype(np.int32)
    metrics = compute_metrics(labels, preds)
    metrics["t_low"] = float(t_low)
    metrics["t_high"] = float(t_high)
    metrics["patchcore_threshold"] = float(pc_thr)
    metrics["heatmap_call_rate"] = float(uncertain_mask.mean()) if len(uncertain_mask) else 0.0
    return metrics


def choose_thresholds(
    scored_val: Dict[str, np.ndarray],
    t_low_range: List[float],
    t_high_range: List[float],
    min_recall: float,
    max_call_rate: float,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    best: Optional[Dict[str, float]] = None
    best_key: Tuple[float, float, float] = (-1.0, -1.0, -1.0)

    labels = scored_val["labels"]
    gate_scores = scored_val["gate_scores"]
    patchcore_scores = scored_val["patchcore_scores"]

    for t_low in t_low_range:
        for t_high in t_high_range:
            if t_low >= t_high:
                continue
            uncertain_mask = (gate_scores >= t_low) & (gate_scores <= t_high)
            pc_thr = find_best_patchcore_threshold(patchcore_scores[uncertain_mask], labels[uncertain_mask])
            metrics = evaluate_with_thresholds(scored_val, t_low, t_high, pc_thr)
            rows.append(metrics)

            qualifies = (
                metrics["recall"] >= min_recall and
                metrics["heatmap_call_rate"] <= max_call_rate
            )
            key = (metrics["f1"], metrics["recall"], -metrics["heatmap_call_rate"])
            if qualifies and key > best_key:
                best = metrics
                best_key = key

    if best is None:
        rows_sorted = sorted(rows, key=lambda r: (r["f1"], r["recall"], -r["heatmap_call_rate"]), reverse=True)
        best = rows_sorted[0]

    return best, rows


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep noleak cascade thresholds on val, evaluate on test.")
    p.add_argument("--versions", nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-recall", type=float, default=0.90)
    p.add_argument("--max-call-rate", type=float, default=0.50)
    p.add_argument("--cache-dir", default="reports/cache/noleak_threshold_sweep")
    p.add_argument("--force-rescore", action="store_true")
    p.add_argument("--output", default="reports/noleak_threshold_sweep_results.json")
    args = p.parse_args()

    cfg_path = PROJECT_ROOT / "configs" / "noleak.yaml"
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    t_low_range = [float(x) for x in cfg["threshold_sweep"]["T_low_range"]]
    t_high_range = [float(x) for x in cfg["threshold_sweep"]["T_high_range"]]
    transform = get_eval_transform(224)
    cache_root = PROJECT_ROOT / args.cache_dir

    results: List[Dict[str, Any]] = []
    for version in args.versions:
        print("=" * 60)
        print(f"Version: {version}")
        gate_model_path = PROJECT_ROOT / "models" / f"{version}_mnv3_small_noleak_gate.pt"
        gate_calib_path = PROJECT_ROOT / "models" / f"{version}_mnv3_small_noleak_calibrator.pkl"
        patchcore_model_path = PROJECT_ROOT / "models" / f"{version}_patchcore_r18_patchcore.pt"
        val_csv = PROJECT_ROOT / "splits_noleak" / f"{version}_val_gate.csv"
        test_csv = PROJECT_ROOT / "splits_noleak" / f"{version}_test_gate.csv"

        gate = GateModel.load(str(gate_model_path), device=args.device)
        if gate_calib_path.exists() and gate.calibrator is None:
            with gate_calib_path.open("rb") as f:
                payload = pickle.load(f)
            gate.calibrator = payload["calibrator"] if isinstance(payload, dict) and "calibrator" in payload else payload

        patchcore = PatchCoreModel.load(str(patchcore_model_path), device="cpu")
        if torch.is_tensor(patchcore._memory_bank):
            patchcore._memory_bank = patchcore._memory_bank.detach().cpu().numpy()
            from sklearn.neighbors import NearestNeighbors
            patchcore._nn_index = NearestNeighbors(
                n_neighbors=patchcore.k_neighbors, metric="euclidean", algorithm="auto"
            ).fit(patchcore._memory_bank)

        val_samples = load_split(val_csv)
        test_samples = load_split(test_csv)
        print(f"  val samples : {len(val_samples)}")
        print(f"  test samples: {len(test_samples)}")

        scored_val = load_or_score_samples(
            val_samples,
            gate,
            patchcore,
            transform,
            cache_root / f"{version}_val_gate_scores.npz",
            split_name=f"{version} val",
            force_rescore=args.force_rescore,
        )
        scored_test = load_or_score_samples(
            test_samples,
            gate,
            patchcore,
            transform,
            cache_root / f"{version}_test_gate_scores.npz",
            split_name=f"{version} test",
            force_rescore=args.force_rescore,
        )

        best, sweep_rows = choose_thresholds(
            scored_val, t_low_range, t_high_range, args.min_recall, args.max_call_rate
        )
        test_metrics = evaluate_with_thresholds(
            scored_test, best["t_low"], best["t_high"], best["patchcore_threshold"]
        )

        print(f"  selected T_low={best['t_low']:.2f}, T_high={best['t_high']:.2f}, pc_thr={best['patchcore_threshold']:.4f}")
        print(f"  val  f1={best['f1']:.4f}, recall={best['recall']:.4f}, call_rate={best['heatmap_call_rate']:.4f}")
        print(f"  test f1={test_metrics['f1']:.4f}, recall={test_metrics['recall']:.4f}, precision={test_metrics['precision']:.4f}")

        results.append({
            "version": version,
            "selection_on_val": best,
            "test_metrics": test_metrics,
            "search_space": {
                "t_low_range": t_low_range,
                "t_high_range": t_high_range,
                "min_recall": args.min_recall,
                "max_call_rate": args.max_call_rate,
            },
            "n_val": len(val_samples),
            "n_test": len(test_samples),
            "sweep_rows": sweep_rows,
        })

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nSaved results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
