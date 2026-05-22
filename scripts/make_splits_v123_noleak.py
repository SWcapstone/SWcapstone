#!/usr/bin/env python3
"""
Leak-free per-version stratified split generator.

Key difference from make_splits_v123.py:
  Splits are done at the ORIGINAL IMAGE level first, then all augmentations
  of the same original stay together in the same split. This prevents
  data leakage where aug1 of image X is in train and aug2 is in test.

Original image grouping:
  filename "10021_png.rf.abc123_v1_aug1.jpg" -> base_id "10021_png.rf.abc123"
  All files sharing the same base_id are kept in the same split.

Outputs (per version):
  splits_noleak/v{N}_train_normal.csv     normal-only, 70% of originals
  splits_noleak/v{N}_val_mix.csv          15% normals + 30% anomalies
  splits_noleak/v{N}_test_mix.csv         15% normals + 70% anomalies
  splits_noleak/v{N}_train_gate_mix.csv   70% stratified by label
  splits_noleak/v{N}_val_gate.csv         15% stratified
  splits_noleak/v{N}_test_gate.csv        15% stratified
  splits_noleak/v{N}_pool.csv             full pre-split pool

Usage:
  python scripts/make_splits_v123_noleak.py
  python scripts/make_splits_v123_noleak.py --versions v1 v2 v3 --out-dir splits_noleak
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
from collections import defaultdict
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_base_id(filepath: str) -> str:
    fname = filepath.split("/")[-1]
    return re.sub(r"_(v[0-9]+_aug[0-9]+|orig)\.jpg$", "", fname)


def collect_records(version_root: Path, version_tag: str) -> list[dict]:
    records: list[dict] = []
    for sub in ("pretrain", "additional"):
        for cls in ("anomaly", "normal"):
            d = version_root / sub / cls
            if not d.exists():
                print(f"  !! missing {d}")
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in EXTS:
                    records.append({
                        "path": str(p.resolve()),
                        "dataset_type": "Kolektor",
                        "defect_type": "surface_defect",
                        "label": cls,
                        "version": version_tag,
                        "subset": sub,
                        "base_id": extract_base_id(str(p)),
                    })
    return records


def group_by_base_id(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["base_id"]].append(r)
    return dict(groups)


def split_ids(ids: list[str], rng: random.Random, ratios=(0.70, 0.15, 0.15)):
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def expand_ids_to_records(id_list: list[str], groups: dict[str, list[dict]]) -> list[dict]:
    out = []
    for bid in id_list:
        out.extend(groups[bid])
    return out


def split_for_version(records: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    groups = group_by_base_id(records)

    normal_ids = sorted(set(r["base_id"] for r in records if r["label"] == "normal"))
    anomaly_ids = sorted(set(r["base_id"] for r in records if r["label"] == "anomaly"))

    # -- Heatmap splits (normal: 70/15/15, anomaly: 0/30/70) --
    n_train_ids, n_val_ids, n_test_ids = split_ids(list(normal_ids), rng)
    a_val_n = max(1, int(len(anomaly_ids) * 0.30))
    rng.shuffle(anomaly_ids)
    a_val_ids = anomaly_ids[:a_val_n]
    a_test_ids = anomaly_ids[a_val_n:]

    train_normal = expand_ids_to_records(n_train_ids, groups)
    val_mix = expand_ids_to_records(n_val_ids, groups) + expand_ids_to_records(a_val_ids, groups)
    test_mix = expand_ids_to_records(n_test_ids, groups) + expand_ids_to_records(a_test_ids, groups)

    # -- Gate splits (stratified 70/15/15 by label, at original-image level) --
    gate_normal_ids = sorted(set(r["base_id"] for r in records if r["label"] == "normal"))
    gate_anomaly_ids = sorted(set(r["base_id"] for r in records if r["label"] == "anomaly"))

    gn_train, gn_val, gn_test = split_ids(list(gate_normal_ids), rng)
    ga_train, ga_val, ga_test = split_ids(list(gate_anomaly_ids), rng)

    train_gate = expand_ids_to_records(gn_train, groups) + expand_ids_to_records(ga_train, groups)
    val_gate = expand_ids_to_records(gn_val, groups) + expand_ids_to_records(ga_val, groups)
    test_gate = expand_ids_to_records(gn_test, groups) + expand_ids_to_records(ga_test, groups)

    rng.shuffle(train_gate)
    rng.shuffle(val_gate)
    rng.shuffle(test_gate)

    return {
        "train_normal": train_normal,
        "val_mix": val_mix,
        "test_mix": test_mix,
        "train_gate_mix": train_gate,
        "val_gate": val_gate,
        "test_gate": test_gate,
    }


def write_csv(items: list[dict], version_tag: str, split_name: str, filepath: Path) -> None:
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "dataset_type", "defect_type", "label", "round", "split"])
        w.writeheader()
        for r in items:
            w.writerow({
                "path": r["path"],
                "dataset_type": r["dataset_type"],
                "defect_type": r["defect_type"],
                "label": r["label"],
                "round": version_tag,
                "split": split_name,
            })


def validate_no_leakage(out_dir: Path, version_tag: str) -> None:
    files = {
        s: out_dir / f"{version_tag}_{s}.csv"
        for s in ("train_gate_mix", "val_gate", "test_gate")
    }

    def get_base_ids(csv_path):
        ids = set()
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                ids.add(extract_base_id(row["path"]))
        return ids

    train_ids = get_base_ids(files["train_gate_mix"])
    val_ids = get_base_ids(files["val_gate"])
    test_ids = get_base_ids(files["test_gate"])

    tv_leak = train_ids & val_ids
    tt_leak = train_ids & test_ids
    vt_leak = val_ids & test_ids

    if tv_leak:
        print(f"  !! LEAK train-val: {len(tv_leak)} original images")
    else:
        print(f"  train-val leakage: NONE")

    if tt_leak:
        print(f"  !! LEAK train-test: {len(tt_leak)} original images")
    else:
        print(f"  train-test leakage: NONE")

    if vt_leak:
        print(f"  !! LEAK val-test: {len(vt_leak)} original images")
    else:
        print(f"  val-test leakage: NONE")

    # Count stats
    for split_name, fpath in files.items():
        with open(fpath) as f:
            rows = list(csv.DictReader(f))
        n_normal = sum(1 for r in rows if r["label"] == "normal")
        n_anomaly = sum(1 for r in rows if r["label"] == "anomaly")
        n_base = len(set(extract_base_id(r["path"]) for r in rows))
        print(f"  {split_name}: {len(rows)} images ({n_base} originals) | normal={n_normal}, anomaly={n_anomaly}")

    # Path existence
    missing = 0
    for fpath in files.values():
        with open(fpath) as f:
            for r in csv.DictReader(f):
                if not os.path.exists(r["path"]):
                    missing += 1
    if missing == 0:
        print(f"  all paths exist on disk")
    else:
        print(f"  WARNING: {missing} missing paths!")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    ap.add_argument("--out-dir", default="splits_noleak")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    for v in args.versions:
        print(f"\n=== {v} (leak-free) ===")
        version_root = data_root / "split_versions" / v
        if not version_root.exists():
            print(f"  !! version dir not found: {version_root}")
            continue

        records = collect_records(version_root, v)
        groups = group_by_base_id(records)
        n_originals = len(groups)
        n_anom_orig = len(set(r["base_id"] for r in records if r["label"] == "anomaly"))
        n_norm_orig = len(set(r["base_id"] for r in records if r["label"] == "normal"))

        print(f"  pool: {len(records)} images from {n_originals} originals")
        print(f"  originals: normal={n_norm_orig}, anomaly={n_anom_orig}")

        write_csv(records, v, f"{v}_pool", out_dir / f"{v}_pool.csv")

        splits = split_for_version(records, rng)
        for split_name, items in splits.items():
            fname = f"{v}_{split_name}.csv"
            write_csv(items, v, split_name, out_dir / fname)
            n_a = sum(1 for r in items if r["label"] == "anomaly")
            n_n = sum(1 for r in items if r["label"] == "normal")
            n_base = len(set(r["base_id"] for r in items))
            print(f"  {fname}: {len(items)} images ({n_base} originals) | normal={n_n}, anomaly={n_a}")

        print(f"\n  -- leakage validation --")
        validate_no_leakage(out_dir, v)

    print(f"\nDone. Leak-free splits saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
