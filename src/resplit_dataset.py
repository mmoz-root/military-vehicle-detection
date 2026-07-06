#!/usr/bin/env python3
"""
resplit_dataset.py
------------------
Re-split a YOLOv8 dataset into train/valid/test with a stratified RANDOM
split, so every class is distributed across all three splits in the same
proportions.

Why: the original split ran along the seams between merged data sources, so
some classes' test images came from a distribution the training set barely
contained (domain shift) -- which is what tanked `person` and `vehicle`.
A stratified random re-split makes the test set a fair sample of the same
distribution the model trains on.

Deterministic (seeded) so the split is reproducible across runs/machines.

Known limitation: this is a per-image split. If the dataset contains
near-duplicate frames from the same video, some may land in different splits
(mild leakage that can slightly inflate test scores). A group-aware split
would be stricter; documented here as a tradeoff.

Usage:
  python src/resplit_dataset.py --src data/dataset_clean --dst data/dataset_split
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["train", "valid", "test"]


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_names(data_yaml: Path):
    d = yaml.safe_load(Path(data_yaml).read_text())
    names = d["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return names


def label_classes(lbl_path: Path):
    classes = set()
    text = lbl_path.read_text().strip()
    if text:
        for line in text.splitlines():
            parts = line.split()
            if parts:
                classes.add(int(float(parts[0])))
    return classes


def find_image(lbl_path: Path, images_dir: Path):
    stem = lbl_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"):
        c = images_dir / f"{stem}{ext}"
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/dataset_clean")
    ap.add_argument("--dst", default="data/dataset_split")
    ap.add_argument("--ratios", default="0.8,0.1,0.1", help="train,valid,test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = resolve(args.src), resolve(args.dst)
    names = load_names(src / "data.yaml")
    tr, va, te = (float(x) for x in args.ratios.split(","))
    assert abs(tr + va + te - 1.0) < 1e-6, "ratios must sum to 1"
    random.seed(args.seed)

    # 1) Pool every (label, image) pair across all original splits
    pool = []                 # (label_path, image_path, {classes})
    global_counts = Counter()  # class -> total instances (for rarity)
    for split in SPLITS:
        lbl_dir, img_dir = src / split / "labels", src / split / "images"
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            img = find_image(lbl, img_dir)
            if img is None:
                continue
            cls = label_classes(lbl)
            pool.append((lbl, img, cls))
            for c in cls:
                global_counts[c] += 1

    # 2) Stratify: bucket each image by the RAREST class it contains, so
    #    scarce classes (trench, plane) get spread across all splits.
    def rarity_key(classes):
        if not classes:
            return -1  # background-only image
        return min(classes, key=lambda c: global_counts[c])

    buckets = defaultdict(list)
    for item in pool:
        buckets[rarity_key(item[2])].append(item)

    # 3) Split each bucket by ratio, then merge
    assign = {s: [] for s in SPLITS}
    for _key, items in buckets.items():
        items = items[:]
        random.shuffle(items)
        n = len(items)
        n_tr = int(round(n * tr))
        n_va = int(round(n * va))
        assign["train"] += items[:n_tr]
        assign["valid"] += items[n_tr:n_tr + n_va]
        assign["test"] += items[n_tr + n_va:]

    # 4) Write out (copy images + labels)
    for split in SPLITS:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for lbl, img, _ in assign[split]:
            shutil.copy2(lbl, dst / split / "labels" / lbl.name)
            shutil.copy2(img, dst / split / "images" / img.name)

    # 5) New data.yaml
    (dst / "data.yaml").write_text(yaml.safe_dump({
        "path": str(dst.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }, sort_keys=False))

    # 6) Report per-class distribution across the new splits
    per = {s: Counter() for s in SPLITS}
    for split in SPLITS:
        for lbl, _img, _cls in assign[split]:
            for line in lbl.read_text().strip().splitlines():
                if line.split():
                    per[split][int(float(line.split()[0]))] += 1

    print("\n===============  NEW SPLIT (instances per class)  ===============")
    print(f"{'class':<14}{'train':>8}{'valid':>8}{'test':>8}{'test%':>7}")
    print("-" * 47)
    for c in range(len(names)):
        t, v, e = per['train'][c], per['valid'][c], per['test'][c]
        tot = t + v + e
        pct = (100 * e / tot) if tot else 0
        print(f"{names[c]:<14}{t:>8}{v:>8}{e:>8}{pct:>6.0f}%")
    print("-" * 47)
    for split in SPLITS:
        print(f"{split} images: {len(assign[split])}")
    print(f"\nWritten to: {dst.resolve()}")


if __name__ == "__main__":
    main()
