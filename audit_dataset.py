#!/usr/bin/env python3
"""
audit_dataset.py
----------------
Taxonomy & quality audit for a YOLOv8-format detection dataset
(e.g. a Roboflow export with train/valid/test splits).

What it does:
  1. Reads class names from data.yaml
  2. Counts instances + images per class, per split
  3. Flags low-support classes, likely duplicate names, and numeric names
  4. Saves a few ANNOTATED example images per class so you can SEE
     what each (especially numerically-named) class actually is.

Usage:
  python audit_dataset.py --data path/to/dataset --out audit_out
  # inside a notebook cell:
  # !python audit_dataset.py --data dataset --out audit_out

Requires: pyyaml, opencv-python, pandas  (all already on Kaggle/Colab)
"""

import argparse
import random
from collections import defaultdict, Counter
from pathlib import Path

import yaml
import cv2
import pandas as pd

SPLITS = ["train", "valid", "test"]
random.seed(42)


def load_class_names(data_yaml: Path):
    with open(data_yaml) as f:
        d = yaml.safe_load(f)
    names = d.get("names")
    # names may be a list ['tank', ...] or a dict {0: 'tank', ...}
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return names


def iter_label_files(split_dir: Path):
    labels_dir = split_dir / "labels"
    if not labels_dir.exists():
        return
    yield from labels_dir.glob("*.txt")


def parse_label(lbl_path: Path):
    rows = []
    text = lbl_path.read_text().strip()
    if not text:
        return rows
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            rows.append((cls, cx, cy, w, h))
    return rows


def find_image_for_label(lbl_path: Path):
    img_dir = lbl_path.parent.parent / "images"
    stem = lbl_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"):
        cand = img_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (contains data.yaml)")
    ap.add_argument("--out", default="audit_out", help="output folder for report + samples")
    ap.add_argument("--samples", type=int, default=4, help="example images to save per class")
    args = ap.parse_args()

    root = Path(args.data)
    data_yaml = root / "data.yaml"
    names = load_class_names(data_yaml)
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    inst_counts = defaultdict(Counter)   # inst_counts[split][cls] = #boxes
    img_counts = defaultdict(Counter)    # img_counts[split][cls]  = #images containing cls
    examples = defaultdict(list)         # cls -> list of (label_path, [boxes of that cls])

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for lbl in iter_label_files(split_dir):
            rows = parse_label(lbl)
            classes_here = set()
            for (cls, *_box) in rows:
                inst_counts[split][cls] += 1
                classes_here.add(cls)
            for cls in classes_here:
                img_counts[split][cls] += 1
                if len(examples[cls]) < args.samples:
                    examples[cls].append((lbl, [r for r in rows if r[0] == cls]))

    present = [set(inst_counts[s]) for s in SPLITS if inst_counts[s]]
    all_classes = sorted(set().union(*present)) if present else []

    records = []
    for cls in all_classes:
        name = names[cls] if names and cls < len(names) else f"<id {cls}>"
        rec = {"id": cls, "name": name}
        total = 0
        for s in SPLITS:
            rec[f"{s}_inst"] = inst_counts[s][cls]
            total += inst_counts[s][cls]
        rec["total_inst"] = total
        rec["train_imgs"] = img_counts["train"][cls]
        records.append(rec)

    df = pd.DataFrame(records).sort_values("total_inst", ascending=False)
    df.to_csv(out / "class_report.csv", index=False)

    print("\n==================  CLASS REPORT  ==================")
    print(df.to_string(index=False))

    print("\n==================  FLAGS  ==================")

    low = df[df["total_inst"] < 30]
    if len(low):
        print("\n[LOW SUPPORT  < 30 total instances]  — too few to learn/evaluate:")
        for _, r in low.iterrows():
            print(f"   id {int(r['id']):>2}  {str(r['name']):<14}  total={int(r['total_inst'])}")

    seen = defaultdict(list)
    for _, r in df.iterrows():
        seen[str(r["name"]).lower()].append((int(r["id"]), r["name"]))
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        print("\n[LIKELY DUPLICATE NAMES]  — same label under different ids:")
        for k, v in dupes.items():
            print(f"   '{k}': {v}")

    numeric = [(int(r["id"]), r["name"]) for _, r in df.iterrows()
               if str(r["name"]).strip().lstrip("-").isdigit()]
    if numeric:
        print("\n[NUMERIC NAMES]  — unclear meaning, inspect samples/ to identify:")
        for cid, nm in numeric:
            print(f"   id {cid}  name '{nm}'")

    print(f"\nSaving annotated samples to: {out / 'samples'}")
    for cls, items in examples.items():
        name = names[cls] if names and cls < len(names) else f"cls{cls}"
        safe = "".join(c if c.isalnum() else "_" for c in str(name))
        for i, (lbl, boxes) in enumerate(items):
            img_path = find_image_for_label(lbl)
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            for (_c, cx, cy, w, h) in boxes:
                x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
                x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{cls}:{name}", (x1, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(out / "samples" / f"id{cls:02d}_{safe}_{i}.jpg"), img)

    print("\nDone.")
    print(f"  -> class_report.csv  (the table above)")
    print(f"  -> samples/          (open these to identify classes 0-4)")


if __name__ == "__main__":
    main()
