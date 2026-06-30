#!/usr/bin/env python3
"""
clean_taxonomy.py
-----------------
Deduplicate and harmonize the class taxonomy of a YOLOv8 dataset.

It reads the ORIGINAL dataset (read-only) and writes a NEW cleaned copy,
so your raw download is never modified. Every decision lives in the REMAP
dict below -- change one line there to change one merge, nothing else.

Usage:
  python clean_taxonomy.py --src dataset --dst dataset_clean

Requires: pyyaml  (already installed)
"""

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

SPLITS = ["train", "valid", "test"]

# Resolve paths from this file's location (src/ -> project root),
# so the script works no matter which directory you run it from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# THE ONLY THING YOU EDIT.
# Map each ORIGINAL class name -> the name you want it to become.
# Use None to DROP a class entirely (its boxes are removed).
# ---------------------------------------------------------------------------
REMAP = {
    "0":           "armored_car",   # numeric duplicate -> armored vehicle
    "1":           "tank",          # numeric duplicate -> tank
    "2":           "truck",         # numeric duplicate -> truck
    "3":           "armored_car",   # numeric duplicate -> armored vehicle
    "4":           "armored_car",   # numeric duplicate -> armored vehicle
    "Person":      "person",        # case duplicate
    "Trench":      "trench",        # case normalize (kept, low support)
    "Vehicle":     "vehicle",       # generic catch-all (kept)
    "armored_car": "armored_car",
    "car":         "car",
    "person":      "person",
    "plane":       "plane",
    "rszo":        "rszo",
    "sau":         "sau",
    "tank":        "tank",
    "truck":       "truck",
}


def load_names(data_yaml: Path):
    d = yaml.safe_load(Path(data_yaml).read_text())
    names = d["names"]
    if isinstance(names, dict):  # {0: 'tank', ...}
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(PROJECT_ROOT / "data" / "dataset"),
                    help="original dataset root")
    ap.add_argument("--dst", default=str(PROJECT_ROOT / "data" / "dataset_clean"),
                    help="output (cleaned) dataset root")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    old_names = load_names(src / "data.yaml")

    # safety check: every original class must have a remap decision
    missing = [n for n in old_names if n not in REMAP]
    if missing:
        raise SystemExit(f"REMAP is missing entries for: {missing}")

    # new class list = sorted unique non-None targets -> deterministic ids
    new_names = sorted({v for v in REMAP.values() if v is not None})
    new_name_to_id = {n: i for i, n in enumerate(new_names)}

    # old class id -> new class id (or None to drop)
    old_id_to_new = {}
    for old_id, old_name in enumerate(old_names):
        target = REMAP[old_name]
        old_id_to_new[old_id] = new_name_to_id[target] if target is not None else None

    before, after = Counter(), Counter()
    dropped = 0

    for split in SPLITS:
        src_lbl, src_img = src / split / "labels", src / split / "images"
        if not src_lbl.exists():
            continue
        dst_lbl, dst_img = dst / split / "labels", dst / split / "images"
        dst_lbl.mkdir(parents=True, exist_ok=True)
        dst_img.mkdir(parents=True, exist_ok=True)

        for lbl in src_lbl.glob("*.txt"):
            out_lines = []
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                old_id = int(float(parts[0]))
                before[old_names[old_id]] += 1
                new_id = old_id_to_new[old_id]
                if new_id is None:
                    dropped += 1
                    continue
                after[new_names[new_id]] += 1
                out_lines.append(" ".join([str(new_id)] + parts[1:]))
            # write label (empty file = valid YOLO "background" image)
            (dst_lbl / lbl.name).write_text(
                "\n".join(out_lines) + ("\n" if out_lines else "")
            )
            # copy the matching image
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"):
                cand = src_img / f"{lbl.stem}{ext}"
                if cand.exists():
                    shutil.copy2(cand, dst_img / cand.name)
                    break

    # write the new data.yaml
    new_yaml = {
        "path": str(dst.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(new_names),
        "names": new_names,
    }
    (dst / "data.yaml").write_text(yaml.safe_dump(new_yaml, sort_keys=False))

    # provenance: which old names fed each new class
    provenance = defaultdict(list)
    for old_name, new_name in REMAP.items():
        if new_name is not None:
            provenance[new_name].append(old_name)

    print("\n================  BEFORE -> AFTER  ================")
    print(f"Original classes: {len(old_names)}   ->   Cleaned classes: {len(new_names)}")
    print(f"Boxes dropped: {dropped}\n")
    print(f"{'NEW CLASS':<14}{'instances':>10}   <- merged from")
    print("-" * 60)
    for n in new_names:
        srcs = ", ".join(sorted(provenance[n]))
        print(f"{n:<14}{after[n]:>10}   <- {srcs}")
    print("-" * 60)
    print(f"TOTAL boxes:  {sum(after.values()):>10}")
    print(f"\nCleaned dataset written to: {dst.resolve()}")
    print(f"New data.yaml:              {(dst / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
