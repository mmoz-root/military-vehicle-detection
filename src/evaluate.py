#!/usr/bin/env python3
"""
evaluate.py
-----------
Evaluate a trained YOLOv8 model on the HELD-OUT TEST split and print
overall + per-class metrics.

This is deliberately separate from training: validation metrics guide
training, but the test split is the honest, untouched measure of how the
model generalizes. Report test numbers in your README.

Usage:
  python src/evaluate.py --weights runs/baseline_yolov8n/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="path to trained weights, e.g. runs/.../weights/best.pt")
    ap.add_argument("--data", default="data/dataset_clean/data.yaml",
                    help="dataset yaml (resolved relative to project root)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--exclude", default="trench,vehicle",
                    help="comma-separated class names to exclude from the "
                         "'core' scope (data-limited / ambiguous classes)")
    args = ap.parse_args()

    weights = resolve(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    data = str(resolve(args.data))

    model = YOLO(str(weights))
    # split='test' evaluates the held-out test set (not val)
    metrics = model.val(data=data, split="test", imgsz=args.imgsz)

    print("\n================  TEST METRICS  ================")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"precision:  {metrics.box.mp:.4f}")
    print(f"recall:     {metrics.box.mr:.4f}")

    names = model.names
    print("\nPer-class mAP50:")
    ap50 = metrics.box.ap50                      # AP@0.5 for classes with data
    for i, c in enumerate(metrics.box.ap_class_index):
        print(f"  {names[int(c)]:<14} {ap50[i]:.4f}")

    # classes that had no test instances won't appear above
    evaluated = {int(c) for c in metrics.box.ap_class_index}
    missing = [names[k] for k in names if k not in evaluated]
    if missing:
        print("\n(No test instances, not evaluated): " + ", ".join(missing))

    # dual-scope: report overall AND a 'core' mean that excludes classes
    # known to be data-limited (trench) or ambiguous by definition (vehicle)
    exclude = {n.strip() for n in args.exclude.split(",") if n.strip()}
    core = [ap50[i] for i, c in enumerate(metrics.box.ap_class_index)
            if names[int(c)] not in exclude]
    if exclude and core:
        core_mean = sum(core) / len(core)
        n_all = len(metrics.box.ap_class_index)
        print("\n----------------  DUAL-SCOPE mAP50  ----------------")
        print(f"All {n_all} classes:                       {metrics.box.map50:.4f}")
        print(f"Core {len(core)} classes (excl. {', '.join(sorted(exclude))}):  {core_mean:.4f}")
        print("  (excluded: trench = too few instances; "
              "vehicle = generic/ambiguous)")


if __name__ == "__main__":
    main()
