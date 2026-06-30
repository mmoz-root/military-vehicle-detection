#!/usr/bin/env python3
"""
train.py
--------
Train a YOLOv8 detector from a YAML config file.

Every hyperparameter lives in the config (configs/*.yaml), so runs are
reproducible and comparable — no settings buried in code. Paths in the
config are resolved relative to the project root, so this runs correctly
from any working directory.

Usage:
  python src/train.py --config configs/baseline.yaml
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(p) -> Path:
    """Make a path absolute relative to the project root (if not already)."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml",
                    help="path to a training config YAML")
    args = ap.parse_args()

    cfg_path = resolve(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    # 'model' selects the weights for YOLO(); everything else is a train() kwarg.
    model_weights = cfg.pop("model")
    cfg["data"] = str(resolve(cfg["data"]))
    cfg["project"] = str(resolve(cfg.get("project", "runs")))

    print("=" * 60)
    print(f"Config:  {cfg_path}")
    print(f"Model:   {model_weights}")
    print(f"Data:    {cfg['data']}")
    print(f"Output:  {cfg['project']}/{cfg.get('name', 'exp')}")
    print("=" * 60)

    model = YOLO(model_weights)
    model.train(**cfg)

    print("\nTraining complete.")
    print(f"Best weights: {cfg['project']}/{cfg.get('name', 'exp')}/weights/best.pt")
    print("Next: evaluate on the test split with")
    print(f"  python src/evaluate.py --weights "
          f"{cfg.get('project','runs')}/{cfg.get('name','exp')}/weights/best.pt")


if __name__ == "__main__":
    main()
