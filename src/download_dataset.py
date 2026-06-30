#!/usr/bin/env python3
"""
download_dataset.py
-------------------
Download the Roboflow dataset into data/dataset/.

The Roboflow API key is read from a .env file in the project root
(never hardcoded). Paths are resolved relative to this file's location,
so the script works no matter which directory you run it from.

Usage:
  python src/download_dataset.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

# src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEST = PROJECT_ROOT / "data" / "dataset"


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY not found. Make sure a .env file containing\n"
            "  ROBOFLOW_API_KEY=your_key_here\n"
            f"exists in the project root: {PROJECT_ROOT}"
        )

    rf = Roboflow(api_key=api_key)
    project = (
        rf.workspace("muhammed-hocy2")
        .project("military-vehicle-detection-juleg-x1b34")
    )
    dataset = project.version(1).download("yolov8", location=str(DEST))
    print(f"\nDataset downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()
