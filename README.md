# Military Vehicle Detection (YOLOv8)

Object detection for military vehicles, people, and structures in aerial and
ground imagery, built with Ultralytics YOLOv8.

This project is as much about **data quality and rigorous experimentation** as
about the model. The largest gains came from fixing the dataset and the training
recipe — not from swapping architectures. Every result below is backed by a
clean, one-variable-at-a-time experiment.

**Final model:** `yolov8s` @ 640px, mixup disabled, trained on a
taxonomy-cleaned, re-split dataset.

---

## Results (held-out test split)

| Scope | mAP50 | mAP50-95 | recall |
|---|---|---|---|
| All 10 classes | 0.757 | 0.471 | 0.726 |
| Core 8 (excl. `trench`, `vehicle`) | **0.795** | — | — |

`trench` (only 51 instances) and `vehicle` (a generic, ambiguous catch-all) are
kept in the dataset and trained on, but excluded from the *core* headline metric
with justification. Full per-class breakdown and every experiment is in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

---

## The short story

The original notebook scored ~0.58 on the raw data. The path to the final model:

| Lever | Effect on mAP50 | Verdict |
|---|---|---|
| Fix broken train/test split | **+0.20** | by far the biggest |
| Disable mixup (training recipe) | **+0.07** | big |
| Model capacity `n -> s` | +0.04 | modest help |
| Model capacity `s -> m` | -0.03 | too much model |
| Resolution `640 -> 1280` | -0.03 | scale mismatch, hurt |

The biggest wins were the unglamorous ones — data and recipe — while the
"obvious" upgrades (bigger model, higher resolution) gave diminishing returns or
backfired. There's a real optimum in the middle. The full journey, **including
the wrong turns and what corrected them**, is in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

---

## Classes (10)

`tank`, `truck`, `armored_car`, `car`, `rszo` (rocket artillery),
`sau` (self-propelled artillery), `plane`, `person`, `vehicle` (generic),
`trench`

---

## Project structure

```
military-vehicle-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── .env                      # local only — Roboflow API key (never committed)
├── configs/                  # one YAML per experiment
│   ├── baseline.yaml
│   ├── resplit.yaml
│   ├── yolov8s_640.yaml
│   ├── yolov8s_1280.yaml
│   ├── s640_nomixup.yaml     # <- the final model
│   └── m640_nomixup.yaml
├── data/                     # datasets (gitignored)
├── runs/                     # training outputs (gitignored)
├── src/
│   ├── download_dataset.py   # pull from Roboflow
│   ├── audit_dataset.py      # class counts, flags, samples
│   ├── clean_taxonomy.py     # 16 -> 10 class harmonization
│   ├── resplit_dataset.py    # stratified train/valid/test re-split
│   ├── train.py              # config-driven training
│   └── evaluate.py           # test-split metrics (dual-scope)
└── docs/
    ├── EXPERIMENTS.md        # the full journey, dead ends included
    ├── MATH.md               # metrics + loss-function math
    └── PROJECT_NOTES.md      # YOLO background + phase-by-phase log
```

Code in `src/`, data in `data/`, config in `configs/`, outputs in `runs/`. All
scripts are location-independent (paths resolve from the project root).

---

## Reproduce

```bash
# 1. Clone and enter
git clone https://github.com/mmoz-root/military-vehicle-detection.git
cd military-vehicle-detection

# 2. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Credentials — create a .env in the project root:
echo "ROBOFLOW_API_KEY=your_key_here" > .env

# 4. Build the dataset (download -> clean taxonomy -> stratified re-split)
python src/download_dataset.py
python src/clean_taxonomy.py
python src/resplit_dataset.py

# 5. Train the final model and evaluate on the test split
python src/train.py    --config configs/s640_nomixup.yaml
python src/evaluate.py --weights runs/s640_nomixup/weights/best.pt \
                       --data data/dataset_split/data.yaml
```

Training was done on Kaggle (2x Tesla T4). Any CUDA GPU works; adjust
`device`/`batch` in the config to fit your hardware.

---

## Documentation

- **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** — the full experiment log:
  every hypothesis, result, and dead end, with the leverage ranking.
- **[docs/MATH.md](docs/MATH.md)** — the math: IoU, precision/recall, AP/mAP,
  and the YOLOv8 loss (CIoU + DFL + BCE), each with plain-English intuition.
- **[docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md)** — YOLO background and the
  phase-by-phase build log.

---

## Known limitations

- **`trench`** — only 51 instances; too scarce to learn or evaluate reliably.
- **`vehicle`** — a generic catch-all that overlaps the specific vehicle types
  by definition; inherently ambiguous.
- **Small objects** (`person`) remain the hardest, a known challenge for aerial
  imagery where objects span only a few pixels.
- **Video-frame artifacts** — some images are frames from surveillance/drone
  footage and contain player overlays or timestamps baked into the pixels.

---

## Tech stack

Ultralytics YOLOv8 (PyTorch backend), fine-tuning COCO-pretrained weights via
transfer learning. Dataset from Roboflow. Trained on Kaggle GPUs.
