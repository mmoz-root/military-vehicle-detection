# Project Notes & Learning Log — Military Vehicle Detection (YOLOv8)

Personal study notes: what YOLO is, where `yolov8n` fits, and how this project
is structured phase by phase. Written to be re-read later, not to impress —
if something here isn't clear to future-me, fix it.

---

## Part 1 — What YOLO actually is

### The core idea (why "You Only Look Once")

Before YOLO, the best object detectors were **two-stage** (the R-CNN family:
R-CNN → Fast R-CNN → Faster R-CNN). They first *propose* candidate regions,
then *classify* each one. Accurate, but slow — you run the network many times
per image.

YOLO reframed detection as a **single regression problem**: one forward pass of
one network looks at the whole image once and predicts *all* bounding boxes and
class probabilities simultaneously, over a grid. That's the "you only look once"
— hence real-time speed. This is the **one-stage** paradigm.

- Original paper: *"You Only Look Once: Unified, Real-Time Object Detection"*,
  Redmon et al., 2015/2016 (CVPR 2016). This is "the YOLO paper" to read for
  the foundational concept.
- YOLOv1 divides the image into an S×S grid; each cell predicts a few boxes +
  confidence + class scores. Fast, but weak on small/clustered objects and
  imprecise localization — problems later versions chip away at.

### The lineage (v1 → v8 → beyond)

Each version keeps the one-pass idea and improves the architecture:

| Version | Year | Key change | By |
|---|---|---|---|
| v1 | 2016 | Grid-based single-pass detection | Redmon et al. |
| v2 / YOLO9000 | 2016 | Batch norm, anchor boxes, Darknet-19 | Redmon & Farhadi |
| v3 | 2018 | Darknet-53, multi-scale (FPN-style) prediction | Redmon & Farhadi |
| v4 | 2020 | CSPDarknet53, SPP, PAN neck | Bochkovskiy et al. |
| v5 | 2020 | Move to **PyTorch**, SPPF, mosaic aug, AutoAnchor | Ultralytics |
| v6 / v7 | 2022 | Reparameterization, model scaling | Meituan / Wang et al. |
| **v8** | **2023** | **Anchor-free, decoupled head, C2f module** | **Ultralytics** |
| v9, v10 | 2024 | Efficiency, edge-friendly designs | various |
| YOLO11 | 2024 | Ultralytics dropped the "v"; fewer params, higher mAP | Ultralytics |
| v12+ | 2025+ | ongoing | various |

Note the field forked after v3: Redmon (original author) left, and different
groups now release under the "YOLO" name. **YOLOv8 has no peer-reviewed paper** —
Ultralytics released it as software + docs because the models evolve faster than
the publication cycle. So for v8 specifics, the source of truth is the
Ultralytics documentation, not a paper.

### What makes YOLOv8 different from v5

Three things worth remembering:

1. **Anchor-free.** Older YOLOs predicted boxes as offsets from predefined
   "anchor boxes" (preset shapes/sizes). v8 instead predicts the object's center
   directly. Fewer box predictions, simpler post-processing (NMS), and no need to
   tune anchor priors that may not match a custom dataset's object shapes.
2. **Decoupled head.** The final layer splits into two branches — one for *where*
   the object is (box regression), one for *what* it is (classification) —
   instead of one shared head. Each task can specialize, improving accuracy,
   especially on small/overlapping objects.
3. **C2f module** replaces the older C3/CSPLayer in the backbone — richer
   gradient flow and better feature fusion.

Structurally, every YOLOv8 model is still **backbone → neck → head**:
- **Backbone** (CSPDarknet-derived): extracts features from the image.
- **Neck** (PAN-FPN): fuses features across scales so both small and large
  objects are detectable.
- **Head** (decoupled, anchor-free): outputs the final boxes + classes.

---

## Part 2 — Where `yolov8n` fits (the model I'm using)

YOLOv8 ships as a **family of five sizes**, all the same architecture scaled up
or down by depth/width multipliers:

| Variant | Params | COCO mAP50-95 | Note |
|---|---:|---:|---|
| **n** (nano) | ~3.2M | 37.3 | smallest/fastest — **what I started with** |
| s (small) | ~11M | 44.9 | |
| m (medium) | ~26M | 50.2 | |
| l (large) | ~44M | 52.9 | |
| x (extra) | ~68M+ | ~53.9 | most accurate/slowest |

- **"Nano"** applies depth multiplier **0.33** and width multiplier **0.25** to
  the base architecture — aggressively shrinking layer count and channel width.
- Result: ~3.2M parameters, 8.7 GFLOPs, real-time even on modest/edge hardware
  (~1 ms per image on an A100). Built for speed and small memory footprint.
- The tradeoff is accuracy: nano scores ~7.6 percentage points lower AP on COCO
  than the small variant. On a *specific* task like ours, that gap can shrink —
  but it's real.

**Why this matters for my project:** I started on nano because it's fast to
iterate with. In Phase 2 I'll try `yolov8s` / `yolov8m` — I have GPU headroom, so
trading some speed for accuracy is worth testing. Each size-up is a **one-variable
experiment** measured against the baseline.

### The two things nano gives me "for free"

- **Pretrained weights** (`yolov8n.pt`): the model already learned general visual
  features from COCO (80 everyday classes). I *fine-tune* those on my 10 military
  classes = **transfer learning**. Low-level features (edges, textures, shapes)
  transfer; mainly the later layers + head re-specialize. This is why I get
  decent results from a few thousand images instead of needing millions.
- **The framework** (PyTorch, wrapped by Ultralytics): `model.train()` runs a full
  PyTorch training loop under the hood — I don't write raw PyTorch for a standard
  fine-tune. (I *could* drop to `model.model` for the raw `torch.nn.Module` if I
  ever needed custom loss/architecture. I don't for this.)

Pretrained ≠ framework: they're independent. I benefit from both at once.

---

## Part 3 — How I approached THIS project

Three principles, in priority order:

1. **Data-centric before model-centric.** The original run capped at ~58% mAP50.
   The instinct is "use a bigger model." But auditing revealed the real ceiling
   was a **contaminated taxonomy**, not model capacity — 16 classes that were
   actually a merge of multiple sources with duplicate (`Person`/`person`),
   generic (`Vehicle`), and unnamed-numeric (`0–4`) classes. Fixing data is
   higher-leverage than swapping models. **Lesson: look at your data first.**

2. **Baseline-driven, one variable at a time.** Retrain the *same* model
   (`yolov8n`, same hyperparameters) on the *cleaned* data first. Any change in
   mAP is then attributable to the data alone. Only after that do I change the
   model or the augmentation — one knob per experiment — so I can *prove* what
   helped instead of guessing.

3. **Reproducible & documented.** Every step is a committed script, not a
   notebook cell: credentials in `.env` (never in code), config-driven training,
   location-independent paths, a README that tells the story. Anyone can clone,
   set a key, and rebuild the exact cleaned dataset.

The narrative this produces: *"I diagnosed a data problem and proved the fix moved
the needle"* — stronger than *"I trained a model."*

---

## Part 4 — Phase-by-phase breakdown

### Phase 0 — Data foundation
- Rotated the leaked Roboflow API key; moved it to a git-ignored `.env`.
- **Audited** the dataset (`audit_dataset.py`): per-class counts, duplicate/
  numeric-name flags, saved annotated sample images.
- **Identified** the numeric classes by visual inspection (`1`→tank, `2`→truck,
  `0`/`3`/`4`→armored vehicles).
- **Harmonized** the taxonomy 16 → 10 classes, **0 boxes lost** (22,691 kept),
  via a reproducible config-driven script (`clean_taxonomy.py`), writing a clean
  copy and leaving the raw download untouched.

### Phase 1 — Repo structure
- Restructured into `src/` (code), `data/` (datasets, gitignored),
  `configs/` (hyperparameters), `runs/` (outputs, gitignored).
- Made all scripts **location-independent** (resolve paths from project root).
- Fixed a data leak (untracked `dataset_clean/` from Git — `--cached`, no files
  lost).
- Wrote the README (data story + reproduce steps + known limitations) and pinned
  dependencies in `requirements.txt`.

### Phase 2 — Model training & improvement (in progress)
1. Training config (`configs/baseline.yaml`) — mirrors original settings.
2. Train + eval scripts (`src/train.py`, `src/evaluate.py`) — eval reports on
   the **held-out test split**, which the original notebook never did.
3. **Baseline run** — retrain nano on clean data; measure jump from ~0.58.
4. **Compute decision** — Kaggle (fast/familiar) vs Modal (better long-term).
5. **Model improvements** — nano → s/m, revisit augmentation, tune; each change
   measured against the baseline.

---

## Part 5 — Glossary (quick reference)

- **mAP** — mean Average Precision. Averages detection precision across classes
  and recall levels. The headline detection metric.
- **mAP50** — mAP at IoU threshold 0.5 (a box counts as correct if it overlaps
  the true box by ≥50%). More forgiving.
- **mAP50-95** — averaged over IoU thresholds 0.5→0.95. Stricter, rewards tight
  localization. (Nano's COCO 37.3 is this metric.)
- **IoU** (Intersection over Union) — overlap between predicted and true box,
  0→1. The basis for "is this detection correct?"
- **Precision / Recall** — precision = of my detections, how many were right;
  recall = of the real objects, how many I found.
- **Anchor box** — a preset box shape/size used as a starting guess (older YOLOs).
- **Anchor-free** — predict the object center directly, no presets (YOLOv8).
- **Backbone / Neck / Head** — feature extractor / multi-scale fuser / final
  predictor.
- **NMS** (Non-Maximum Suppression) — post-processing that removes duplicate
  overlapping boxes for the same object.
- **Transfer learning / fine-tuning** — start from weights pretrained on a big
  dataset (COCO) and adapt them to a smaller task-specific one.
- **Mosaic augmentation** — training trick that stitches 4 images into one, so
  the model sees objects at more scales/contexts. (`close_mosaic` turns it off
  for the final epochs to stabilize training.)
- **mixup** — blends two images/labels together as augmentation.
- **Epoch** — one full pass over the training set.
- **Patience / early stopping** — stop training if validation stops improving for
  N epochs (mine: 20), to avoid wasting time / overfitting.

---

## References

- Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection* (2016)
- Ultralytics YOLOv8 docs — https://docs.ultralytics.com/models/yolov8
- *A Comprehensive Review of YOLO Architectures* (arXiv:2304.00501)
- *YOLOv5, YOLOv8 and YOLOv10: The Go-To Detectors for Real-time Vision*
  (arXiv:2407.02988)

---

## Results & Iterations

### Iteration 0 — Baseline (yolov8n, clean data, original split)
- **Test mAP50: 0.447** | recall 0.393 | precision 0.751
- Per-class revealed a bimodal split: 6 vehicle classes averaged ~0.63,
  but `person` (0.09), `vehicle` (0.14), `trench` (0.02) cratered the mean.
- Diagnosis: the train/test split ran along the seams between merged data
  sources. `person`/`vehicle` test instances came from a distribution the
  training set barely contained (domain shift). `plane` had 0 test instances.

### Iteration 1 — Stratified re-split (SAME model, only the split changed)
- **Test mAP50: 0.447 -> 0.649** | recall 0.393 -> 0.564
- `person`: 0.091 -> 0.580 (the predicted tell — confirms the diagnosis)
- `plane`: unevaluable -> 0.648 (now has test instances)
- `armored_car`: 0.37 -> 0.82 | `tank`: 0.76 -> 0.88 | `truck`: 0.77 -> 0.82
- **What changed:** pooled all images, re-split stratified so every class
  sits at ~8-11% test. Nothing about the model changed (same yolov8n, 50
  epochs, imgsz 640, same augmentation).
- **Two effects, one change:** (1) fairer *training* — the model finally
  saw enough of every class (recall jumped); (2) fairer *test* — honest
  measurement instead of testing on unseen distribution.
- **Caveat:** baseline (0.447) and re-split (0.649) use DIFFERENT test sets
  by design (old one was broken). So 0.649 is the first *trustworthy*
  number, not a clean "+0.20 of new capability."
- **Lesson:** ~0.20 mAP50 gained without touching the model — purely by
  fixing how data was organized and measured. Data/split integrity can
  matter more than architecture.

### Still-weak classes (kept, documented as limitations)
- `trench` (0.24): ~51 instances total — data scarcity. Needs more data,
  not more tuning.
- `vehicle` (0.44): generic catch-all, conceptually ambiguous vs the
  specific vehicle types.

### Iteration 2 — planned (yolov8s @ imgsz 1280)
- Changing TWO variables at once (model n->s, imgsz 640->1280) to save
  compute — buys result over attribution. Logged as a known tradeoff.
- Target: small objects (`person`, distant vehicles) benefit most from
  capacity + resolution.
