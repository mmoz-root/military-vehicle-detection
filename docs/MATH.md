# The Math Behind the Model

Reference notes for the mathematics used in this project — the detection
metrics (how the model is scored) and the loss functions (how it learns).
Medium depth: every formula you actually need, each paired with plain-English
intuition. Formulas are written in plain text so they render anywhere.

---

## Part A — How the model is SCORED (metrics)

### 1. IoU — Intersection over Union

The foundation of everything. Measures how well a predicted box overlaps a
ground-truth box.

```
IoU = area(A ∩ B) / area(A ∪ B)
```

- `A ∩ B` = the overlapping area of the two boxes.
- `A ∪ B` = the total area they cover together.
- Range: 0 (no overlap) to 1 (perfect overlap).

**Why it matters:** a prediction only counts as "correct" if its IoU with a
real object exceeds a threshold (commonly 0.5). IoU is the yardstick for
"did we find it, and did we find it *tightly*?"

### 2. Precision and Recall

Every prediction is sorted into three buckets, given an IoU threshold:

- **TP** (true positive): predicted box matches a real object (IoU ≥ threshold,
  correct class).
- **FP** (false positive): predicted box matches nothing real (or wrong class,
  or a duplicate).
- **FN** (false negative): a real object that was missed entirely.

```
Precision = TP / (TP + FP)     "of what I predicted, how much was right"
Recall    = TP / (TP + FN)     "of what exists, how much I found"
```

**The tension:** lower the confidence threshold and you predict more boxes —
recall rises (you catch more) but precision falls (more false alarms). Raise it
and the reverse happens. There's always a trade-off.

> **This project:** recall was our "is the improvement real?" signal. When
> removing mixup raised recall 0.64 → 0.73, that meant the model was genuinely
> *finding more objects*, not just getting luckier on confidence. When the
> `m` model's recall *dropped*, that revealed it was over-conservative
> (too much model for the data).

### 3. The Precision–Recall curve → Average Precision (AP)

Sweep the confidence threshold from high to low. Each threshold gives one
(precision, recall) point. Plot them all → the **PR curve**.

**Average Precision** is the area under that curve:

```
AP = ∫₀¹ p(r) dr        (precision p as a function of recall r, from 0 to 1)
```

In practice it's computed by interpolating the curve to be monotonically
decreasing, then summing the area. Intuition: **AP rewards a model that keeps
precision high even as recall increases** — i.e., stays accurate while catching
more objects. One number, per class, that summarizes the whole trade-off curve.

### 4. mAP — mean Average Precision

AP is per-class at one IoU threshold. We aggregate:

```
mAP@0.5      = mean over all classes of AP, measured at IoU threshold 0.5
mAP@0.5:0.95 = average of mAP measured at IoU = 0.50, 0.55, ..., 0.95 (10 steps)
```

- **mAP@0.5** ("mAP50") — lenient. A box counts if it's ≥50% overlapping.
  This was our main headline metric.
- **mAP@0.5:0.95** ("mAP50-95") — strict. Averages across tightening IoU
  thresholds, so it rewards *precise* localization, not just rough hits.

> **This project:** our mAP50 (~0.76) sat well above our mAP50-95 (~0.47). That
> gap is diagnostic: it means the model reliably *finds* objects (good at 0.5)
> but its boxes aren't always *tightly* placed (loses points as the threshold
> tightens). Common in aerial imagery where object edges are fuzzy.

### 5. Why "dual-scope" reporting is valid math, not cheating

The mean in mAP is an unweighted average over classes. A class the model
*cannot* learn (e.g. `trench`, 51 instances) contributes equally to the mean as
a well-supported one, dragging it down. Reporting a "core" mean over the
adequately-supported classes is standard scoped evaluation — the same reasoning
papers use when they write "excluding classes with < N samples." We reported
both (all-10 = 0.757, core-8 = 0.795) so nothing is hidden.

---

## Part B — How the model LEARNS (loss functions)

YOLOv8's training loss is a weighted sum of three terms. The weights (gains)
below are the defaults seen in our training logs:

```
L_total = 7.5 · L_box  +  0.5 · L_cls  +  1.5 · L_dfl
          └─ box ─┘        └─ class ─┘     └─ distribution ─┘
```

### 1. Classification loss — Binary Cross-Entropy (BCE)

For each object, "is this class present or not?" scored per class independently.

```
BCE = −[ y · log(p) + (1 − y) · log(1 − p) ]
```

- `y` = target (1 if the class is present, else 0).
- `p` = model's predicted probability for that class.
- Penalty grows sharply as the prediction moves away from the truth
  (log blows up near 0). Confidently wrong = heavily punished.

### 2. Box regression loss — CIoU (Complete IoU)

Plain IoU has a flaw for training: if two boxes don't overlap at all, IoU = 0
and its gradient is flat — no signal about *which way* to move the box. CIoU
fixes this by adding two geometric terms:

```
L_box = 1 − IoU + ρ²(b, b_gt) / c²  +  α · v

  ρ²(b, b_gt) = squared distance between the two box CENTERS
  c           = diagonal of the smallest box enclosing both
  v           = aspect-ratio mismatch term (below)
  α           = v / ((1 − IoU) + v)   (a weighting factor)

  v = (4 / π²) · ( arctan(w_gt/h_gt) − arctan(w/h) )²
```

Three things it rewards, in plain terms:
1. **Overlap** (the `IoU` term) — cover the object.
2. **Center closeness** (the `ρ²/c²` term) — even with no overlap, pulls the
   predicted center toward the true center.
3. **Shape match** (the `v` term) — get the width/height *ratio* right.

### 3. Distribution Focal Loss (DFL)

YOLOv8 doesn't regress each box edge as a single number. It predicts a
**probability distribution** over discrete candidate distances (bins), then
takes the expected value. DFL trains that distribution to concentrate mass
around the true value.

For a true edge distance `y` lying between integer bins `y_i` and `y_{i+1}`:

```
DFL = −[ (y_{i+1} − y) · log(s_i)  +  (y − y_i) · log(s_{i+1}) ]

  s_i, s_{i+1} = predicted (softmax) probabilities on the two nearest bins
```

Intuition: it's cross-entropy that pushes probability onto the two bins
straddling the real edge, weighted by how close each bin is. Predicting a
distribution (instead of one number) lets the model express uncertainty and
localize edges more precisely — which is what lifts mAP50-95.

---

## Part C — One-line summary of each piece

| Symbol / term | Answers the question |
|---|---|
| IoU | How much does my box overlap the real one? |
| Precision | Of my detections, how many were right? |
| Recall | Of the real objects, how many did I find? |
| AP | How good is my whole precision/recall trade-off (per class)? |
| mAP50 | Average AP across classes, at a lenient overlap bar. |
| mAP50-95 | Same, but averaged over strict overlap bars → localization quality. |
| BCE | Class loss — punishes confident wrong labels. |
| CIoU | Box loss — rewards overlap + center + shape. |
| DFL | Edge loss — sharpens a predicted distribution onto the true edge. |
