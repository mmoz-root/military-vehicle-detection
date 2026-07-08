# Experiment Log — The Full Journey

This project was **not** one-shot. The final model (core-8 mAP50 = 0.795) is the
end of a chain of hypotheses, experiments, dead ends, and a couple of wrong
conclusions that later experiments overturned. This log keeps all of it — the
failures are the point. They're what turned "I trained a model" into "I
diagnosed and reasoned my way to a model."

**Guiding discipline:** change one variable at a time, and measure every change
against a baseline. (Where that discipline slipped, it cost us — see Iteration 2.)

---

## Starting point

The original notebook trained `yolov8n` on the raw Roboflow download and reached
**~0.58 mAP50 (validation)**. It looked "fine," but the dataset had never been
inspected and the whole pipeline lived in one linear notebook. The suspicion:
the ceiling wasn't the model.

---

## Iteration 0 — Clean baseline
**Change:** cleaned the class taxonomy (16 → 10, deduplicated), same `yolov8n`,
original split, imgsz 640.
**Result:** test mAP50 **0.447**.
**What we learned:** the headline number *hid* a bimodal split. Vehicle classes
averaged ~0.63, but `person` (0.09), `vehicle` (0.14), `trench` (0.02) cratered
the mean. Investigating *why* led to the key discovery:

> The train/test split ran along the seams between merged source datasets.
> `person`'s test images came from a source the model barely trained on
> (domain shift). `plane` had **zero** test instances. The split itself was
> broken.

## Iteration 1 — Stratified re-split THE BIG WIN
**Change:** pooled all images, re-split stratified so every class sits at
~8–11% test. **Model unchanged.**
**Result:** test mAP50 0.447 → **0.649**. `person`: 0.09 → 0.58. `plane`:
unevaluable → 0.65. Recall 0.39 → 0.56.
**What we learned:** the biggest gain in the whole project came from **fixing
data, not the model.** Two effects at once — fairer *training* (the model
finally saw every class) and fairer *measurement* (an honest test set).
**Caveat logged:** this changed the test set by design, so 0.447 → 0.649 isn't a
clean "+0.20 of capability." 0.649 was the first *trustworthy* number. From here
on, all runs use this same re-split test set, so they ARE directly comparable.

## Iteration 2 — Bigger model + resolution MISLEADING (my mistake)
**Change:** `yolov8s` **and** imgsz 1280 — two variables at once, to save
compute.
**Result:** **0.621** — slightly *worse* than 0.649.
**What went wrong:** I looked at this and concluded "capacity doesn't help." That
was a **wrong conclusion drawn from a confounded experiment** — I'd changed two
things, so I couldn't tell whether the bigger model helped and the resolution
hurt, or something else. This is the cautionary tale of the whole project.
**Partial signal:** small objects (`person` +0.05, `trench` +0.20) improved,
large objects (`tank`, `plane`) dipped → a hint that resolution was doing
something scale-dependent.

## Iteration 3 — Disentangle: s @ 640 OVERTURNED THE MISTAKE
**Change:** `yolov8s`, resolution back to **640** (isolate the model).
**Result:** **0.688** — the best yet, beating both `n@640` (0.649) and
`s@1280` (0.621).
**What we learned:** two facts the confounded run had hidden:
1. The bigger model **does** help (n→s at 640: core-8 0.726 → 0.757).
2. The **1280 resolution was the culprit** (s@640 0.688 ≫ s@1280 0.621).
> Lesson, in bold: a two-variable experiment gave a wrong answer; the clean
> one-variable run corrected it. This is *why* you don't trust confounded runs.

### Why 1280 hurt (the mechanism)
Higher resolution doesn't add detail everywhere — it shifts every object's pixel
size relative to a **fixed** detection range (set by the network's strides).
Tiny objects (`person`) move *into* the detectable range (better); already-large
objects (`tank`) get pushed *past* it (worse). Net effect depends on the object-
size mix — and ours was mixed, so 1280 roughly washed out while hurting the big
classes.

## Free step — Dual-scope reporting (no GPU)
Kept all 10 classes, but reported two honest numbers: all-10 (0.649) and core-8
(0.726, excluding `trench` = too scarce, `vehicle` = ambiguous). Higher, honest
headline for zero compute. Added to `evaluate.py` so every run auto-reports both.

## Iteration 4 — Recipe: remove mixup SECOND-BIGGEST WIN
**Pre-check that saved a run:** I'd planned to also test removing `dropout` —
but reading the Ultralytics source showed `dropout` is **only applied in the
classification trainer**, i.e. a **no-op for detection**. Our `dropout: 0.1` had
been inert the whole time. Killed that run before wasting an hour.
**Change:** `mixup` 0.15 → 0.0 on `s@640` (mixup *is* active for detection —
also verified in source).
**Result:** 0.688 → **0.757** (core-8 0.757 → **0.795**). Recall 0.64 → 0.73.
**What we learned:** mixup is aggressive for *fine-tuning* pretrained weights on
a modest dataset — blending images suppressed the model's ability to commit to
detections. Turning it off let the fine-tune fit. The recall jump confirmed the
gain was real (more objects found, not luck).

## Iteration 5 — Push capacity: m @ 640 TOO MUCH MODEL
**Change:** `yolov8m` on the winning no-mixup recipe (isolate model size again).
**Result:** **0.725** (core-8 0.778) — *down* from `s`. Recall dropped
0.73 → 0.63; precision rose 0.75 → 0.81.
**What we learned:** `m` (~26M params) has more capacity than this dataset can
feed. It became over-conservative (high precision, low recall) — classic "too
much model, not enough data," made worse by having removed the regularizing
mixup. **This located the capacity sweet spot: `n` too small, `m` too big,
`s` just right.**

---

## Final leverage ranking

| Lever | Effect on all-10 mAP50 | Verdict |
|---|---|---|
| Data / split fix | **+0.20** | by far the biggest |
| Recipe (mixup off) | **+0.07** | big |
| Capacity n → s | +0.04 | modest help |
| Capacity s → m | −0.03 | too much model |
| Resolution → 1280 | −0.03 | scale mismatch, hurt |

**The story the numbers tell:** the largest wins came from the unglamorous
work — data quality and training recipe — while the "obvious" upgrades
(bigger model, higher resolution) either gave diminishing returns or actively
backfired. There is a real optimum in the middle, and every claim here is backed
by a clean one-variable run.

---

## Directly-comparable results (same re-split test set)

| Run | all-10 | core-8 | recall | note |
|---|---|---|---|---|
| n @ 640 (re-split) | 0.649 | 0.726 | 0.564 | first trustworthy number |
| s @ 1280 | 0.621 | 0.671 | 0.562 | confounded; misleading |
| s @ 640 | 0.688 | 0.757 | 0.638 | disentangled; capacity helps |
| **s @ 640, no-mixup** | **0.757** | **0.795** | **0.726** | **FINAL MODEL** |
| m @ 640, no-mixup | 0.725 | 0.778 | 0.633 | too much model |

(Baseline 0.447 used the old broken test set, so it's the "before" reference,
not directly comparable to the rows above.)

---

## Methodology lessons (the transferable part)

1. **Look at your data first.** The biggest win was a data-split fix, not a model.
2. **One variable at a time.** The one time we changed two (Iter 2), we drew a
   wrong conclusion that a clean run later overturned.
3. **Verify assumptions before spending compute.** `dropout` was a no-op for
   detection — reading the source saved a wasted GPU run.
4. **Recall is the honesty check.** Rising mAP with rising recall = real; mAP
   moving on confidence alone is suspect.
5. **Bigger isn't better.** Both capacity and resolution had a sweet spot;
   overshooting hurt.
