# Military Vehicle Detection (YOLOv8)

Object detection model for identifying military vehicles, people, and structures
in aerial and ground imagery, built on Ultralytics YOLOv8.

> **Status:** work in progress. Dataset preparation is complete; model training
> and evaluation are in progress. Results below will be filled in once training
> on the cleaned dataset finishes.

---

## Dataset

The data comes from a [Roboflow](https://roboflow.com) project exported in YOLOv8
format. The raw export shipped with a **contaminated 16-class taxonomy** — a
telltale sign that it had been assembled by merging several separately-labeled
sources without harmonizing their class maps.

Problems found during the audit:

- **Numeric classes** `0`–`4` with no human-readable names, sitting alongside
  properly-named classes.
- **A case-duplicate**: both `Person` and `person` as separate classes for the
  same object.
- **A generic `Vehicle`** class overlapping the specific vehicle types.

Inspecting sample annotations showed the numeric classes were *duplicate
labelings* of vehicle types that already had proper names (e.g. `1` → tanks,
`2` → trucks, `0`/`3`/`4` → armored vehicles).

The taxonomy was then **harmonized from 16 classes down to 10, with zero
annotations lost** (22,691 boxes preserved) using a reproducible, config-driven
script (`src/clean_taxonomy.py`) that writes a cleaned copy and leaves the raw
download untouched.

### Final classes (10)

| Class | Instances | | Class | Instances |
|---|---:|---|---|---:|
| tank | 6,855 | | rszo (rocket artillery) | 767 |
| person | 5,083 | | sau (self-propelled artillery) | 479 |
| truck | 4,206 | | plane | 297 |
| armored_car | 3,306 | | vehicle (generic) | 196 |
| car | 1,451 | | trench | 51 |

---

## Project structure

```
military-vehicle-detection/
├── README.md
├── .gitignore
├── .env                 # local only — Roboflow API key (never committed)
├── configs/             # training hyperparameters (YAML)
├── data/                # datasets (gitignored)
│   ├── dataset/         # raw Roboflow download
│   └── dataset_clean/   # harmonized output
└── src/
    ├── download_dataset.py   # pull dataset from Roboflow
    ├── audit_dataset.py      # class counts, flags, sample images
    └── clean_taxonomy.py     # 16 -> 10 class harmonization
```

Code lives in `src/`, data in `data/`, configuration in `configs/`. All scripts
are location-independent — they resolve paths from the project root, so they run
correctly from any working directory.

---

## Reproduce

```bash
# 1. Clone and enter
git clone https://github.com/mmoz-root/military-vehicle-detection.git
cd military-vehicle-detection

# 2. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install roboflow opencv-python pyyaml pandas ultralytics

# 3. Credentials — create a .env file in the project root:
echo "ROBOFLOW_API_KEY=your_key_here" > .env

# 4. Build the dataset
python src/download_dataset.py        # -> data/dataset/
python src/clean_taxonomy.py          # -> data/dataset_clean/
python src/audit_dataset.py           # verify: 10 clean classes, no flags
```

---

## Results

_Training on the cleaned dataset is in progress. Metrics (mAP50, mAP50-95,
per-class breakdown) and sample predictions will be added here._

An early baseline trained on the **raw, uncleaned** data reached ~58% mAP50;
much of that ceiling was an artifact of the broken taxonomy rather than model
capacity, which is what the dataset cleanup above addresses.

---

## Known limitations

- **`trench`** has only 51 instances total — too few for the model to learn or
  be evaluated on reliably. Kept for completeness, but expect weak performance.
- **`plane`** has no instances in the test split, so it can't be evaluated on
  held-out data.
- **Video-frame artifacts:** some images are frames captured from surveillance/
  drone footage and contain player overlays or timestamps baked into the pixels.

## Results

Best model: yolov8n @ 640, re-split data. Reported at two scopes:

| Scope | mAP50 |
|---|---|
| All 10 classes | 0.649 |
| Core 8 classes (excl. trench, vehicle) | 0.726 |

Two classes are excluded from the "core" scope, with justification:

- **trench** — only 51 instances total; insufficient to learn or evaluate reliably.
- **vehicle** — a generic catch-all that overlaps the specific vehicle classes by definition, so it is inherently ambiguous.

Both are kept in the dataset and trained on (not deleted); they are excluded only from the core-scope headline metric, which reflects the classes with adequate, well-defined data.
