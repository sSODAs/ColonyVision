# ColonyVision

Final Computer Vision pipeline for detecting and counting bacterial colonies on Petri-dish photographs.

This repository is intentionally reduced to the **final inference/evaluation system**. Development-only ADBC datasets, teacher-model tools, pseudo-label datasets, diagnostics, and historical experiment outputs are removed.

## What is included

- Final ColonyVision source code.
- Final student model checkpoint.
- Frozen Test50 set: **50 photographs + manual point Ground Truth**.
- Final pipeline configuration.
- Reproducible Test50 evaluation runner.
- Git safety checks.

Large binary files (the student weight and Test50 photographs) use **Git LFS**.

## Final pipeline

1. Detect Petri-dish geometry.
2. Show the crop boundary and crop the dish.
3. Define the orange counting ROI.
4. Run tiled YOLO inference on original and CLAHE-enhanced views.
5. Refine/filter candidate detections.
6. Resolve touching colonies with lobe-graph evidence.
7. Apply seeded-connected colony resolution.
8. Run a conservative iterative parent/child nested-box re-check.
9. Render the final cropped plate with rectangular colony boxes and small numbered AST-style tags.
10. Calculate Test50 metrics from manual point annotations.

## Final Test50 result

| Method | Precision | Recall | F1 | Count MAE |
|---|---:|---:|---:|---:|
| YOLO baseline | 80.00% | 95.53% | 87.08% | 14.18 |
| Refined | 92.73% | 96.08% | 94.38% | 3.80 |
| **Final iterative-resolved** | **97.07%** | **94.52%** | **95.78%** | **3.00** |

Final aggregate: **TP 3,450 / FP 104 / FN 200** on 3,650 GT colonies inside the evaluation ROI.

## Repository structure

```text
ColonyVision/
├─ config/
│  └─ final_pipeline.json
├─ evaluation/
│  └─ test50/
│     ├─ selection.csv
│     ├─ ground_truth_points.csv
│     ├─ ground_truth_summary.csv
│     ├─ reference_metrics.csv
│     └─ README.md
├─ raw/
│  └─ test50/                     # exactly 50 frozen test photographs (Git LFS)
├─ scripts/
│  ├─ run_test50.py
│  └─ check_git_payload.py
├─ src/
│  ├─ final_pipeline.py
│  ├─ predict_tiled.py
│  ├─ prepare_plate_images.py
│  ├─ colony_evidence.py
│  ├─ lobe_graph_resolver.py
│  ├─ seeded_connected_resolver.py
│  └─ preprocessing/
│     ├─ plate_hough.py
│     └─ plate_roi.py
├─ weights/
│  └─ colonyvision_student_v1.pt  # Git LFS
├─ output/                         # generated locally; ignored by Git
├─ requirements.txt
└─ README.md
```

The large historical `resolve_e3.py` module was removed. The small subset of image-evidence functions still required by the final pipeline now lives in `src/colony_evidence.py`.

## Setup

Python 3.11 is recommended.

```powershell
git clone https://github.com/sSODAs/ColonyVision.git
cd ColonyVision

git lfs install
git lfs pull

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Test50

The repository already contains the 50 frozen Test50 photographs under `raw/test50/`.

Run:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_test50.py
```

Results are generated locally in:

```text
output/Test50_Final/
```

Each `Txxx` folder contains:

- `Txxx_01_final_result.jpg` — cropped dish, orange ROI, rectangular detections, small numbered black tags.
- `Txxx_02_steps.jpg` — AST-style white-background step-by-step pipeline sheet.
- `Txxx_03_pipeline_graph.png` — detection counts across pipeline stages.
- `Txxx_04_measured.json` — final colony coordinates and count.
- `Txxx_debug/` — individual stage images.

The generated `output/` directory is intentionally **not uploaded to GitHub**.

## Test50 annotations

`evaluation/test50/ground_truth_points.csv` contains the manually clicked point Ground Truth.

`evaluation/test50/selection.csv` maps T001–T050 to the 50 photographs stored in `raw/test50/`.

## Git policy

Published:

- final source code,
- final configuration,
- Test50 metadata/annotations,
- exactly 50 Test50 photographs through Git LFS,
- final student model through Git LFS.

Not published:

- generated `output/`,
- the remaining private/raw development photographs,
- ADBC training data,
- teacher model/checkpoints,
- pseudo-label datasets,
- diagnostics and historical experiments,
- local virtual environments and caches.

Audit before commit/push:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_git_payload.py
```
