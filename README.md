# ColonyVision — Automated Bacterial Colony Detection and Counting

<img src="raw/test50/T034__IMG_8278saureus_dplaca_10^-6_62.JPG" width="350" alt="Example Petri-dish photograph from Test50">

ColonyVision is a **Computer Vision project** for detecting and counting bacterial colonies in Petri-dish photographs using **YOLO11n and image processing**.

Instead of directly counting the first bounding boxes returned by YOLO, the system refines candidates, checks duplicate detections, and examines touching colonies before reporting the final count.

This project was developed as part of a university Computer Vision course. It focuses on **colony detection and counting**, not bacterial species identification or clinical diagnosis.

## Overview

A plate may contain only a few colonies or hundreds of small, closely packed colonies. Uneven lighting, reflections, overlapping detections, and touching colonies make individual counting difficult.

**What this project does:**

- Locates the Petri dish, crops the image, and defines a circular counting ROI.
- Enhances contrast with CLAHE.
- Uses tiled YOLO11n inference to retain detail on small colonies.
- Combines and filters candidate bounding boxes.
- Resolves ambiguous touching colonies using Lobe Graph and Seeded-Connected image evidence.
- Re-checks nested boxes before producing the final count.
- Draws rectangular boxes and small numbered tags around detected colonies.
- Evaluates results using manually marked points on **50 held-out test images**.

The Final image displays `(TNTC)` when the detected count exceeds 300. This is only a display indicator; it is **not a validated microbiological TNTC determination**.

## How It Works

```text
Petri-dish photograph
         |
         v
Plate detection + crop + counting ROI
         |
         v
Original / CLAHE / local-enhancement views
         |
         v
Tiled YOLO11n inference + NMS
         |
         v
Candidate refinement and filtering
         |
         v
Lobe Graph / touching-colony analysis
         |
         v
Seeded-Connected resolution
         |
         v
Iterative nested-box re-check
         |
         v
Final bounding boxes + colony count
```

**Main inference configuration**

| Parameter | Value |
|---|---:|
| Object detector | Fine-tuned YOLO11n Student |
| Detection classes | 1 (`colony`) |
| Tile size | 768 × 768 px |
| Tile stride / overlap | 640 px / 128 px |
| YOLO input | 640 × 640 px |
| Confidence threshold | 0.25 |
| Global NMS IoU | 0.60 |
| Counting ROI | 0.93 × estimated plate radius |
| CLAHE clip limit / grid | 2.0 / 8 × 8 |

Settings are defined in `config/config.json`. The plate boundary is estimated from image evidence and is not a verified physical measurement.

## Dataset and Model Training

The original development collection contained **1,200 Petri-dish images**:

| Split | Images | Purpose |
|---|---:|---|
| Train | 900 | YOLO11n Student training |
| Validation | 100 | Model validation |
| Test | 200 | Held-out image evaluation pool |

The photographs initially lacked complete human-verified bounding-box labels. A previously developed **ADBC-based Teacher detector** was used to propose **pseudo-label bounding boxes**. After filtering and image tiling, those pseudo-labels were used to fine-tune a pretrained **YOLO11n Student** model for one class: `colony`.

The original training configuration targeted 50 epochs; the available training log records results through epoch 45. Pseudo-labels are **training references, not independent Ground Truth**.

This final repository includes the Student checkpoint and frozen evaluation data, **not** the full original dataset, Teacher checkpoint, training tiles, or development experiments.

## Manual Ground Truth and Evaluation

For Test50, one point was manually placed near the center of each visible colony. Only points inside the counting ROI are included in the primary evaluation.

| Evaluation data | Count |
|---|---:|
| Test photographs | 50 |
| Total manually marked points | 3,698 |
| Points inside counting ROI | 3,650 |
| Points outside counting ROI | 48 |

Detection evaluation uses **one-to-one matching** between predicted bounding-box centers and manual points. The primary matching tolerance is **0.5 × the per-plate median baseline colony diameter**. Precision, Recall, and F1 are therefore **point-based detection measures**, not classification accuracy or bounding-box mAP.

For colony counts, **MAE** measures the average absolute count error per plate, while **RMSE** penalizes large count errors more heavily.

### Final Test50 Results

| Method | Precision | Recall | F1-score | Count MAE |
|---|---:|---:|---:|---:|
| YOLO Baseline | 80.00% | 95.53% | 87.08% | 14.18 |
| Refined | 92.73% | **96.08%** | 94.38% | 3.80 |
| **Final Iterative-Resolved** | **97.07%** | 94.52% | **95.78%** | **3.00** |

Additional Final measurements:

- **Count RMSE:** 6.03 colonies/plate
- **True Positives:** 3,450
- **False Positives:** 104
- **False Negatives:** 200
- **Predicted colonies in ROI:** 3,554
- **Ground Truth colonies in ROI:** 3,650

The complete pipeline increases F1 from **87.08% to 95.78%** and reduces Count MAE from **14.18 to 3.00 colonies/plate** compared with the YOLO Baseline. However, Final Recall is slightly lower than Refined Recall, showing a precision–recall trade-off.

Results are frozen in `evaluation/test50/reference_metrics.csv`.

## Technologies Used

- Python 3.11 (recommended)
- Ultralytics YOLO11n and PyTorch
- OpenCV
- NumPy
- Pillow
- Matplotlib
- Git LFS for the final model weight and Test50 photographs

## Project Structure

```text
ColonyVision/
+-- config/config.json
+-- evaluation/test50/             # Manual point GT, manifests, reference metrics
+-- raw/test50/                    # 50 frozen photographs (Git LFS)
+-- src/
|   +-- pipeline.py              # Test50 inference/evaluation runner
|   +-- detection.py             # Tiled YOLO and refinement
|   +-- evaluation.py            # Point matching, metrics, graphs
|   +-- visualization.py         # Final and intermediate images
|   +-- preprocessing/plate.py   # Plate detection, crop, CLAHE
|   +-- postprocessing/
|       +-- evidence.py
|       +-- touching.py
|       +-- connected.py
|       +-- nested.py
+-- weights/colonyvision_student_v1.pt  # Final Student (Git LFS)
+-- run.py
+-- requirements.txt
+-- README.md
```

Generated output is written under `output/` and is not committed to Git.

## Setup and Installation

**Requirements**
- Python 3.11 recommended
- Git and Git LFS
- Windows PowerShell for the commands below
- CUDA-capable GPU optional; CPU inference is supported

**1. Clone and download LFS files**

```powershell
git clone https://github.com/sSODAs/ColonyVision.git
cd ColonyVision
git lfs install
git lfs pull
```

**2. Create a virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Project

To run the frozen **Test50** evaluation:

```powershell
python run.py
```

For a small test in a separate output folder:

```powershell
python run.py --limit 5 --output output/Test5_Preview
```

**Important:** The runner **deletes and recreates the selected output directory** before every run. Do not point `--output` to a directory containing files you need to keep.

The included command is a **batch inference/evaluation runner for Test50**, not a standalone GUI or an arbitrary image-upload interface.

### Generated Files

```text
output/Test50_Final/
+-- T001/
|   +-- T001_01_final_result.jpg      # Numbered boxes and final count
|   +-- T001_02_steps.jpg            # White-background pipeline montage
|   +-- T001_03_pipeline_graph.png   # Counts by pipeline stage
|   +-- T001_04_measured.json        # Final detected coordinates
|   +-- T001_debug/                  # Individual processing stages
+-- ... T050/
+-- _evaluation/
    +-- aggregate_metrics.csv
    +-- metrics_by_density.csv
    +-- per_image_metrics.csv
    +-- per_plate_summary.csv
    +-- SUMMARY.txt
```

## Limitations and Notes

- **Dense/touching colonies:** Closely packed or confluent growth can cause missed colonies and under-counting.
- **Training labels:** Teacher-generated pseudo-labels can contain errors.
- **Ground Truth:** Test50 uses one human annotator; inter-annotator agreement has not been established.
- **Data leakage:** The original split was image-level; independent physical-plate separation has not been confirmed.
- **Evaluation reuse:** Test50 was used during development and evaluation. New changes should be validated against a fresh independent test set.
- **Scope:** No bacterial species classification, CFU/mL estimate, or diagnostic interpretation is provided.

## Data Sources and Repository Policy

- Original Petri-dish photographs: [Figshare dataset](https://doi.org/10.6084/m9.figshare.20109377).
- Earlier ADBC Teacher development: [Annotated Dataset for Deep-learning-based Colony Detection](https://doi.org/10.6084/m9.figshare.22022540.v3).
- `evaluation/test50/` contains reproducible reference metadata and point annotations.
- The frozen Test50 photographs and Student model are stored using Git LFS.
- Other raw/training images, generated outputs, development diagnostics, local Colab files (`colab_demo/`), and notebooks are not published.

This repository is for academic demonstration and reproducible evaluation, not validated laboratory or medical use.
