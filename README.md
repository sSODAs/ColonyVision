# ColonyVision — Automated Petri Dish Colony Detection & Counting

**Phase 1 scope:** identify colony positions on petri-dish photographs and count all visible colonies as ONE generic class (`colony`). Species recognition is not part of this baseline.

## Input images

- 1,252 original images: `raw/unlabeled/Petri_plates/`, **unaltered and not used for training** because they lack reviewed per-colony bounding boxes.
- Capture variations, background, glare, colony size and crowding can cause domain shift between training and these photographs.
- The count-like number in a source filename is **not independently verified** and might be capped.
- Dish localization baseline: `src/preprocessing/plate_roi.py`; that tool creates estimates/previews, not confirmed plate geometry.

## Training data — ADBC (not a pretrained model)

The external training set is **Annotated dataset for deep-learning-based bacterial colony detection**:
https://doi.org/10.6084/m9.figshare.22022540.v3 (Figshare, CC BY 4.0; attribution required).

Official release: **369 photos, 24 class IDs, 56,865 source YOLO boxes.**
Three official boxes have zero height; the preparation script warns and excludes
those three invalid entries. Original Figshare images and labels remain unchanged
under `raw/external/ADBC/`.

`src/dataset/prepare_adbc.py`:
- Fetches official images/labels and checks size + MD5 against Figshare metadata.
- Applies EXIF orientation correction to displayed images before interpreting YOLO coordinates.
- Converts every original species class to class `0 colony`.
- Crops into **768×768 image tiles**, stride **640**, with aligned clipped per-tile labels.
- Assigns **all photographs from the same species group to the same split** to avoid data leakage; this is an intentionally challenging cross-species evaluation split.
- Creates separate `dataset/adbc_pilot/` and `dataset/adbc_full/` with YOLO `data.yaml`, train/val/test images and labels, and a source manifest.

A checkpoint trained on ADBC is **our own YOLO11n fine-tune**, not an ADBC-provided model.

## Commands (Windows PowerShell)

Run these from `C:\Users\user\Desktop\ColonyVision`.

```powershell
# Optional: regenerate a small 24-photo preparation/smoke test
.\.venv\Scripts\python.exe src\dataset\prepare_adbc.py --name adbc_pilot --max-per-species 1

# Full labeled training set: 369 official images
.\.venv\Scripts\python.exe src\dataset\prepare_adbc.py --name adbc_full --max-per-species 0

# GPU training on RTX 3050 (4 GB); ADBC labels only
.\.venv\Scripts\python.exe src\train.py --data dataset\adbc_full\data.yaml --model yolo11n.pt --epochs 40 --imgsz 640 --batch 2 --name adbc_yolo11n

# Counting previews on 12 original private photographs using tiled inference + NMS
.\.venv\Scripts\python.exe src\predict_tiled.py --model outputs\train\adbc_yolo11n\weights\best.pt --limit 12

# If results are sufficiently plausible, run prediction for all input photos
.\.venv\Scripts\python.exe src\predict_tiled.py --model outputs\train\adbc_yolo11n\weights\best.pt --limit 0
```

**Note:** ADBC full training is a nontrivial GPU run and takes substantially longer than the 1-epoch pilot check; do not interpret a smoke-test checkpoint as an accurate finished model.

Tiled inference outputs `outputs/predict/adbc_tiled_preview/` with:
- `counts.csv` — image and predicted count (not ground truth)
- `overlays/*_marked.jpg` — boxes drawn on EXIF-upright images
- `detections/*.json` — each predicted bounding box and its confidence

## How to assess accuracy

- Use ADBC held-out labels to report precision, recall and mAP, with the actual training split specified.
- For **real ColonyVision photographs**, evaluate against independent reviewed counts/boxes from a sample of physical plates before making accuracy or CFU measurement claims.
- Because tiles overlap, cross-tile NMS is required at inference to reduce double-counting.
- No source-photo filename token or pseudo-label is treated as independent ground truth.

## Project hygiene

- Original photos, external data, generated labels, checkpoints and predictions are ignored by Git.
- WeCFU was removed from the project, its stand-alone Desktop installation and its downloaded runtime. No WeCFU predictions are included in the ADBC training pipeline.
- See `raw/external/ADBC/source_metadata.json` and `dataset/adbc_full/SOURCE_AND_LICENSE.txt` for provenance.

## Experimental Circular ROI training (2026-11-18)

**Keep the in-progress ADBC baseline training intact.** Do not modify its tiles
(`dataset/adbc_full`) or restart its active training process.

Rationale: only plate interior should influence colony detection; avoid shadows,
glare, background and table objects outside the Petri dish.

- ADBC image-only ROI research audit:
  `src/dataset/audit_circular_roi.py` checks 369 images and reports to
  `outputs/roi_circle_audit/audit.csv`.
  The old brightness-threshold estimator returned a circle for 154/369 images.
  OpenCV Hough returned circles for 369/369 images **as hypotheses**, not
  manually verified geometries. At radius 1.02 times Hough radius,
  60/56,865 official annotation centers were outside on 6 source images.
- New `src/preprocessing/plate_hough.py` estimates candidate circles with
  OpenCV Hough on image pixels alone. It is tested on ADBC images, **not yet
  reliable on all 1,252 user images**; one real phone photo did not yield a circle.
- `src/dataset/build_adbc_circular.py` generates **new training tiles only**
  into `dataset/adbc_circular/` without changing the baseline. It masks the
  exterior of confidently estimated circles to black with a 3.5% radial
  safety margin; image and label coordinates remain unchanged. If the circle
  cannot be found or could cover any official GT box corner, that training
  image remains unmasked, and its fallback reason is recorded in
  `roi_audit.csv`. This avoids silently deleting true annotated colonies.
  The Validation and Test paths continue to use the same **unmodified**
  ADBC baseline data for fair cross-run comparison.
- The current prediction script `src/predict_tiled.py` can use
  `--circular-mask --strict-roi` to mask the exterior before YOLO, and skip
  instead of reporting count=0 if plate localization fails. Its real-photo
  plate estimator still uses the previous brightness-based algorithm,
  which must be evaluated separately.

Commands for a **separate experimental run after baseline finishes**:

```powershell
cd "C:\Users\user\Desktop\ColonyVision"

# Prepare circular ROI training data (separate dataset, safe to regenerate under a new name)
.\.venv\Scripts\python.exe src\dataset\build_adbc_circular.py --name adbc_circular

# Compare to the same validation and test splits (run separately from existing training)
.\.venv\Scripts\python.exe src\train.py --data dataset\adbc_circular\data.yaml --epochs 40 --imgsz 640 --batch 2 --name adbc_yolo11n_roi

# Infer using ROI mask; preview 12 images, never promotes predictions to ground truth
.\.venv\Scripts\python.exe src\predict_tiled.py --model outputs\train\adbc_yolo11n_roi\weights\best.pt --circular-mask --strict-roi --limit 12 --name adbc_roi_preview
```

Note: The ROI experiment may increase or decrease accuracy. Compare Precision,
Recall, mAP and actual counting error on fixed, independently annotated test
images before treating the hypothesis as successful. Inference localization
for real patient/user photos remains to be independently verified.
