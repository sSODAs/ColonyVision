# ColonyVision

**Scope (Phase 1):** locate a Petri dish, detect/count colonies as **one class**, and evaluate the predicted number per dish. Species classification is OUT OF SCOPE.

## Local project status (2026-11-18)

- 1,252 JPG photos (~2.11 GB) imported into `raw/unlabeled/Petri_plates/` without conversion.
- No per-colony bounding-box labels have been verified or assigned to a training split.
- The original filenames may contain colony-count tokens. These are **unverified**, and counts >= 300 may be truncated/TNTC.
- Audited source filenames: `python src/inventory_dataset.py`
- Initial plate locator: `src/preprocessing/plate_roi.py` (Pillow + NumPy; no OpenCV). It produces PREVIEWS only by default.
- Current YOLO scripts are `src/train.py`, `src/predict.py`, `src/validate_dataset.py`. DO NOT TRAIN until verified YOLO boxes exist.

## Commands (PowerShell, from project root)

```powershell
.\.venv\Scripts\Activate.ps1

# Inventory source photos and parse potential count tokens
python .\src\inventory_dataset.py

# Preview overlays and plate crops for 12 distributed samples
python .\src\preprocessing\plate_roi.py --limit 12

# Later, once checked and approved, preview more:
python .\src\preprocessing\plate_roi.py --limit 80

# Only after labels are prepared:
python .\src\validate_dataset.py
python .\src\train.py --epochs 50 --imgsz 512 --batch 2 --name baseline
```

## Annotation pipeline

1. Keep the 1,252 original photos untouched under `raw/unlabeled/Petri_plates`.
2. Try an existing colony model on ~10-20 representative images first.
3. Export individual **bounding boxes in original pixel coordinates**, not just per-image counts.
4. Save output as *candidate* pseudo-labels under `annotations/pseudo_labels/`.
5. Convert external CSV boxes using `python src/annotation/import_boxes_csv.py --csv <external_boxes.csv>`. For boxes predicted on plate crops, add `--crop-metadata outputs/plate_roi_preview/plate_roi_metadata.csv` to map coordinates back to original images.
6. Human-review missed colonies, duplicates, artifacts, confluency, and colony size.
7. Put only reviewed box labels and corresponding images into grouped train/val/test splits.
8. Keep photos of the same original Petri dish/session together in the same split.

**Recommended model probes:**
- WeCFU: convenient human-in-the-loop marker/counter and review tool; marker circles are not measured bounding boxes.
- Colony Grounded SAM2: colony-focused detection/segmentation, with downloadable weights; heavier pipeline better suited to a separate Colab/WSL environment.
- Classical image-analysis baseline as comparison; no claims of accurate ground truth without review.

## Files and artifact policy

Do not commit original photos, `.venv`, weights, annotations, or results to GitHub by default. See `.gitignore`.

Windows Application Control allowed Torch 2.10.0 CUDA 12.8 but blocked OpenCV DLL import in the last test. Plate ROI preview deliberately avoids OpenCV.

## Pilot annotation batch (2026-11-18)

- `src/annotation/select_pilot.py` selects 24 images reproducibly across filename species hints and count-hint ranges (8 per hint). All filenames/count hints are **unverified**.
- `annotations/metadata/pilot_24_manifest.csv` records selected filenames and review status; private and gitignored.
- `raw/pilot_review/` holds **copies** of 24 original photos for external labeling; all 1,252 originals remain unchanged at `raw/unlabeled/Petri_plates/`.
- `outputs/pilot_plate_roi/overlays/` and `outputs/pilot_plate_roi/crops/` hold plate localization **previews**, not validated crops.
- `outputs/pilot_plate_roi/plate_roi_metadata.csv` contains candidate circle and crop offsets for mapping predictions to the original images. Status `estimated_not_verified` means visually inspect before proceeding.

### Suggested external annotation trial
1. Start with WeCFU (https://github.com/WeStrainGroup/WeCFU): count/marker suggestions with manual correction; export its per-image JSON and CSV. Marker coordinates are **points, not measured bounding boxes**, so do not silently convert them to ground-truth boxes.
2. Alternative investigation: Colony Grounded SAM2 (https://huggingface.co/DataScienceWFSR/ColonyGroundedSAM2); check original repository, runtime needs, and license of training data before using its checkpoint.
3. Do not train on predictions until individual colonies, fused colonies, plate-edge reflections, and missed detections are reviewed. Save external output in `annotations/pseudo_labels/` and final inspected labels in `annotations/reviewed/`.
4. Dataset images and labels in `dataset/` remain empty until reviewed labels are ready. Group repeated photos of the same plate/session in one split.

## Automated WeCFU Ultra pilot (2026-11-18)

Use the **separate installed** WeCFU 1.2.4 server on http://127.0.0.1:8765.
The server must have `Ultra is ready` and uses CPU on Windows.

```powershell
cd C:\Users\user\Desktop\ColonyVision
.\.venv\Scripts\python.exe .\src\annotation\prepare_wecfu_inputs.py
.\.venv\Scripts\python.exe .\src\annotation\wecfu_ultra_batch.py --source raw\wecfu_ready --batch colonyvision_ultra_upright
```

**Do not run the second command again for this pilot unless intentionally continuing the trial:** upright copies were already generated. The preparation script refuses to overwrite existing copies, and the batch script skips already counted images. To test a single image, specify `--limit 1`.

Inputs: `raw/wecfu_ready/` (24 EXIF-normalized derived JPEGs; original 1,252 files untouched).
Output: `annotations/pseudo_labels/wecfu_ultra/wecfu_ultra_export.zip` and `ultra_run_report.csv`.
WeCFU saved-state workspace: `annotations/wecfu_workspace/`.

**Probe findings, not ground truth:** WeCFU Ultra returned 0 markers on
`IMG_7741ecoli_T6_10^-7_291.jpg`, and 1 marker on
`IMG_8684ecoli_dplaca_10^-6_42.jpg`. The filename count tokens are unverified;
however, the images visibly show more colonies than the predictions.
Classical CLI returned 0 for the latter image.
Therefore the WeCFU method is **not admitted for bulk pseudo-label creation** at present.
Do not place these candidate markers in dataset/labels/train, val or test.
Review overlay quality and evaluate other pre-annotation methods first.

## Human-approved YOLO annotation workflow

WeCFU markers are **candidate centers** and the imported radius is only a visual radius.
The 24 pilot images in `raw/wecfu_ready/` have been EXIF-normalized; do not use the original
unrotated JPEG coordinates to interpret them.

Open the review GUI (Windows):

```powershell
.\.venv\Scripts\python.exe .\src\annotation\review_colonies.py
```

Or double-click `run_colony_review.cmd`.

- `Add`: click to add a box using the chosen radius, or drag a rectangle manually.
- `Move`: drag an existing rectangle.
- `Resize`: drag from an existing rectangle toward its nearest corner.
- `Delete`: click an existing box; right-click deletes regardless of selected mode.
- Zoom using buttons/mouse wheel, scrollbars for panning.
- `Save draft`: preserves work under `annotations/review_drafts/`; NEVER used for training.
- `Approve reviewed`: **only after human visual verification**, writes `annotations/reviewed/colony/{image}.json` and `{image}.txt` (YOLO format). Approving a zero-colony plate is allowed only when verified.
- A newer draft invalidates any older approval until the user approves again.
- All original photos remain untouched. Generated/derived photos and labels are gitignored.

After at least some images are fully reviewed, initialize the split plan:

```powershell
.\.venv\Scripts\python.exe .\src\annotation\build_reviewed_dataset.py --init-plan
```

Edit `annotations/metadata/reviewed_splits.csv`, assigning `split` (train/val/test)
and `plate_group` (same ID for photos from the same physical Petri dish/session).
Run `--init-plan` again whenever new reviewed photos are added; existing entries stay.
Once there are reviewed images in **both train and val**, build the dataset:

```powershell
.\.venv\Scripts\python.exe .\src\annotation\build_reviewed_dataset.py --build
.\.venv\Scripts\python.exe .\src\validate_dataset.py
```

The builder validates source image hashes, YOLO box contents, manual approval and
plate-group split isolation. It refuses to overwrite conflicting dataset files.
Do not train until there is sufficient verified data for meaningful validation.

## Visual preview for existing WeCFU Ultra results

The existing `wecfu_ultra_export.zip` already contains per-image JSON marker coordinates and count values. To inspect without rerunning inference:

```powershell
cd C:\Users\user\Desktop\ColonyVision
.\.venv\Scripts\python.exe src\annotation\preview_wecfu.py
start .\outputs\wecfu_ultra_preview\index.html
```

Or double-click `open_wecfu_preview.cmd` to regenerate and open the HTML gallery.

Generated artifacts (24 images):
- `outputs/wecfu_ultra_preview/index.html` -- clickable image gallery with counts
- `outputs/wecfu_ultra_preview/overview.jpg` -- single contact sheet of all 24 images
- `outputs/wecfu_ultra_preview/overlays/*.jpg` -- full-resolution images with exact WeCFU circle/center markers
- `outputs/wecfu_ultra_preview/comparisons/*.jpg` -- original next to marked image
- `outputs/wecfu_ultra_preview/preview_counts.csv` -- count table

**Important:** yellow circles show model-provided center and radius; they are not human-verified bounding boxes. Their absence on a visible colony is a false negative candidate. The filename count token is only a hint, not validated ground truth. This visualization does not change inputs, annotations or model results.
