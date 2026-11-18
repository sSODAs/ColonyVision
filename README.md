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
