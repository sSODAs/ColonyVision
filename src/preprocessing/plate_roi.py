"""Conservative plate ROI baseline using Pillow and NumPy, no OpenCV dependency.

Creates preview overlays/crops and metadata. Never modifies original files.
Images that fail geometry sanity checks are reported, not silently cropped.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "raw" / "unlabeled" / "Petri_plates"
DEFAULT_OUTPUT = ROOT / "outputs" / "plate_roi_preview"
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def estimate_plate(image):
    thumb = image.copy()
    thumb.thumbnail((512, 512), Image.Resampling.BILINEAR)
    gray = np.asarray(thumb.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    border = np.concatenate((
        gray[: max(3, h // 20), :].ravel(),
        gray[-max(3, h // 20):, :].ravel(),
        gray[:, : max(3, w // 20)].ravel(),
        gray[:, -max(3, w // 20):].ravel(),
    ))
    threshold = max(35, int(np.median(border)) + 25)
    ys, xs = np.where(gray >= threshold)
    coverage = len(xs) / float(w * h)
    if not (0.12 <= coverage <= 0.78):
        return None, f"invalid_coverage_{coverage:.3f}"
    xlo, xhi = np.percentile(xs, [1, 99])
    ylo, yhi = np.percentile(ys, [1, 99])
    width = xhi - xlo
    height = yhi - ylo
    if not (0.75 <= width / max(1, height) <= 1.33):
        return None, "bad_aspect_ratio"
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    radius = (width + height) / 4
    if radius < 30:
        return None, "radius_too_small"
    scale_x, scale_y = image.width / w, image.height / h
    # The short side of a mildly elliptical plate keeps ROI conservative.
    center_x, center_y = cx * scale_x, cy * scale_y
    radius_original = radius * (scale_x + scale_y) / 2
    result = (center_x, center_y, radius_original, threshold, coverage)
    return result, "estimated_not_verified"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=12,
                        help="Default: only 12 images for review; 0 means all")
    parser.add_argument("--margin", type=float, default=1.06)
    args = parser.parse_args()

    paths = sorted(p for p in args.source.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS)
    if args.limit > 0:
        # Spread previews across the filename-sorted dataset.
        step = max(1, len(paths) // args.limit)
        paths = paths[::step][:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "overlays").mkdir(exist_ok=True)
    (args.out / "crops").mkdir(exist_ok=True)
    rows = []
    for index, path in enumerate(paths, 1):
        try:
            with Image.open(path) as f:
                image = ImageOps.exif_transpose(f).convert("RGB")
            candidate, status = estimate_plate(image)
            row = {"image": path.name, "status": status,
                   "cx": "", "cy": "", "radius": "",
                   "crop_left": "", "crop_top": "",
                   "crop_right": "", "crop_bottom": ""}
            if candidate:
                cx, cy, r, t, cov = candidate
                # Keep original pixel coordinates in metadata.
                crop_r = r * args.margin
                x0, y0 = max(0, int(cx-crop_r)), max(0, int(cy-crop_r))
                x1, y1 = min(image.width, int(cx+crop_r)), min(image.height, int(cy+crop_r))
                row.update({"cx": round(cx, 1), "cy": round(cy, 1),
                            "radius": round(r, 1), "crop_left": x0, "crop_top": y0,
                            "crop_right": x1, "crop_bottom": y1})
                preview = image.copy()
                draw = ImageDraw.Draw(preview)
                draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline="#ff3434", width=7)
                draw.rectangle((x0,y0,x1,y1), outline="#00ff66", width=5)
                preview.thumbnail((768,768))
                preview.save(args.out / "overlays" / f"{path.stem}.jpg", quality=82)
                image.crop((x0,y0,x1,y1)).save(
                    args.out / "crops" / f"{path.stem}.jpg", quality=90)
            rows.append(row)
            print(f"[{index}/{len(paths)}] {path.name}: {status}")
        except Exception as exc:
            rows.append({"image": path.name, "status": f"error_{type(exc).__name__}"})
            print(f"[ERROR] {path.name}: {exc}")
    with (args.out / "plate_roi_metadata.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "status", "cx", "cy", "radius",
            "crop_left", "crop_top", "crop_right", "crop_bottom"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved previews and metadata to:", args.out)
    print("Inspect overlays before using crops as training data!")


if __name__ == "__main__":
    main()
