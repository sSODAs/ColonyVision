"""Validate one-class YOLO bbox labels. Refuse empty/unlabeled training data."""
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
errors = []
total_images = 0
total_boxes = 0

for split in ("train", "val", "test"):
    image_dir = ROOT / "dataset" / "images" / split
    label_dir = ROOT / "dataset" / "labels" / split
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTS)
    labels = {p.stem: p for p in label_dir.glob("*.txt")}
    boxes = 0
    for image in images:
        label = labels.get(image.stem)
        if label is None:
            errors.append(f"{split}/{image.name}: Missing .txt label (empty allowed for true negatives)")
            continue
        for n, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            try:
                values = [float(s) for s in line.split()]
                if len(values) != 5:
                    raise ValueError("expected class + 4 normalized coordinates")
                cls, x, y, w, h = values
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("non-finite coordinate")
                if cls != 0 or not (
                    0 <= x <= 1 and 0 <= y <= 1 and
                    0 < w <= 1 and 0 < h <= 1 and
                    x-w/2 >= -1e-5 and x+w/2 <= 1+1e-5 and
                    y-h/2 >= -1e-5 and y+h/2 <= 1+1e-5
                ):
                    raise ValueError("invalid class / box outside image")
                boxes += 1
            except ValueError as e:
                errors.append(f"{split}/{label.name}:{n}: {e}")
    extra = sorted(set(labels) - {p.stem for p in images})
    for stem in extra[:20]:
        errors.append(f"{split}: label without image: {stem}.txt")
    print(f"{split}: {len(images)} images, {boxes} annotated colonies")
    if split in ("train", "val") and (len(images) == 0 or boxes == 0):
        errors.append(f"{split}: needs at least one labeled positive image before training")
    total_images += len(images)
    total_boxes += boxes

print(f"TOTAL: {total_images} images / {total_boxes} colonies")
if errors:
    for e in errors[:35]:
        print("ERROR:", e)
    raise SystemExit(f"Dataset validation failed ({len(errors)} issues)")
print("PASS: Dataset has labeled training and validation examples")
