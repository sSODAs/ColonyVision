"""Convert external model's pixel XYXY CSV to *unreviewed* single-class YOLO labels.

Expected CSV: image,x_min,y_min,x_max,y_max[,confidence]
Do NOT copy these candidate labels into dataset/train before manual review.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "raw" / "unlabeled" / "Petri_plates"
OUT = ROOT / "annotations" / "pseudo_labels" / "external_xyxy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_index = {p.name: p for p in args.images.iterdir() if p.is_file()}
    boxes = defaultdict(list)
    rejected = []
    with args.csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "x_min", "y_min", "x_max", "y_max"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f"CSV requires columns: {sorted(required)}")
        for num, row in enumerate(reader, 2):
            name = Path(row["image"]).name
            if name not in image_index:
                rejected.append((num, "image not found", name))
                continue
            try:
                confidence = float(row.get("confidence") or 1)
                xyxy = tuple(float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
                if not all(math.isfinite(x) for x in (*xyxy, confidence)):
                    raise ValueError("non-finite value")
                if confidence < args.min_conf:
                    continue
                if xyxy[0] >= xyxy[2] or xyxy[1] >= xyxy[3]:
                    raise ValueError("invalid box order")
                boxes[name].append(xyxy)
            except (ValueError, TypeError) as exc:
                rejected.append((num, str(exc), name))

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, rects in boxes.items():
        with Image.open(image_index[name]) as raw:
            w, h = ImageOps.exif_transpose(raw).size
        labels = []
        for x0,y0,x1,y1 in rects:
            x0,x1 = max(0,x0),min(w,x1)
            y0,y1 = max(0,y0),min(h,y1)
            if x0 >= x1 or y0 >= y1:
                continue
            labels.append(f"0 {(x0+x1)/(2*w):.8f} {(y0+y1)/(2*h):.8f} {(x1-x0)/w:.8f} {(y1-y0)/h:.8f}")
        target = args.out / (Path(name).stem + ".txt")
        if target.exists() and not args.overwrite:
            raise SystemExit(f"Output already exists; use a new --out or --overwrite: {target}")
        target.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        written += 1

    print(f"Converted {written} image labels to: {args.out}")
    print(f"Rejected {len(rejected)} rows")
    for item in rejected[:15]:
        print("REJECTED:", *item)
    print("STATUS: PSEUDO ONLY — MUST REVIEW BEFORE TRAIN OR TEST")


if __name__ == "__main__":
    main()
