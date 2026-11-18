"""Prepare upright EXIF-free copies for WeCFU Ultra without touching originals."""
import argparse
import csv
from pathlib import Path
from PIL import Image, ImageOps
ROOT = Path(__file__).resolve().parents[2]
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT / "raw" / "pilot_review")
    ap.add_argument("--out", type=Path, default=ROOT / "raw" / "wecfu_ready")
    args = ap.parse_args()
    files = sorted(p for p in args.source.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit("No images found")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(files, 1):
        with Image.open(path) as original:
            orientation = original.getexif().get(274, 1)
            original_size = original.size
            upright = ImageOps.exif_transpose(original).convert("RGB")
            dest = args.out / (path.stem + ".jpg")
            if dest.exists():
                raise SystemExit(f"Target already exists; not overwriting: {dest}")
            upright.save(dest, format="JPEG", quality=95, subsampling=0)
            rows.append([path.name, dest.name, orientation, *original_size, *upright.size, "EXIF_removed"])
        print(f"[{i}/{len(files)}] {path.name} -> {dest.name} (EXIF orientation={orientation})")
    report = args.out / "transform_metadata.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source_name","ready_name","source_exif_orientation","source_w","source_h",
                    "upright_w","upright_h","transform_note"])
        w.writerows(rows)
    print("Upright copies:",len(rows), "Metadata:",report)

if __name__ == "__main__":
    main()
