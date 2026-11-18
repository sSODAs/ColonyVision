"""Count one-class colonies using a trained checkpoint (not COCO weights)."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
        default=ROOT / "outputs" / "train" / "baseline" / "weights" / "best.pt")
    parser.add_argument("--source", type=Path,
        default=ROOT / "raw" / "unlabeled" / "Petri_plates")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--limit", type=int, default=12,
        help="Preview only 12 images by default; 0 to run all")
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Trained colony checkpoint missing: {args.model} (train first)")
    if args.source.is_file():
        images = [args.source]
    else:
        images = sorted(p for p in args.source.iterdir()
                        if p.suffix.lower() in EXTS)
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit("No input images")

    import torch
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    device = 0 if torch.cuda.is_available() else "cpu"
    output = ROOT / "outputs" / "predict" / args.name
    output.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=[str(p) for p in images],
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        max_det=3000,
        save=True,
        stream=True,
        project=str(ROOT / "outputs" / "predict"),
        name=args.name,
        exist_ok=True,
    )
    with (output / "counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "predicted_count"])
        for result in results:
            count = len(result.boxes) if result.boxes is not None else 0
            name = Path(result.path).name
            writer.writerow([name, count])
            print(f"{name}: {count} colonies")
    print("Saved:", output)

if __name__ == "__main__":
    main()
