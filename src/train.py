import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    data = ROOT / "dataset" / "data.yaml"

    for split in ("train", "val"):
        images = ROOT / "dataset" / "images" / split
        if not any(
            p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for p in images.iterdir()
        ):
            raise RuntimeError(f"No images in {images}")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)

    model = YOLO(args.model)

    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=0,
        amp=True,
        cache=False,
        seed=42,
        patience=20,
        plots=True,
        project=str(ROOT / "outputs" / "train"),
        name=args.name
    )

if __name__ == "__main__":
    main()
