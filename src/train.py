"""Train one-class YOLO11 colony detector with a prepared labeled dataset."""
import argparse
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=ROOT/"dataset"/"adbc_full"/"data.yaml")
    ap.add_argument("--model",default="yolo11n.pt")
    ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--batch",type=int,default=2)
    ap.add_argument("--workers",type=int,default=0)
    ap.add_argument("--name",default="adbc_yolo11n")
    ap.add_argument("--device",default="0")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--patience",type=int,default=15)
    args=ap.parse_args()
    data=args.data if args.data.is_absolute() else (ROOT/args.data)
    if not data.is_file():
        raise SystemExit("Dataset YAML not found: "+str(data))
    metadata=yaml.safe_load(data.read_text(encoding="utf-8-sig"))
    if metadata.get("names") not in ({0:"colony"},{"0":"colony"}):
        raise RuntimeError("Expected exactly one class: colony")
    base=Path(metadata.get("path") or data.parent)
    if not base.is_absolute():base=(data.parent/base).resolve()
    for split in ("train","val"):
        value=metadata.get(split)
        folder=base/value
        if not folder.is_dir() or not any(p.suffix.lower() in (".jpg",".jpeg",".png") for p in folder.iterdir()):
            raise RuntimeError("Missing "+split+" images: "+str(folder))
    if args.device!="cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for limited tests")
    device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("Using:",device_name,"| dataset:",data,"| model:",args.model,flush=True)
    model=YOLO(args.model)
    model.train(data=str(data),epochs=args.epochs,imgsz=args.imgsz,
                batch=args.batch,device=int(args.device) if args.device.isdigit() else args.device,
                workers=args.workers,amp=args.device!="cpu",
                cache=False,seed=42,patience=args.patience,
                plots=True,project=str(ROOT/"outputs"/"train"),name=args.name,
                exist_ok=True,resume=args.resume)

if __name__=="__main__":
    main()
