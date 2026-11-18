"""High-resolution tiled colony inference with cross-tile NMS.
All results are unverified predictions, not ground truth.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image,ImageOps,ImageDraw
from ultralytics import YOLO
from preprocessing.plate_roi import estimate_plate

ROOT=Path(__file__).resolve().parents[1]
EXTS={".jpg",".jpeg",".png",".tif",".tiff"}

def positions(length,tile,stride):
    if length<=tile:return [0]
    xs=list(range(0,length-tile+1,stride))
    if xs[-1]!=length-tile:xs.append(length-tile)
    return xs

def infer(model,image,tile,stride,conf,imgsz,iou,device):
    w,h=image.size
    candidates=[]
    for top in positions(h,tile,stride):
        for left in positions(w,tile,stride):
            crop=np.asarray(image.crop((left,top,min(w,left+tile),min(h,top+tile))))
            result=model.predict(source=crop,device=device,conf=conf,
                                 imgsz=imgsz,max_det=2000,verbose=False)[0]
            if result.boxes is None:continue
            boxes=result.boxes.xyxy.cpu().numpy()
            scores=result.boxes.conf.cpu().numpy()
            for b,score in zip(boxes,scores):
                x0,y0,x1,y1=(float(v) for v in b)
                x0=max(0,min(w,x0+left));x1=max(0,min(w,x1+left))
                y0=max(0,min(h,y0+top));y1=max(0,min(h,y1+top))
                if x1-x0<2 or y1-y0<2:continue
                candidates.append([x0,y0,x1,y1,float(score)])
    if not candidates:return [],0
    # Global NMS avoids recounting the same colony in adjacent tiles.
    boxes=torch.tensor([c[:4] for c in candidates],dtype=torch.float32)
    scores=torch.tensor([c[4] for c in candidates],dtype=torch.float32)
    keep=torchvision.ops.nms(boxes,scores,iou).tolist()
    kept=[candidates[i] for i in keep]
    return kept,len(candidates)

def annotate(image,marks):
    result=image.copy().convert("RGB")
    draw=ImageDraw.Draw(result)
    width=max(2,int(min(image.size)/900))
    for x0,y0,x1,y1,score in marks:
        draw.rectangle((x0,y0,x1,y1),outline="#15eb72",width=width)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,default=ROOT/"outputs"/"train"/"adbc_yolo11n"/"weights"/"best.pt")
    p.add_argument("--source",type=Path,default=ROOT/"raw"/"unlabeled"/"Petri_plates")
    p.add_argument("--name",default="adbc_tiled_preview")
    p.add_argument("--tile",type=int,default=768)
    p.add_argument("--stride",type=int,default=640)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--conf",type=float,default=0.25)
    p.add_argument("--iou",type=float,default=0.60)
    p.add_argument("--limit",type=int,default=12,help="0 means all source images")
    p.add_argument("--plate-inset",type=float,default=0.96,help="Retain detections within this fraction of estimated plate radius")
    p.add_argument("--no-plate-filter",action="store_true",help="Disable estimated circular plate filtering")
    p.add_argument("--circular-mask",action="store_true",help="Set pixels outside estimated dish to black before YOLO")
    p.add_argument("--strict-roi",action="store_true",help="If dish localization fails, skip counting (do not return misleading zero)")
    args=p.parse_args()
    if not args.model.is_file():raise SystemExit("Checkpoint missing: "+str(args.model))
    if not 0<args.stride<=args.tile:raise SystemExit("0 < stride <= tile is required")
    if args.no_plate_filter and (args.circular_mask or args.strict_roi):
        raise SystemExit("--no-plate-filter conflicts with --circular-mask or --strict-roi")
    if args.source.is_file():
        files=[args.source]
    else:
        files=sorted(p for p in args.source.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    if args.limit>0:files=files[:args.limit]
    if not files:raise SystemExit("No input images")
    device=0 if torch.cuda.is_available() else "cpu"
    model=YOLO(str(args.model))
    out=ROOT/"outputs"/"predict"/args.name
    (out/"overlays").mkdir(parents=True,exist_ok=True)
    (out/"detections").mkdir(parents=True,exist_ok=True)
    rows=[]
    for n,path in enumerate(files,1):
        with Image.open(path) as image:
            upright=ImageOps.exif_transpose(image).convert("RGB")
        plate,plate_status=estimate_plate(upright) if not args.no_plate_filter else (None,"filter_disabled")
        if plate is None and args.strict_roi:
            failed={"source":path.name,"status":"roi_not_found_no_prediction",
                    "plate_status":plate_status,"count":None,"boxes_xyxy_conf":[]}
            (out/"detections"/(path.name+".json")).write_text(
                json.dumps(failed,indent=2),encoding="utf-8")
            rows.append([path.name,"",""])
            print(f"[{n}/{len(files)}] {path.name}: ROI not found; SKIPPED (not zero colonies)",flush=True)
            continue
        if plate is not None:
            cx,cy,r,_,_=plate
            roi_size=1.06*r
            xlo=max(0,int(cx-roi_size)); ylo=max(0,int(cy-roi_size))
            xhi=min(upright.width,int(cx+roi_size)); yhi=min(upright.height,int(cy+roi_size))
            working=upright.crop((xlo,ylo,xhi,yhi))
        else:
            working=upright
            xlo=ylo=0
        if args.circular_mask and plate is not None:
            # Same safety-margin convention as the ADBC circular train variant.
            localx=cx-xlo;localy=cy-ylo
            mr=r*1.035
            mask=Image.new("L",working.size,0)
            ImageDraw.Draw(mask).ellipse((localx-mr,localy-mr,localx+mr,localy+mr),fill=255)
            working=Image.composite(working,Image.new("RGB",working.size,(0,0,0)),mask)
        candidates,raw=infer(model,working,args.tile,args.stride,args.conf,args.imgsz,args.iou,device)
        detections=[]
        for x0,y0,x1,y1,score in candidates:
            mapped=[x0+xlo,y0+ylo,x1+xlo,y1+ylo,score]
            if plate is not None:
                midx=(mapped[0]+mapped[2])/2;midy=(mapped[1]+mapped[3])/2
                if (midx-cx)**2+(midy-cy)**2 > (r*args.plate_inset)**2:
                    continue
            detections.append(mapped)
        annotated=annotate(upright,detections)
        if plate is not None:
            drawing=ImageDraw.Draw(annotated)
            drawing.ellipse((cx-r*args.plate_inset,cy-r*args.plate_inset,
                             cx+r*args.plate_inset,cy+r*args.plate_inset),
                            outline="#ffb000",width=5)
        annotated.save(out/"overlays"/(path.stem+"_marked.jpg"),quality=92)
        record={"source":path.name,"orientation":"EXIF_transposed_for_display",
                "image_size":{"width":upright.width,"height":upright.height},
                "model":str(args.model),"source_pseudo_label":False,
                "plate_status":plate_status,
                "plate_estimate":list(plate[:3]) if plate is not None else None,
                "plate_inset":args.plate_inset if plate is not None else None,
                "circular_mask_applied":bool(args.circular_mask and plate is not None),
                "boxes_xyxy_conf":detections,"count":len(detections)}
        (out/"detections"/(path.name+".json")).write_text(
            json.dumps(record,indent=2),encoding="utf-8")
        rows.append([path.name,len(detections),raw])
        print(f"[{n}/{len(files)}] {path.name}: {len(detections)} colonies ({raw} before NMS)",flush=True)
    with (out/"counts.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f);writer.writerow(["image","predicted_count","raw_boxes_before_global_nms"])
        writer.writerows(rows)
    print("Saved:",out,flush=True)

if __name__=="__main__":main()
