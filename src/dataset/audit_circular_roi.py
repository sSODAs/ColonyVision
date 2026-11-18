"""Audit image-only circular ROI estimates against ADBC labels (verification only).
Never uses labels to fit a circle or train; stores diagnostic summary under outputs.
"""
from pathlib import Path
from PIL import Image, ImageOps
import csv, sys, zipfile, math
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from preprocessing.plate_roi import estimate_plate
from preprocessing.plate_hough import estimate_plate_hough
raw=ROOT/"raw"/"external"/"ADBC"/"images"
zip_path=ROOT/"raw"/"external"/"ADBC"/"annot_YOLO.zip"
out=ROOT/"outputs"/"roi_circle_audit"
out.mkdir(parents=True,exist_ok=True)
rows=[]
with zipfile.ZipFile(zip_path) as z:
    files=sorted(raw.glob("*.jpg"))
    for i,p in enumerate(files,1):
        with Image.open(p) as im:
            image=ImageOps.exif_transpose(im).convert("RGB")
        plate,status=estimate_plate(image)
        circle=estimate_plate_hough(image)
        coordinates=[list(map(float,line.split())) for line in z.read(p.stem+".txt").decode("utf-8-sig").splitlines() if line.strip()]
        cx=cy=r=0
        if plate: cx,cy,r,_,_=plate
        total=len(coordinates)
        excluded=0
        if plate:
            for _,x,y,_,_ in coordinates:
                px=x*image.width;py=y*image.height
                if (px-cx)**2+(py-cy)**2 > (r*0.98)**2:excluded+=1
        hough_outside=0
        if circle:
            xhc,yhc,rhc,_=circle
            for _,x,y,_,_ in coordinates:
                px=x*image.width;py=y*image.height
                if (px-xhc)**2+(py-yhc)**2 > (rhc*1.02)**2:hough_outside+=1
        rows.append([p.name,status,total,excluded,image.width,image.height,
                     round(cx),round(cy),round(r),
                     bool(circle),hough_outside])
        if i%80==0:print("Checked",i,flush=True)
with (out/"audit.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f);w.writerow(["image","roi_status","gt_count","gt_centers_outside_98pct","width","height","cx","cy","r","hough_found","hough_gt_outside_102pct"]);w.writerows(rows)
success=[r for r in rows if r[1]=="estimated_not_verified"]
print("TOTAL",len(rows),"ROI_ESTIMATED",len(success),"ROI_FAILED",len(rows)-len(success),
      "GT_OUTSIDE",sum(r[3] for r in success),"GT_TOTAL",sum(r[2] for r in success),
      "ESTIMATES_WITH_OUTSIDE",sum(r[3]>0 for r in success),flush=True)
hough_rows=[r for r in rows if r[-2]]
print("HOUGH_FOUND",len(hough_rows),"HOUGH_FAILED",len(rows)-len(hough_rows),
      "GT_OUTSIDE_HOUGH_102%",sum(r[-1] for r in hough_rows),
      "GT_TOTAL",sum(r[2] for r in hough_rows),
      "HOUGH_WITH_OUTSIDE",sum(r[-1]>0 for r in hough_rows),flush=True)
for r in rows[:12]:print(r,flush=True)
