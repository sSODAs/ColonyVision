"""Preview original versus Hough circular-mask ADBC training input."""
import csv,sys
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from preprocessing.plate_hough import estimate_plate_hough
manifest=ROOT/"dataset"/"adbc_circular"/"roi_audit.csv"
records=list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
picked=[]
used=set()
for row in records:
    group=row["original"].split("_")[0]
    if row["roi_action"]=="masked" and group not in used:
        picked.append(row)
        used.add(group)
    if len(picked)==4:break
width=1050;panel=500;row_h=565
board=Image.new("RGB",(width,max(1,len(picked))*row_h),"#23272e")
p=ImageDraw.Draw(board)
for i,row in enumerate(picked):
    with Image.open(ROOT/"raw"/"external"/"ADBC"/"images"/row["original"]) as f:
        original=ImageOps.exif_transpose(f).convert("RGB")
    geometry=estimate_plate_hough(original)
    if geometry is None:continue
    cx,cy,r,_=geometry
    mr=r*1.035
    circle=Image.new("L",original.size,0)
    ImageDraw.Draw(circle).ellipse((cx-mr,cy-mr,cx+mr,cy+mr),fill=255)
    masked=Image.composite(original,Image.new("RGB",original.size,"black"),circle)
    original.thumbnail((panel,panel),Image.Resampling.LANCZOS)
    masked.thumbnail((panel,panel),Image.Resampling.LANCZOS)
    board.paste(original,(16,i*row_h+18))
    board.paste(masked,(535,i*row_h+18))
    p.text((18,i*row_h+524),row["original"]+" / original",fill="white")
    p.text((536,i*row_h+524),"circular masked for YOLO",fill="#d5e6f5")
dest=ROOT/"outputs"/"roi_hough_experiment"/"circular_before_after.jpg"
dest.parent.mkdir(parents=True,exist_ok=True)
board.save(dest,quality=89)
print("SAVED",dest)
